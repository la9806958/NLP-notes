#!/usr/bin/env python3
"""

Earnings Call Qwen Embeddings Extraction Pipeline

(1) data loading and target construction; 

(2) NER masking; 

(3) FinBERT teacher fine-tuning and soft-label generation; 

(4) Qwen student fine-tuning with distillation and ordinal loss, 

(5) Embedding extraction and a light regression head for OOS evaluation

"""

import os
import gc
import json
import pickle
import warnings
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from multiprocessing import Pool, cpu_count
from functools import partial

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from scipy.stats import spearmanr, pearsonr
import spacy
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Transformers and related
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModel,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup, BitsAndBytesConfig
)
from transformers.trainer_utils import EvalPrediction
from datasets import Dataset as HFDataset
from peft import LoraConfig, get_peft_model, TaskType
import evaluate

warnings.filterwarnings('ignore')


class CORALLoss(nn.Module):
    """CORAL (COnsistent RAnk Logits) Loss for ordinal regression

    Predicts K-1 boundary logits z_k ≈ P(y > k) and applies BCE.
    This enforces monotonic cumulative probabilities and respects ordinal structure.

    Reference: Cao, Mirjalili, Raschka (2020) - Rank consistent ordinal regression
    """

    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.Km1 = num_classes - 1
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, boundary_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            boundary_logits: (batch_size, K-1) - boundary logits z_k for P(y > k)
            targets: (batch_size,) - true class labels in {0, 1, ..., K-1}

        Returns:
            loss: scalar tensor
        """
        B = targets.size(0)

        # Create ordinal one-vs-threshold matrix
        # For 3-class: target=0 → [0, 0], target=1 → [1, 0], target=2 → [1, 1]
        thresholds = torch.arange(self.Km1, device=targets.device, dtype=targets.dtype)
        thresholds = thresholds.unsqueeze(0).expand(B, -1)  # [B, K-1]
        ordinal_targets = (targets.unsqueeze(1) > thresholds).float()  # [B, K-1]

        return self.bce(boundary_logits, ordinal_targets)

    def predict_proba(self, boundary_logits: torch.Tensor) -> torch.Tensor:
        """Convert boundary logits to class probabilities

        Args:
            boundary_logits: (batch_size, K-1) boundary logits

        Returns:
            probs: (batch_size, K) class probabilities
        """
        # Apply sigmoid to get cumulative probabilities P(y > k)
        cum_probs = torch.sigmoid(boundary_logits)  # [B, K-1]

        # Convert to class probabilities
        # P(y=0) = 1 - P(y>0) = 1 - σ(z_0)
        # P(y=k) = P(y>k-1) - P(y>k) = σ(z_{k-1}) - σ(z_k) for k=1..K-2
        # P(y=K-1) = P(y>K-2) = σ(z_{K-2})

        probs = []
        # First class
        probs.append(1.0 - cum_probs[:, 0:1])

        # Middle classes
        for k in range(self.Km1 - 1):
            probs.append(cum_probs[:, k:k+1] - cum_probs[:, k+1:k+2])

        # Last class
        probs.append(cum_probs[:, -1:])

        return torch.cat(probs, dim=1)


@dataclass
class PipelineConfig:
    """Configuration for the OOS embeddings extraction pipeline"""
    # Data paths
    earnings_csv_path: str = "data/earnings_calls_with_analyst_changes.csv"
    analysis_csv_path: str = "data/earnings_analysis_50q_results_pre_2023.csv"
    trained_model_dir: str = "earnings_qwen_analyst_target_output/qwen_checkpoints"
    output_dir: str = "earnings_qwen_oos_embeddings_output"

    # NER settings
    spacy_model: str = "en_core_web_sm"
    mask_persons: bool = True
    mask_orgs: bool = True
    person_mask_token: str = "[PERSON]"
    org_mask_token: str = "[ORG]"
    num_workers: int = 10  # Number of parallel workers for NER

    # Target column
    target_column: str = "analyst_target_pct_change"
    num_quantile_classes: int = 3

    # Qwen model settings - L4 optimized
    qwen_model: str = "Qwen/Qwen2-1.5B"  # Smaller model for L4 GPU
    max_length: int = 512  # Reduced for memory efficiency
    num_labels: int = 3

    # Batch size for embedding extraction
    batch_size: int = 8

    # LoRA settings for parameter-efficient fine-tuning
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Quantization for memory efficiency
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"

    # Ordinal loss settings
    use_ordinal_loss: bool = True  # Use ordinal cross-entropy

    # Other settings
    seed: int = 42
    save_embeddings: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EarningsDataLoader:
    """Handles loading and preprocessing of earnings call data"""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def load_data(self, oos: bool = False) -> pd.DataFrame:
        """Load and join earnings calls and analysis CSV data

        Args:
            oos: If True, load data from 2023 onwards (OOS). If False, load data before 2023 (training).
        """
        date_filter = ">= 2023-01-01" if oos else "< 2023-01-01"
        print(f"Loading {'OOS' if oos else 'training'} data (et_timestamp {date_filter})")

        # Load earnings CSV
        print(f"Loading earnings data from {self.config.earnings_csv_path}")
        df_earnings = pd.read_csv(self.config.earnings_csv_path)
        print(f"Loaded {len(df_earnings)} earnings records")

        # Load analysis CSV
        print(f"Loading analysis data from {self.config.analysis_csv_path}")
        df_analysis = pd.read_csv(self.config.analysis_csv_path)
        print(f"Loaded {len(df_analysis)} analysis records")

        # Convert timestamps
        df_earnings['et_timestamp'] = pd.to_datetime(df_earnings['et_timestamp'], errors='coerce')
        df_analysis['et_timestamp'] = pd.to_datetime(df_analysis['et_timestamp'], errors='coerce')

        # Join the dataframes
        print("\nJoining earnings and analysis data on et_timestamp and ticker...")
        df_joined = pd.merge(
            df_earnings,
            df_analysis[['ticker', 'et_timestamp', 'analysis']],
            on=['ticker', 'et_timestamp'],
            how='inner'
        )
        print(f"After earnings-analysis join: {len(df_joined)} records")

        # Basic filtering - only keep rows with analysis text
        df_joined = df_joined.dropna(subset=['analysis'])
        print(f"After removing missing analysis: {len(df_joined)} records")

        # Filter out very short texts (less than 100 chars)
        df_joined = df_joined[df_joined['analysis'].str.len() >= 100]
        print(f"After filtering short texts (<100 chars): {len(df_joined)} records")

        # Filter for et_timestamp
        if oos:
            df_joined = df_joined[df_joined['et_timestamp'] >= '2023-01-01']
        else:
            df_joined = df_joined[df_joined['et_timestamp'] < '2023-01-01']

        print(f"After timestamp filter ({date_filter}): {len(df_joined)} records")

        if len(df_joined) == 0:
            raise ValueError(f"No {'OOS' if oos else 'training'} data available after filtering")

        return df_joined


def _mask_entities_worker(text: str, spacy_model: str, mask_persons: bool,
                          mask_orgs: bool, person_mask_token: str, org_mask_token: str) -> str:
    """Worker function for parallel NER masking (must be top-level for pickling)"""
    if not isinstance(text, str) or len(text.strip()) == 0:
        return text

    # Load spaCy model in each worker process
    nlp = spacy.load(spacy_model, disable=["lemmatizer", "textcat"])
    doc = nlp(text)

    # Collect entities to replace (in reverse order to maintain positions)
    replacements = []
    for ent in doc.ents:
        if (mask_persons and ent.label_ == "PERSON"):
            replacements.append((ent.start_char, ent.end_char, person_mask_token))
        elif (mask_orgs and ent.label_ in ["ORG", "GPE"]):  # GPE for geo-political entities
            replacements.append((ent.start_char, ent.end_char, org_mask_token))

    # Apply replacements in reverse order
    masked_text = text
    for start, end, replacement in sorted(replacements, key=lambda x: x[0], reverse=True):
        masked_text = masked_text[:start] + replacement + masked_text[end:]

    return masked_text


class NERMasker:
    """Handles Named Entity Recognition and masking with parallel processing"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        print(f"Loading spaCy model: {config.spacy_model}")
        print(f"Using {config.num_workers} workers for parallel NER processing")

        # Verify spaCy model is available
        try:
            nlp = spacy.load(config.spacy_model)
            del nlp  # Don't need to keep it loaded in main process
        except OSError:
            print(f"spaCy model '{config.spacy_model}' not found. Please install it:")
            print(f"python -m spacy download {config.spacy_model}")
            raise

    def mask_entities(self, text: str) -> str:
        """Apply NER masking to single text (for backward compatibility)"""
        return _mask_entities_worker(
            text,
            self.config.spacy_model,
            self.config.mask_persons,
            self.config.mask_orgs,
            self.config.person_mask_token,
            self.config.org_mask_token
        )

    def process_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts with parallel NER masking"""
        print(f"Processing {len(texts)} texts with {self.config.num_workers} workers")

        # Create partial function with fixed parameters
        worker_func = partial(
            _mask_entities_worker,
            spacy_model=self.config.spacy_model,
            mask_persons=self.config.mask_persons,
            mask_orgs=self.config.mask_orgs,
            person_mask_token=self.config.person_mask_token,
            org_mask_token=self.config.org_mask_token
        )

        # Use multiprocessing pool for parallel processing
        with Pool(processes=self.config.num_workers) as pool:
            # Use imap for progress bar support
            masked_texts = list(tqdm(
                pool.imap(worker_func, texts, chunksize=max(1, len(texts) // (self.config.num_workers * 4))),
                total=len(texts),
                desc="Applying NER masking (parallel)"
            ))

        return masked_texts


class QwenEmbeddingExtractor:
    """Handles loading a trained Qwen model and extracting embeddings"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.load_model()

    def load_model(self):
        """Load the fine-tuned Qwen model"""
        print(f"Loading fine-tuned Qwen model from: {self.config.trained_model_dir}")

        # Quantization config for L4 GPU memory efficiency
        if self.config.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.trained_model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        base_model = AutoModel.from_pretrained(
            self.config.qwen_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.use_4bit else torch.float32
        )

        # Setup LoRA for parameter-efficient fine-tuning
        if self.config.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            base_model = get_peft_model(base_model, lora_config)

        # Load trained weights
        # Custom model class for classification (same as training)
        class QwenClassifier(nn.Module):
            def __init__(self, base_model, num_labels, config):
                super().__init__()
                self.base_model = base_model
                self.num_labels = num_labels
                self.config = config

                # Output K-1 boundary logits if ordinal, else K class logits
                output_dim = num_labels - 1 if config.use_ordinal_loss else num_labels
                self.classifier = nn.Linear(base_model.config.hidden_size, output_dim)

                # Ensure head is on same device as base model (critical with device_map="auto")
                try:
                    base_device = next(self.base_model.parameters()).device
                except StopIteration:
                    base_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.classifier.to(base_device)

                # Setup loss function
                if config.use_ordinal_loss:
                    self.loss_fct = CORALLoss(num_classes=num_labels)
                else:
                    self.loss_fct = nn.CrossEntropyLoss()

                # For ordinal, keep a CORAL loss instance for prediction
                if config.use_ordinal_loss:
                    self.coral_loss = CORALLoss(num_classes=num_labels)

            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

                # Use last hidden state of the last token (or mean pooling)
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_state = outputs.last_hidden_state
                    # Mean pooling over sequence length
                    pooled_output = hidden_state.mean(dim=1)
                else:
                    pooled_output = outputs[0].mean(dim=1)

                logits = self.classifier(pooled_output)

                loss = None
                if labels is not None:
                    loss = self.loss_fct(logits, labels)

                # For ordinal, convert boundary logits to class probs for metrics
                if self.config.use_ordinal_loss:
                    # Convert K-1 boundary logits to K class logits (for compatibility)
                    with torch.no_grad():
                        class_probs = self.coral_loss.predict_proba(logits)
                        # Convert probs back to pseudo-logits for compatibility
                        class_logits = torch.log(class_probs + 1e-8)
                    return {"loss": loss, "logits": class_logits, "boundary_logits": logits}
                else:
                    return {"loss": loss, "logits": logits}

        # Wrap the model
        self.model = QwenClassifier(base_model, self.config.num_labels, self.config)

        # Load the trained weights
        checkpoint_path = os.path.join(self.config.trained_model_dir, "pytorch_model.bin")
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.config.device)
            self.model.load_state_dict(state_dict)
            print(f"Loaded trained model weights from {checkpoint_path}")
        else:
            print(f"Warning: No trained weights found at {checkpoint_path}. Using base model.")

        self.model.eval()
        print(f"Model loaded on {self.model.device if hasattr(self.model, 'device') else 'auto'}")

    def extract_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings from the fine-tuned model"""
        print("Extracting embeddings from fine-tuned model...")

        self.model.eval()
        embeddings = []

        batch_size = self.config.batch_size
        for i in tqdm(range(0, len(texts), batch_size), desc="Extracting embeddings"):
            batch_texts = texts[i:i + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            )

            # Move inputs to the same device as the model
            try:
                model_device = next(self.model.parameters()).device
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            except StopIteration:
                # Fallback if model has no parameters
                inputs = {k: v.to(self.config.device) for k, v in inputs.items()}

            with torch.no_grad():
                if hasattr(self.model, 'base_model'):
                    outputs = self.model.base_model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )
                else:
                    outputs = self.model(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask']
                    )

                # Extract embeddings (mean pooling)
                if hasattr(outputs, 'last_hidden_state'):
                    hidden_states = outputs.last_hidden_state
                else:
                    hidden_states = outputs[0]

                # Mean pooling over sequence length
                batch_embeddings = hidden_states.mean(dim=1).cpu().numpy()
                embeddings.append(batch_embeddings)

            del inputs, outputs, hidden_states
            torch.cuda.empty_cache()

        return np.vstack(embeddings)


class OOSEmbeddingsPipeline:
    """Main pipeline orchestrator for embeddings extraction (both in-sample and OOS)"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_output_dir()

    def setup_output_dir(self):
        """Create output directory"""
        os.makedirs(self.config.output_dir, exist_ok=True)

    def process_data_split(self, df: pd.DataFrame, split_name: str, ner_masker: NERMasker,
                          extractor: QwenEmbeddingExtractor) -> Tuple[np.ndarray, pd.DataFrame]:
        """Process a single data split (in-sample or OOS)

        Args:
            df: DataFrame to process
            split_name: Name of the split ('in_sample' or 'oos')
            ner_masker: NER masker instance
            extractor: Embedding extractor instance

        Returns:
            Tuple of (embeddings array, dataframe with masked texts)
        """
        print(f"\nProcessing {split_name} data: {len(df)} records")

        # Show target statistics (including NaN)
        if self.config.target_column in df.columns:
            n_valid_targets = df[self.config.target_column].notna().sum()
            n_nan_targets = df[self.config.target_column].isna().sum()
            print(f"Target column '{self.config.target_column}' statistics:")
            print(f"  - Valid targets: {n_valid_targets} ({n_valid_targets/len(df)*100:.2f}%)")
            print(f"  - NaN targets: {n_nan_targets} ({n_nan_targets/len(df)*100:.2f}%)")

        # Apply NER masking
        print(f"Applying NER masking to {split_name} data...")
        masked_texts = ner_masker.process_batch(df['analysis'].tolist())
        print(f"Masked {len(masked_texts)} texts")

        # Extract embeddings
        print(f"Extracting embeddings for {split_name} data...")
        embeddings = extractor.extract_embeddings(masked_texts)
        print(f"Extracted embeddings shape: {embeddings.shape}")

        # Add masked texts to dataframe
        df_copy = df.copy()
        df_copy['masked_text'] = masked_texts

        return embeddings, df_copy

    def save_split_data(self, embeddings: np.ndarray, df: pd.DataFrame, split_name: str):
        """Save embeddings and metadata for a data split

        Args:
            embeddings: Embeddings array
            df: DataFrame with metadata and masked texts
            split_name: Name of the split ('in_sample' or 'oos')
        """
        print(f"\nSaving {split_name} data...")

        # Save all embeddings
        embeddings_path = os.path.join(self.config.output_dir, f"{split_name}_embeddings_all.npy")
        np.save(embeddings_path, embeddings)
        print(f"Embeddings saved to {embeddings_path}")

        # Save all metadata
        metadata_path = os.path.join(self.config.output_dir, f"{split_name}_metadata_all.csv")
        df.to_csv(metadata_path, index=False)
        print(f"Metadata saved to {metadata_path}")

        # Save separate files for records with valid targets
        if self.config.target_column in df.columns:
            df_with_targets = df[df[self.config.target_column].notna()].copy()
            if len(df_with_targets) > 0:
                metadata_with_targets_path = os.path.join(
                    self.config.output_dir, f"{split_name}_metadata_with_targets.csv"
                )
                df_with_targets.to_csv(metadata_with_targets_path, index=False)
                print(f"Metadata with valid targets saved to {metadata_with_targets_path}")

                # Save corresponding embeddings
                with_targets_indices = df[self.config.target_column].notna().values
                embeddings_with_targets = embeddings[with_targets_indices]
                embeddings_with_targets_path = os.path.join(
                    self.config.output_dir, f"{split_name}_embeddings_with_targets.npy"
                )
                np.save(embeddings_with_targets_path, embeddings_with_targets)
                print(f"Embeddings with valid targets saved to {embeddings_with_targets_path}")

    def run(self):
        """Execute the complete embeddings extraction pipeline for both in-sample and OOS data"""
        print("=" * 60)
        print("EARNINGS CALL QWEN - EMBEDDINGS EXTRACTION (ALL DATA)")
        print("=" * 60)

        # Initialize shared components
        data_loader = EarningsDataLoader(self.config)
        ner_masker = NERMasker(self.config)

        # Load model once for both splits
        print("\nLOADING FINE-TUNED MODEL")
        print("-" * 30)
        extractor = QwenEmbeddingExtractor(self.config)

        # Process in-sample data (pre-2023)
        print("\n" + "=" * 60)
        print("PROCESSING IN-SAMPLE DATA (pre-2023)")
        print("=" * 60)
        try:
            df_in_sample = data_loader.load_data(oos=False)
            embeddings_in_sample, df_in_sample_processed = self.process_data_split(
                df_in_sample, "in_sample", ner_masker, extractor
            )
            self.save_split_data(embeddings_in_sample, df_in_sample_processed, "in_sample")
        except ValueError as e:
            print(f"Warning: Could not process in-sample data: {e}")
            df_in_sample_processed = None
            embeddings_in_sample = None

        # Process OOS data (2023+)
        print("\n" + "=" * 60)
        print("PROCESSING OUT-OF-SAMPLE DATA (2023+)")
        print("=" * 60)
        try:
            df_oos = data_loader.load_data(oos=True)
            embeddings_oos, df_oos_processed = self.process_data_split(
                df_oos, "oos", ner_masker, extractor
            )
            self.save_split_data(embeddings_oos, df_oos_processed, "oos")
        except ValueError as e:
            print(f"Warning: Could not process OOS data: {e}")
            df_oos_processed = None
            embeddings_oos = None

        # Create summary report
        print("\n" + "=" * 60)
        print("CREATING SUMMARY REPORT")
        print("=" * 60)
        self.create_summary_report(
            df_in_sample_processed, embeddings_in_sample,
            df_oos_processed, embeddings_oos
        )

        # Save configuration
        config_path = os.path.join(self.config.output_dir, "pipeline_config.json")
        with open(config_path, 'w') as f:
            json.dump(vars(self.config), f, indent=2)
        print(f"Configuration saved to {config_path}")

        print("\n" + "=" * 60)
        print("EMBEDDINGS EXTRACTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    def create_summary_report(self, df_in_sample: Optional[pd.DataFrame],
                            embeddings_in_sample: Optional[np.ndarray],
                            df_oos: Optional[pd.DataFrame],
                            embeddings_oos: Optional[np.ndarray]):
        """Create a summary report of the embeddings extraction for both splits"""
        report_path = os.path.join(self.config.output_dir, "summary_report.txt")

        with open(report_path, 'w') as f:
            f.write("EMBEDDINGS EXTRACTION SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Overall statistics
            total_records = 0
            if df_in_sample is not None:
                total_records += len(df_in_sample)
            if df_oos is not None:
                total_records += len(df_oos)

            f.write(f"Total records processed: {total_records}\n")
            if df_in_sample is not None:
                f.write(f"  - In-sample (pre-2023): {len(df_in_sample)}\n")
            if df_oos is not None:
                f.write(f"  - Out-of-sample (2023+): {len(df_oos)}\n")
            f.write("\n")

            # In-sample statistics
            if df_in_sample is not None and embeddings_in_sample is not None:
                f.write("=" * 60 + "\n")
                f.write("IN-SAMPLE DATA (pre-2023)\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Total records: {len(df_in_sample)}\n")
                f.write(f"Embeddings shape: {embeddings_in_sample.shape}\n")
                f.write(f"Embedding dimension: {embeddings_in_sample.shape[1]}\n\n")

                f.write(f"Data coverage:\n")
                f.write(f"  - Date range: {df_in_sample['et_timestamp'].min()} to {df_in_sample['et_timestamp'].max()}\n")
                f.write(f"  - Unique tickers: {df_in_sample['ticker'].nunique()}\n")
                f.write(f"  - Average text length: {df_in_sample['analysis'].str.len().mean():.0f} characters\n\n")

                # Target statistics
                if self.config.target_column in df_in_sample.columns:
                    n_valid = df_in_sample[self.config.target_column].notna().sum()
                    n_nan = df_in_sample[self.config.target_column].isna().sum()
                    f.write(f"Target column: {self.config.target_column}\n")
                    f.write(f"  - Records with valid targets: {n_valid} ({n_valid/len(df_in_sample)*100:.2f}%)\n")
                    f.write(f"  - Records with NaN targets: {n_nan} ({n_nan/len(df_in_sample)*100:.2f}%)\n\n")

                    if n_valid > 0:
                        valid_targets = df_in_sample[self.config.target_column].dropna()
                        f.write(f"Valid target statistics:\n")
                        f.write(f"  - Mean: {valid_targets.mean():.4f}\n")
                        f.write(f"  - Std: {valid_targets.std():.4f}\n")
                        f.write(f"  - Min: {valid_targets.min():.4f}\n")
                        f.write(f"  - Max: {valid_targets.max():.4f}\n")
                        f.write(f"  - Median: {valid_targets.median():.4f}\n\n")

                f.write(f"Output files generated:\n")
                f.write(f"  - in_sample_embeddings_all.npy (all {len(embeddings_in_sample)} embeddings)\n")
                f.write(f"  - in_sample_metadata_all.csv (metadata for all records)\n")
                if self.config.target_column in df_in_sample.columns and df_in_sample[self.config.target_column].notna().sum() > 0:
                    f.write(f"  - in_sample_embeddings_with_targets.npy (embeddings with valid targets only)\n")
                    f.write(f"  - in_sample_metadata_with_targets.csv (metadata with valid targets only)\n")
                f.write("\n")

            # OOS statistics
            if df_oos is not None and embeddings_oos is not None:
                f.write("=" * 60 + "\n")
                f.write("OUT-OF-SAMPLE DATA (2023+)\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"Total records: {len(df_oos)}\n")
                f.write(f"Embeddings shape: {embeddings_oos.shape}\n")
                f.write(f"Embedding dimension: {embeddings_oos.shape[1]}\n\n")

                f.write(f"Data coverage:\n")
                f.write(f"  - Date range: {df_oos['et_timestamp'].min()} to {df_oos['et_timestamp'].max()}\n")
                f.write(f"  - Unique tickers: {df_oos['ticker'].nunique()}\n")
                f.write(f"  - Average text length: {df_oos['analysis'].str.len().mean():.0f} characters\n\n")

                # Target statistics
                if self.config.target_column in df_oos.columns:
                    n_valid = df_oos[self.config.target_column].notna().sum()
                    n_nan = df_oos[self.config.target_column].isna().sum()
                    f.write(f"Target column: {self.config.target_column}\n")
                    f.write(f"  - Records with valid targets: {n_valid} ({n_valid/len(df_oos)*100:.2f}%)\n")
                    f.write(f"  - Records with NaN targets: {n_nan} ({n_nan/len(df_oos)*100:.2f}%)\n\n")

                    if n_valid > 0:
                        valid_targets = df_oos[self.config.target_column].dropna()
                        f.write(f"Valid target statistics:\n")
                        f.write(f"  - Mean: {valid_targets.mean():.4f}\n")
                        f.write(f"  - Std: {valid_targets.std():.4f}\n")
                        f.write(f"  - Min: {valid_targets.min():.4f}\n")
                        f.write(f"  - Max: {valid_targets.max():.4f}\n")
                        f.write(f"  - Median: {valid_targets.median():.4f}\n\n")

                f.write(f"Output files generated:\n")
                f.write(f"  - oos_embeddings_all.npy (all {len(embeddings_oos)} embeddings)\n")
                f.write(f"  - oos_metadata_all.csv (metadata for all records)\n")
                if self.config.target_column in df_oos.columns and df_oos[self.config.target_column].notna().sum() > 0:
                    f.write(f"  - oos_embeddings_with_targets.npy (embeddings with valid targets only)\n")
                    f.write(f"  - oos_metadata_with_targets.csv (metadata with valid targets only)\n")
                f.write("\n")

            # General information
            f.write("=" * 60 + "\n")
            f.write("PIPELINE CONFIGURATION\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"  - pipeline_config.json (configuration)\n")
            f.write(f"  - summary_report.txt (this report)\n\n")

            f.write(f"Key features of this pipeline:\n")
            f.write(f"  - Extracted embeddings for ALL records (both in-sample and OOS)\n")
            f.write(f"  - Did NOT filter for NaN targets before extraction\n")
            f.write(f"  - Saved separate files for records with/without valid targets\n")
            f.write(f"  - Processed in-sample and OOS data separately\n")

        print(f"Summary report saved to {report_path}")


def main():
    """Main execution function"""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Create configuration
    config = PipelineConfig()

    # Verify trained model exists
    if not os.path.exists(config.trained_model_dir):
        print(f"Error: Trained model not found at {config.trained_model_dir}")
        print("Please run the original training pipeline first:")
        print("  python earnings_qwen_analyst_target_pipeline.py")
        return

    # Run pipeline
    pipeline = OOSEmbeddingsPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
