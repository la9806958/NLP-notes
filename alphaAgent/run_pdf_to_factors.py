#!/usr/bin/env python3
"""
PDF to Factors Pipeline

This script integrates the PDF hypothesis extraction pipeline with the FactorAgent
to create a complete pipeline from research papers to testable alpha factors.

Workflow:
1. Extract text from PDFs in /pdfs directory
2. Use LLM to extract market hypotheses
3. For each hypothesis, generate 25 alpha factors using FactorAgent
4. Evaluate factors on historical data
5. Save results and top-performing factors
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional
import argparse

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pdf_hypothesis_pipeline import (
    process_pdf_directory,
    get_all_hypotheses_text,
    save_hypotheses_summary
)
from alpha_agent_factor import FactorAgent, load_credentials

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_to_factors.log')
    ]
)
logger = logging.getLogger(__name__)


def run_pdf_to_factors_pipeline(
    pdf_dir: str = "/pdfs",
    data_path: str = "/home/lichenhui/data/1min",
    output_dir: str = "/home/lichenhui/data/alphaAgent/pdf_results",
    max_pdfs: Optional[int] = None,
    max_hypotheses: Optional[int] = None,
    model: str = "gpt-4o"
):
    """Run complete pipeline from PDFs to evaluated factors.

    Args:
        pdf_dir: Directory containing PDF research papers
        data_path: Directory containing market data (1-minute OHLCV CSVs)
        output_dir: Directory to save results
        max_pdfs: Maximum number of PDFs to process (None for all)
        max_hypotheses: Maximum number of hypotheses to test (None for all)
        model: OpenAI model to use
    """
    logger.info("="*80)
    logger.info("PDF TO FACTORS PIPELINE STARTING")
    logger.info("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load API key
    credentials = load_credentials("credentials.json")
    api_key = credentials.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")

    if not api_key:
        logger.error("No OpenAI API key found")
        raise ValueError("OpenAI API key required")

    # Initialize OpenAI client for PDF processing
    import openai
    openai_client = openai.OpenAI(api_key=api_key)

    # Step 1: Extract hypotheses from PDFs
    logger.info("\n" + "="*80)
    logger.info("STEP 1: EXTRACTING HYPOTHESES FROM PDFS")
    logger.info("="*80)

    hypotheses_json = os.path.join(output_dir, "extracted_hypotheses.json")
    hypotheses_summary = os.path.join(output_dir, "hypotheses_summary.md")

    results = process_pdf_directory(
        pdf_dir=pdf_dir,
        openai_client=openai_client,
        model=model,
        output_file=hypotheses_json,
        max_pdfs=max_pdfs
    )

    if not results:
        logger.error("No PDFs processed successfully")
        return

    # Save summary
    save_hypotheses_summary(results, hypotheses_summary)

    # Get all hypothesis texts
    hypotheses = get_all_hypotheses_text(results)

    if not hypotheses:
        logger.error("No hypotheses extracted from PDFs")
        return

    logger.info(f"\n✅ Extracted {len(hypotheses)} hypotheses from {len(results)} PDFs")

    # Limit hypotheses if specified
    if max_hypotheses is not None and len(hypotheses) > max_hypotheses:
        logger.info(f"Limiting to first {max_hypotheses} hypotheses")
        hypotheses = hypotheses[:max_hypotheses]

    # Step 2: Initialize FactorAgent
    logger.info("\n" + "="*80)
    logger.info("STEP 2: INITIALIZING FACTOR AGENT")
    logger.info("="*80)

    agent = FactorAgent(api_key=api_key, model=model, data_path=data_path)

    # Step 3: Generate and evaluate factors for each hypothesis
    logger.info("\n" + "="*80)
    logger.info("STEP 3: GENERATING FACTORS FROM HYPOTHESES")
    logger.info("="*80)

    all_hypothesis_results = []

    for hyp_idx, hypothesis in enumerate(hypotheses, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing Hypothesis {hyp_idx}/{len(hypotheses)}")
        logger.info(f"{'='*80}")
        logger.info(f"Hypothesis: {hypothesis}")

        try:
            # Generate factors from hypothesis
            logger.info(f"Generating 25 factors from hypothesis...")
            specs = agent.parse_hypothesis(hypothesis)

            if not specs:
                logger.warning(f"No factors generated for hypothesis {hyp_idx}")
                continue

            logger.info(f"✅ Generated {len(specs)} factors")

            # Compile factors
            compiled_factors = agent.compile_factors(specs)

            # Save results for this hypothesis
            hypothesis_result = {
                "hypothesis_id": hyp_idx,
                "hypothesis": hypothesis,
                "num_factors": len(specs),
                "factors": [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "reasoning": spec.reasoning,
                        "ast": spec.ast
                    }
                    for spec in specs
                ]
            }

            all_hypothesis_results.append(hypothesis_result)

            # Save individual hypothesis results
            hyp_output_file = os.path.join(output_dir, f"hypothesis_{hyp_idx}_factors.json")
            with open(hyp_output_file, 'w') as f:
                json.dump(hypothesis_result, f, indent=2)

            logger.info(f"✅ Saved factors to {hyp_output_file}")

        except Exception as e:
            logger.error(f"Error processing hypothesis {hyp_idx}: {e}")
            continue

    # Step 4: Save all results
    logger.info("\n" + "="*80)
    logger.info("STEP 4: SAVING FINAL RESULTS")
    logger.info("="*80)

    all_results_file = os.path.join(output_dir, "all_hypotheses_and_factors.json")
    with open(all_results_file, 'w') as f:
        json.dump(all_hypothesis_results, f, indent=2)

    logger.info(f"✅ Saved all results to {all_results_file}")

    # Summary
    total_factors = sum(r["num_factors"] for r in all_hypothesis_results)

    logger.info("\n" + "="*80)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*80)
    logger.info(f"PDFs processed: {len(results)}")
    logger.info(f"Hypotheses extracted: {len(hypotheses)}")
    logger.info(f"Hypotheses tested: {len(all_hypothesis_results)}")
    logger.info(f"Total factors generated: {total_factors}")
    logger.info(f"Output directory: {output_dir}")

    logger.info("\n" + "="*80)
    logger.info("PDF TO FACTORS PIPELINE FINISHED")
    logger.info("="*80)

    return all_hypothesis_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PDF to Factors Pipeline')
    parser.add_argument('--pdf-dir', type=str, default='/pdfs',
                       help='Directory containing PDF files')
    parser.add_argument('--data-path', type=str, default='/home/lichenhui/data/1min',
                       help='Directory containing market data')
    parser.add_argument('--output-dir', type=str, default='/home/lichenhui/data/alphaAgent/pdf_results',
                       help='Directory to save results')
    parser.add_argument('--max-pdfs', type=int, default=None,
                       help='Maximum number of PDFs to process')
    parser.add_argument('--max-hypotheses', type=int, default=None,
                       help='Maximum number of hypotheses to test')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='OpenAI model to use')

    args = parser.parse_args()

    run_pdf_to_factors_pipeline(
        pdf_dir=args.pdf_dir,
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_pdfs=args.max_pdfs,
        max_hypotheses=args.max_hypotheses,
        model=args.model
    )
