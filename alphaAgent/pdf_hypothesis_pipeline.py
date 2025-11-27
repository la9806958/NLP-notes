#!/usr/bin/env python3
"""
PDF Hypothesis Extraction Pipeline

This module extracts text from PDF research papers and uses an LLM to generate
market hypotheses that are predictive of 30-minute forward-looking returns.

The pipeline:
1. Extracts raw text from PDFs in /pdfs directory
2. Sends text to LLM with specialized prompt
3. Extracts market hypotheses that can be used with OHLCV data
4. Returns hypotheses for factor generation
"""

import os
import glob
import logging
from typing import List, Dict, Any, Optional
import json

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text content as string
    """
    try:
        # Try PyPDF2 first (most common)
        try:
            import PyPDF2

            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []

                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_parts.append(page.extract_text())

                text = '\n'.join(text_parts)
                logger.info(f"Extracted {len(text)} characters from {os.path.basename(pdf_path)} using PyPDF2")
                return text

        except ImportError:
            logger.warning("PyPDF2 not available, trying pdfplumber")

            # Fallback to pdfplumber
            try:
                import pdfplumber

                text_parts = []
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_parts.append(page_text)

                text = '\n'.join(text_parts)
                logger.info(f"Extracted {len(text)} characters from {os.path.basename(pdf_path)} using pdfplumber")
                return text

            except ImportError:
                logger.error("Neither PyPDF2 nor pdfplumber is available. Install with: pip install PyPDF2 or pip install pdfplumber")
                return ""

    except Exception as e:
        logger.error(f"Error extracting text from {pdf_path}: {e}")
        return ""


def build_hypothesis_extraction_prompt(paper_text: str, max_chars: int = 15000) -> str:
    """Build prompt for extracting market hypotheses from research paper text.

    Args:
        paper_text: Extracted text from research paper
        max_chars: Maximum characters to include from paper (to avoid token limits)

    Returns:
        Formatted prompt for LLM
    """
    # Truncate paper text if too long
    if len(paper_text) > max_chars:
        logger.warning(f"Paper text truncated from {len(paper_text)} to {max_chars} characters")
        paper_text = paper_text[:max_chars] + "\n\n[... text truncated due to length ...]"

    prompt = f"""You are a quantitative researcher analyzing academic research papers for trading signals.

AVAILABLE DATA:
- OHLC (Open, High, Low, Close) prices at 1-minute granularity
- Volume at 1-minute granularity
- Historical data going back 300 minutes (5 hours)

TASK:
Extract ALL market hypotheses from the research paper below that could be predictive of 30-minute forward-looking returns.

For each hypothesis:
1. It must be implementable using only OHLC and volume data
2. It should relate to market microstructure, price patterns, volume patterns, volatility, or liquidity
3. It should be specific and actionable (not vague statements)
4. It should focus on short-term predictive signals (minutes to hours)

EXAMPLE OF A GOOD HYPOTHESIS:
"Assets that are costly to trade tend to yield higher future returns because investors demand compensation for holding positions that are harder to liquidate. When trading becomes more expensive or less frequent, prices adjust downward to reflect this liquidity premium, implying that assets with higher transaction costs should earn higher subsequent returns."

This example can be proxied using OHLCV data:
- Bid-ask spread proxies: high-low range relative to price, intraday volatility
- Volume patterns: low volume indicates lower liquidity and higher transaction costs
- Price impact measures: large price moves on small volume indicate illiquidity
- Trading frequency: gaps between trades, irregular volume patterns

RESEARCH PAPER TEXT:
{paper_text}

OUTPUT FORMAT (JSON):
{{
  "paper_title": "title of the paper if identifiable",
  "hypotheses": [
    {{
      "hypothesis": "Clear, specific statement of the market hypothesis (like the liquidity example above)",
      "reasoning": "Why this hypothesis might predict 30-min forward returns",
      "data_requirements": "What OHLCV patterns are needed to test this",
      "source_section": "Which part of the paper this came from"
    }},
    ... (extract as many relevant hypotheses as possible)
  ]
}}

IMPORTANT:
- Extract hypotheses that are DIRECTLY testable with OHLCV data
- Avoid hypotheses that require order book data, news sentiment, or other unavailable data
- Focus on patterns that emerge in 1-minute to 300-minute windows
- Each hypothesis should be independently testable
- Include liquidity-related hypotheses (using volume, volatility, and price range as proxies)
- Look for relationships between transaction costs, liquidity, volatility, volume, and returns

Return ONLY valid JSON with no additional text or formatting."""

    return prompt


def extract_hypotheses_from_pdf(pdf_path: str, openai_client, model: str = "gpt-4o") -> Dict[str, Any]:
    """Extract market hypotheses from a PDF research paper.

    Args:
        pdf_path: Path to PDF file
        openai_client: OpenAI client instance
        model: Model name to use for extraction

    Returns:
        Dictionary with paper title and list of hypotheses
    """
    logger.info(f"Processing PDF: {pdf_path}")

    # Extract text from PDF
    paper_text = extract_text_from_pdf(pdf_path)

    if not paper_text or len(paper_text) < 100:
        logger.warning(f"Insufficient text extracted from {pdf_path} (only {len(paper_text)} chars)")
        return {
            "paper_title": os.path.basename(pdf_path),
            "hypotheses": [],
            "error": "Insufficient text extracted from PDF"
        }

    # Build prompt
    prompt = build_hypothesis_extraction_prompt(paper_text)

    # Call LLM
    try:
        logger.info(f"Sending {len(paper_text)} characters to LLM for hypothesis extraction")

        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a quantitative researcher extracting trading hypotheses from academic papers. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content.strip()

        # Parse JSON response
        result = json.loads(content)

        # Add metadata
        result["pdf_path"] = pdf_path
        result["pdf_filename"] = os.path.basename(pdf_path)

        num_hypotheses = len(result.get("hypotheses", []))
        logger.info(f"✅ Extracted {num_hypotheses} hypotheses from {os.path.basename(pdf_path)}")

        return result

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from LLM response: {e}")
        return {
            "paper_title": os.path.basename(pdf_path),
            "hypotheses": [],
            "error": f"JSON parse error: {e}"
        }
    except Exception as e:
        logger.error(f"Error calling LLM for {pdf_path}: {e}")
        return {
            "paper_title": os.path.basename(pdf_path),
            "hypotheses": [],
            "error": str(e)
        }


def process_pdf_directory(
    pdf_dir: str,
    openai_client,
    model: str = "gpt-4o",
    output_file: Optional[str] = None,
    max_pdfs: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Process all PDFs in a directory and extract hypotheses.

    Args:
        pdf_dir: Directory containing PDF files
        openai_client: OpenAI client instance
        model: Model name to use for extraction
        output_file: Optional path to save results as JSON
        max_pdfs: Maximum number of PDFs to process (None for all)

    Returns:
        List of results, one per PDF
    """
    logger.info(f"Processing PDFs from directory: {pdf_dir}")

    # Find all PDF files
    pdf_pattern = os.path.join(pdf_dir, "*.pdf")
    pdf_files = glob.glob(pdf_pattern)

    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF files")

    # Limit number of PDFs if specified
    if max_pdfs is not None and len(pdf_files) > max_pdfs:
        logger.info(f"Limiting to first {max_pdfs} PDFs")
        pdf_files = pdf_files[:max_pdfs]

    # Process each PDF
    all_results = []
    for i, pdf_path in enumerate(pdf_files, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing PDF {i}/{len(pdf_files)}: {os.path.basename(pdf_path)}")
        logger.info(f"{'='*80}")

        result = extract_hypotheses_from_pdf(pdf_path, openai_client, model)
        all_results.append(result)

    # Save results if output file specified
    if output_file:
        try:
            with open(output_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"\n✅ Saved results to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results to {output_file}: {e}")

    # Summary statistics
    total_hypotheses = sum(len(r.get("hypotheses", [])) for r in all_results)
    successful_pdfs = sum(1 for r in all_results if len(r.get("hypotheses", [])) > 0)

    logger.info(f"\n{'='*80}")
    logger.info("PIPELINE SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"PDFs processed: {len(all_results)}")
    logger.info(f"PDFs with hypotheses extracted: {successful_pdfs}")
    logger.info(f"Total hypotheses extracted: {total_hypotheses}")
    logger.info(f"Average hypotheses per PDF: {total_hypotheses/len(all_results):.1f}")

    return all_results


def get_all_hypotheses_text(results: List[Dict[str, Any]]) -> List[str]:
    """Extract all hypothesis text strings from results.

    Args:
        results: List of results from process_pdf_directory

    Returns:
        List of hypothesis text strings
    """
    hypotheses = []
    for result in results:
        for hyp in result.get("hypotheses", []):
            hypothesis_text = hyp.get("hypothesis", "")
            if hypothesis_text:
                hypotheses.append(hypothesis_text)

    return hypotheses


def save_hypotheses_summary(results: List[Dict[str, Any]], output_file: str):
    """Save a human-readable summary of extracted hypotheses.

    Args:
        results: List of results from process_pdf_directory
        output_file: Path to save summary markdown file
    """
    lines = []
    lines.append("# Extracted Market Hypotheses from Research Papers\n")
    lines.append(f"Generated: {os.popen('date').read().strip()}\n")
    lines.append(f"Total papers processed: {len(results)}\n")

    total_hyps = sum(len(r.get("hypotheses", [])) for r in results)
    lines.append(f"Total hypotheses extracted: {total_hyps}\n")
    lines.append("\n---\n")

    for result in results:
        paper_title = result.get("paper_title", result.get("pdf_filename", "Unknown"))
        hypotheses = result.get("hypotheses", [])

        if not hypotheses:
            continue

        lines.append(f"\n## {paper_title}\n")
        lines.append(f"**Source:** {result.get('pdf_filename', 'Unknown')}\n")
        lines.append(f"**Hypotheses extracted:** {len(hypotheses)}\n\n")

        for i, hyp in enumerate(hypotheses, 1):
            lines.append(f"### Hypothesis {i}\n")
            lines.append(f"**Statement:** {hyp.get('hypothesis', 'N/A')}\n\n")
            lines.append(f"**Reasoning:** {hyp.get('reasoning', 'N/A')}\n\n")
            lines.append(f"**Data Requirements:** {hyp.get('data_requirements', 'N/A')}\n\n")
            lines.append(f"**Source Section:** {hyp.get('source_section', 'N/A')}\n\n")
            lines.append("---\n\n")

    try:
        with open(output_file, 'w') as f:
            f.writelines(lines)
        logger.info(f"✅ Saved hypotheses summary to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save summary to {output_file}: {e}")


if __name__ == "__main__":
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description='Extract market hypotheses from PDF research papers')
    parser.add_argument('--pdf-dir', type=str, default='/pdfs',
                       help='Directory containing PDF files')
    parser.add_argument('--output-json', type=str, default='extracted_hypotheses.json',
                       help='Output JSON file for results')
    parser.add_argument('--output-summary', type=str, default='hypotheses_summary.md',
                       help='Output markdown file for human-readable summary')
    parser.add_argument('--max-pdfs', type=int, default=None,
                       help='Maximum number of PDFs to process (default: all)')
    parser.add_argument('--model', type=str, default='gpt-4o',
                       help='OpenAI model to use')

    args = parser.parse_args()

    # Load OpenAI API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set")
        exit(1)

    # Initialize OpenAI client
    import openai
    client = openai.OpenAI(api_key=api_key)

    # Process PDFs
    results = process_pdf_directory(
        pdf_dir=args.pdf_dir,
        openai_client=client,
        model=args.model,
        output_file=args.output_json,
        max_pdfs=args.max_pdfs
    )

    # Save human-readable summary
    if results:
        save_hypotheses_summary(results, args.output_summary)

        # Print all hypothesis texts
        hypotheses = get_all_hypotheses_text(results)
        logger.info(f"\n{'='*80}")
        logger.info("ALL EXTRACTED HYPOTHESES:")
        logger.info(f"{'='*80}")
        for i, hyp in enumerate(hypotheses, 1):
            logger.info(f"{i}. {hyp}")
