#!/usr/bin/env python3
"""
Hypothesis Generator from Research Papers
Reads PDF research papers and generates novel hypotheses based on their content.
"""

import os
import sys
import json
import csv
import re
from pathlib import Path
from datetime import datetime

try:
    import PyPDF2
except ImportError:
    print("PyPDF2 not installed. Installing...")
    os.system(f"{sys.executable} -m pip install PyPDF2 -q")
    import PyPDF2

try:
    from openai import OpenAI
except ImportError:
    print("openai not installed. Installing...")
    os.system(f"{sys.executable} -m pip install openai -q")
    from openai import OpenAI


def load_credentials(creds_path="credentials.json"):
    """Load API credentials from credentials.json file."""
    try:
        with open(creds_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {creds_path} not found")
        return None
    except json.JSONDecodeError:
        print(f"Error: {creds_path} is not valid JSON")
        return None


def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


def chunk_text(text, chunk_size=10000):
    """Split text into chunks of approximately chunk_size characters."""
    chunks = []
    words = text.split()
    current_chunk = []
    current_size = 0

    for word in words:
        word_len = len(word) + 1  # +1 for space
        if current_size + word_len > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_size = word_len
        else:
            current_chunk.append(word)
            current_size += word_len

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def parse_hypotheses(hypotheses_text):
    """Parse hypothesis text and extract individual hypotheses."""
    # Split by hypothesis headers
    pattern = r'##\s*Hypothesis\s*\d+[:\s]*(.+?)(?=##\s*Hypothesis\s*\d+|$)'
    matches = re.findall(pattern, hypotheses_text, re.DOTALL | re.IGNORECASE)

    hypotheses_list = []
    for match in matches:
        # Clean up the hypothesis text
        hypothesis = match.strip()
        if hypothesis:
            hypotheses_list.append(hypothesis)

    return hypotheses_list


def generate_hypotheses(paper_text, paper_name, api_key):
    """Generate hypotheses based on research paper content using GPT-4o-mini."""
    client = OpenAI(api_key=api_key)

    prompt = f"""You are a scientific research assistant. Based on the following research paper content, generate 5 novel, testable hypotheses that:

1. Extend the findings of this paper
2. Address gaps or limitations mentioned
3. Explore related research directions
4. Combine insights from this paper with other domains

Paper: {paper_name}

Content:
{paper_text[:15000]}  # Limit to avoid token limits

Please provide 5 hypotheses in the following format:

## Hypothesis 1: [Title]
**Rationale:** [Why this hypothesis is worth investigating]
**Testable Prediction:** [Specific, measurable prediction]
**Methodology:** [Brief outline of how to test this]

Generate creative, scientifically rigorous hypotheses."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error generating hypotheses: {e}")
        return None


def main():
    # Load credentials
    credentials = load_credentials()
    if not credentials:
        return

    api_key = credentials.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in credentials.json")
        return

    pdf_dir = Path("data/alphaAgent/pdfs")
    output_dir = Path("data/alphaAgent/hypotheses")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_dir.exists():
        print(f"Error: Directory {pdf_dir} does not exist")
        return

    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF file(s)")

    # Prepare CSV file
    csv_file = output_dir / "hypotheses.csv"
    csv_data = []

    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        print("-" * 60)

        # Extract text from PDF
        paper_text = extract_text_from_pdf(pdf_file)

        if not paper_text.strip():
            print(f"Warning: No text extracted from {pdf_file.name}")
            continue

        print(f"Extracted {len(paper_text)} characters")

        # Chunk the text
        chunks = chunk_text(paper_text, chunk_size=10000)
        print(f"Split into {len(chunks)} chunks")

        # Generate hypotheses for each chunk
        for i, chunk in enumerate(chunks, 1):
            print(f"\n  Processing chunk {i}/{len(chunks)}...")
            hypotheses_text = generate_hypotheses(chunk, pdf_file.name, api_key)

            if hypotheses_text:
                # Parse individual hypotheses
                hypotheses_list = parse_hypotheses(hypotheses_text)

                # Add to CSV data
                for hypothesis in hypotheses_list:
                    csv_data.append({
                        'pdf_title': pdf_file.stem,
                        'hypothesis': hypothesis
                    })
                    print(f"    ✓ Extracted hypothesis")

                # Save markdown file for this chunk
                output_file = output_dir / f"{pdf_file.stem}_chunk{i}_hypotheses.md"
                with open(output_file, 'w') as f:
                    f.write(f"# Hypotheses from {pdf_file.name} (Chunk {i})\n\n")
                    f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(hypotheses_text)
            else:
                print(f"    ✗ Failed to generate hypotheses for chunk {i}")

    # Write all hypotheses to CSV
    if csv_data:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['pdf_title', 'hypothesis'])
            writer.writeheader()
            writer.writerows(csv_data)

        print("\n" + "=" * 60)
        print(f"✓ All hypotheses saved to: {csv_file}")
        print(f"Total hypotheses generated: {len(csv_data)}")
    else:
        print("\n" + "=" * 60)
        print("✗ No hypotheses were generated")

    print("Processing complete!")


if __name__ == "__main__":
    main()
