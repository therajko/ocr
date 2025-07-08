# OCR Document Processing Workflow

This repository contains a workflow for OCR (Optical Character Recognition) processing of PDF documents. The workflow converts PDFs to images, processes them with OCR, extracts data, and generates analysis reports.

## Workflow Overview

1. **PDF to Image Conversion**: Convert PDF files to JPEG images
2. **OCR Processing**: Extract text from images using OCR models
3. **Text Analysis**: Process extracted text to identify and extract structured data
4. **Batch Analysis**: Combine multiple document analyses into a single report

## Requirements

- Python 3.6+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### 1. Converting PDFs to Images

```
python convert_to_images.py --data_path path/to/data/directory
```

This script:
- Takes PDFs from the specified data directory
- Converts each page to a JPEG image
- Resizes images to improve processing speed
- Saves images in a subdirectory with naming format `{pdf_name}_page_{page_number}.jpg`

### 2. Extracting and Processing Images

```
python extract_and_process_images.py --base_path path/to/base/directory
```

This script:
- Processes images with OCR to extract text content
- Saves transcriptions to text files
- Analyzes transcriptions to extract structured data
- Generates analysis JSON files for each document

### 3. Creating Batch Analysis

```
python create_batch_prompt.py --base_path path/to/base/directory
```

This script:
- Combines multiple analysis files into a batch prompt
- Formats data for further processing or reporting
- Outputs a structured batch prompt file

## File Structure

- `convert_to_images.py`: Converts PDF files to JPEG images
- `extract_and_process_images.py`: Processes images with OCR and extracts data
- `create_batch_prompt.py`: Creates batch analysis from multiple documents
- `requirements.txt`: List of Python dependencies

## Environment Variables

This workflow requires the following environment variables:
- `HUGGINGFACE_KEY`: API key for HuggingFace endpoints (used for OCR processing)

## Data Flow

1. PDFs → Images (via `convert_to_images.py`)
2. Images → Text Transcriptions → Structured Data (via `extract_and_process_images.py`)
3. Multiple Structured Data Files → Batch Analysis Report (via `create_batch_prompt.py`)

## Advanced Usage

The workflow supports customization through additional parameters and configuration files:
- Process specific subsets of documents
- Apply different analysis templates
- Generate various output formats

## Troubleshooting

- Ensure all dependencies are correctly installed
- Check that input directories contain the expected file formats
- Verify environment variables are properly set

## License

[Specify your license here] 