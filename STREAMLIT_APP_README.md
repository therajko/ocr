# OCR Document Processing Streamlit App

This Streamlit application provides an easy-to-use interface for the OCR document processing workflow. It allows you to upload ZIP files containing PDFs and optionally a Word template, then processes them through the OCR pipeline to extract and analyze text content.

## Prerequisites

Before running the app, make sure you have the following:

1. Python 3.6+ installed
2. Required dependencies installed (see Installation section)
3. Environment variables set up for the API endpoints

## Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Install Poppler (required by pdf2image):
   - **macOS**: `brew install poppler`
   - **Ubuntu/Debian**: `apt-get install poppler-utils`
   - **Windows**: Download from [poppler releases](http://blog.alivate.com.au/poppler-windows/) and add to PATH

## Setting Up Environment Variables

Set the following environment variables before running the app:

```bash
# On Linux/macOS
export HUGGINGFACE_KEY=your_huggingface_api_key
export HF_LLAMA_GGUF_MODEL_ENDPOINT=your_llama_model_endpoint
export HF_QWEN2_5_VL_MODEL_ENDPOINT=your_qwen_vl_model_endpoint

# On Windows
set HUGGINGFACE_KEY=your_huggingface_api_key
set HF_LLAMA_GGUF_MODEL_ENDPOINT=your_llama_model_endpoint
set HF_QWEN2_5_VL_MODEL_ENDPOINT=your_qwen_vl_model_endpoint
```

## Running the App

Run the Streamlit app with:

```
streamlit run streamlit_app.py
```

The app will be available at http://localhost:8501 in your web browser.

## How to Use

1. **Upload Files**:
   - Upload a ZIP file containing PDF documents
   - Optionally upload a Word template document

2. **Start Processing**:
   - Click the "Start Processing" button to begin the OCR workflow
   - The app will process the files in the following order:
     1. Process the Word template to generate a question set and markdown template
     2. Convert PDFs to images
     3. Extract text from images using OCR
     4. Analyze the extracted text
     5. Generate a consolidated PQR report

3. **Download Results**:
   - When processing is complete, download links will be available for:
     - PQR Report (Markdown)
     - Question Set (JSON)
     - PQR Template (Markdown)

## Workflow Steps Explained

1. **Template Processing**:
   - Reads a Word document template
   - Generates a structured JSON question set
   - Creates a markdown template for the final report

2. **PDF to Image Conversion**:
   - Converts each page of each PDF into JPEG images
   - Resizes images for optimal OCR processing

3. **OCR Processing**:
   - Extracts text from images using a vision-language model
   - Saves transcriptions as text files

4. **Text Analysis**:
   - Processes transcriptions to extract structured information
   - Combines and analyzes data from multiple documents
   - Generates a consolidated PQR report

## Troubleshooting

- **Missing Environment Variables**: Ensure all required environment variables are set
- **PDF Conversion Issues**: Make sure Poppler is properly installed
- **Processing Errors**: Check the logs for specific error messages

## File Structure

- `streamlit_app.py`: Main Streamlit application
- `template_processor.py`: Processes Word templates to generate question sets and templates
- `convert_to_images.py`: Converts PDF documents to images
- `extract_and_process_images.py`: Extracts text from images and processes it
- `create_batch_prompt.py`: Creates batch prompts from processed data
- `requirements.txt`: List of Python dependencies 