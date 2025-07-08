# COA PDF to Excel Converter

A Streamlit web application that converts Certificate of Analysis (COA) PDF documents into structured Excel files with multiple tabs grouped by product.

## Features

- 📄 **PDF Upload**: Easy drag-and-drop PDF file upload
- 🤖 **AI-Powered OCR**: Uses OpenAI's vision models to extract text from images
- 📊 **Product Grouping**: Automatically groups pages by product name
- 📁 **Multi-Tab Excel**: Creates Excel files with separate tabs for each product
- 🎯 **Structure Preservation**: Maintains table structure using "|" separators
- 📱 **User-Friendly Interface**: Clean, intuitive web interface

## Prerequisites

1. **Python 3.8+**
2. **OpenAI API Key** - Set as environment variable `OPENAI_API_KEY`
3. **Required Python packages** (see requirements.txt)

## Installation

1. **Clone/Download the files to your local machine**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set your OpenAI API key:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   
   Or on Windows:
   ```cmd
   set OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Running the Streamlit App

1. **Start the app:**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser** and go to `http://localhost:8501`

3. **Upload a PDF** by clicking "Choose a PDF file" or dragging and dropping

4. **Click "Process PDF"** to start the analysis

5. **Download the Excel file** once processing is complete

### Command Line Usage (Alternative)

If you prefer to run the processing directly:

```bash
python process_transcripts.py
```

This will process any PDF files in the current directory and generate Excel output.

## How It Works

1. **PDF → Images**: Converts each PDF page to high-quality images
2. **Images → Text**: Uses OpenAI's vision model to extract text and preserve structure
3. **Text Processing**: 
   - Extracts content from ```markdown``` blocks
   - Identifies product names using regex patterns
   - Processes tables using "|" separators
4. **Excel Generation**: Creates multi-tab Excel file with:
   - Summary tab with product overview
   - Individual tabs for each product with all related pages

## Input Requirements

Your PDF should contain Certificate of Analysis documents with:
- **Product Name fields** in formats like:
  - `**Product Name:** value`
  - `| Product Name: | value |`
  - `| **Product Name:** | value |`
- **Tabular data** using standard table formatting
- **Clear text** that can be read by OCR

## Output

The generated Excel file contains:
- **Summary Tab**: Overview of all products and page counts
- **Product Tabs**: One tab per product containing:
  - Product information header
  - All pages for that product
  - Page separators for easy navigation
  - Preserved table structure

## Example Structure

```
Excel File:
├── Summary (tab)
│   ├── Product Name | Number of Pages | Source Files
│   ├── TENOVAMED    | 3              | page_001.txt, page_002.txt, page_003.txt
│   └── Pregabalin   | 4              | page_004.txt, page_005.txt, page_006.txt, page_007.txt
├── TENOVAMED (tab)
│   ├── Product: TENOVAMED INOVAMED
│   ├── Total Pages: 3
│   ├── === Page 1: page_001.txt ===
│   ├── [Table data with | separators]
│   └── [Additional pages...]
└── Pregabalin (tab)
    └── [Similar structure for Pregabalin products]
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   - Ensure your API key is set correctly
   - Check that you have sufficient API credits

2. **No Images Generated**:
   - Verify PDF file is not corrupted
   - Ensure PDF contains text-based content (not just images)

3. **No Product Names Found**:
   - Check if your PDF follows the expected format
   - Product names should be clearly labeled

4. **Import Errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

### Performance Tips

- **File Size**: Larger PDFs will take longer to process
- **API Limits**: Be aware of OpenAI API rate limits
- **Processing Time**: Typical processing time is 1-2 minutes per page

## Files

- `streamlit_app.py` - Main Streamlit web application
- `process_transcripts.py` - Command-line processing script
- `requirements.txt` - Python dependencies
- `README.md` - This documentation

## Support

For issues or questions, check the console output for error messages and ensure all prerequisites are met.

---

**Happy Processing!** 🎉 