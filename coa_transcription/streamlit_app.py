import streamlit as st
import os
import sys
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import re
from collections import defaultdict

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import convert_to_images
from extract_and_process_images import process_images_with_model

def extract_markdown_content(content):
    """Extract content between ```markdown ``` blocks."""
    markdown_pattern = r'```markdown\s*\n(.*?)\n```'
    matches = re.findall(markdown_pattern, content, re.DOTALL)
    
    if matches:
        return '\n\n'.join(matches)
    else:
        return content

def extract_product_name(content):
    """Extract product name from the transcript content."""
    product_name_patterns = [
        r'\|\s*\*\*Product Name:\*\*\s*\|\s*([^\|]+)\s*\|',  # | **Product Name:** | value |
        r'\|\s*Product Name:\s*\|\s*([^\|]+)\s*\|',          # | Product Name: | value |
        r'\*\*Product Name:\*\*\s*([^\n\|]+)',               # **Product Name:** value
        r'Product Name:\s*([^\n\|]+)',                       # Product Name: value
    ]
    
    for pattern in product_name_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            product_name = match.group(1).strip()
            # Clean up the product name
            product_name = re.sub(r'[*|]', '', product_name).strip()
            # Remove "Product Name:" if it appears at the start
            product_name = re.sub(r'^Product Name:\s*', '', product_name, flags=re.IGNORECASE).strip()
            if len(product_name) > 2:
                return product_name
    
    # Fallback: look for pharmaceutical terms
    lines = content.split('\n')
    for line in lines:
        if any(term in line.lower() for term in ['capsules', 'tablets', 'mg', 'hartkapseln']):
            cleaned_line = re.sub(r'[|*#]', ' ', line).strip()
            if 10 < len(cleaned_line) < 80:
                return cleaned_line[:50]
    
    return ""

def process_content_to_rows(content):
    """Convert content to Excel rows using "|" as separator and preserving newlines."""
    lines = content.split('\n')
    rows = []
    
    for line in lines:
        line = line.strip()
        if not line:
            rows.append([''])  # Empty row for spacing
            continue
            
        # If line contains "|", split by it
        if '|' in line:
            cells = [cell.strip() for cell in line.split('|')]
            # Remove empty cells from start and end (markdown format)
            if cells and cells[0] == '':
                cells = cells[1:]
            if cells and cells[-1] == '':
                cells = cells[:-1]
            if cells:  # Only add non-empty rows
                rows.append(cells)
        else:
            # Single cell row
            if line.strip():
                rows.append([line])
    
    return rows

def create_excel_from_transcripts(output_dir, excel_filename="combined_analysis.xlsx"):
    """Create Excel file with tabs grouped by product name."""
    if not os.path.exists(output_dir):
        return None
    
    # Group documents by product name
    products_data = defaultdict(list)
    
    # Process transcript files
    transcript_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    
    if not transcript_files:
        return None
    
    for file in transcript_files:
        with open(os.path.join(output_dir, file), 'r', encoding='utf-8') as f:
            raw_content = f.read()
        
        # Pre-process: Extract markdown content
        content = extract_markdown_content(raw_content)
        
        # Extract product name
        product_name = extract_product_name(content)
        
        # Convert content to rows
        rows = process_content_to_rows(content)
        
        # Store document data
        document_data = {
            'source_file': file,
            'rows': rows,
            'processed_content': content,
            'raw_content': raw_content
        }
        
        # Use filename as fallback if no product name found
        if not product_name:
            product_name = file.replace('.txt', '').replace('_', ' ')
        
        products_data[product_name].append(document_data)
    
    # Create Excel file
    excel_path = os.path.join(output_dir, excel_filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Summary sheet
        summary_data = []
        for product_name, documents in products_data.items():
            summary_data.append({
                'Product Name': product_name,
                'Number of Pages': len(documents),
                'Source Files': ', '.join([doc['source_file'] for doc in documents])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Individual product sheets
        for product_name, documents in products_data.items():
            # Create clean sheet name
            sheet_name = re.sub(r'[^\w\s-]', '', product_name)[:31]
            if not sheet_name.strip():
                sheet_name = "Document"
            
            # Combine all rows from all pages for this product
            all_rows = []
            
            # Add header
            all_rows.append([f"Product: {product_name}"])
            all_rows.append([f"Total Pages: {len(documents)}"])
            all_rows.append([''])  # Empty row
            
            # Add content from each page
            for i, doc in enumerate(documents, 1):
                all_rows.append([f"=== Page {i}: {doc['source_file']} ==="])
                all_rows.append([''])  # Empty row
                
                # Add the processed rows
                all_rows.extend(doc['rows'])
                all_rows.append([''])  # Separator between pages
            
            # Find maximum number of columns
            max_cols = max(len(row) for row in all_rows) if all_rows else 1
            
            # Normalize all rows to have same number of columns
            normalized_rows = []
            for row in all_rows:
                if len(row) < max_cols:
                    row.extend([''] * (max_cols - len(row)))
                normalized_rows.append(row)
            
            # Create DataFrame and save
            df = pd.DataFrame(normalized_rows)
            df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
    
    return excel_path, products_data

def process_pdf_pipeline(pdf_path, temp_dir, model_name="openai"):
    """Complete pipeline to process PDF and generate Excel."""
    
    # Create isolated subdirectory for the uploaded PDF only
    pdf_dir = os.path.join(temp_dir, "pdf_input")
    images_dir = os.path.join(temp_dir, "images")
    transcriptions_dir = os.path.join(temp_dir, "transcriptions")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(transcriptions_dir, exist_ok=True)
    
    # Copy PDF to isolated directory
    temp_pdf_path = os.path.join(pdf_dir, "uploaded_document.pdf")
    shutil.copy2(pdf_path, temp_pdf_path)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Convert PDF to images
    status_text.text("Step 1/3: Converting PDF to images...")
    progress_bar.progress(10)
    
    # Change to temp directory and process only the isolated PDF directory
    original_cwd = os.getcwd()
    os.chdir(temp_dir)
    

    convert_to_images.main("./pdf_input")
    
    progress_bar.progress(30)
    
    # Check if images were created
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        st.error("No images were generated from the PDF")
        return None
    
    st.success(f"✅ Generated {len(image_files)} images from PDF")
    
    # Step 2: Process images with model
    status_text.text("Step 2/3: Processing images with AI model...")
    progress_bar.progress(50)
    
    process_images_with_model(images_dir, transcriptions_dir, model_name)
    progress_bar.progress(80)
    
    # Check if transcriptions were created
    transcript_files = [f for f in os.listdir(transcriptions_dir) if f.endswith('.txt')]
    if not transcript_files:
        st.error("No transcriptions were generated")
        return None
    
    st.success(f"✅ Generated {len(transcript_files)} transcriptions")
    
    # Step 3: Create Excel file
    status_text.text("Step 3/3: Creating Excel file...")
    progress_bar.progress(90)
    
    result = create_excel_from_transcripts(transcriptions_dir)
    
    if result:
        excel_path, products_data = result
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Display results summary
        st.success(f"🎉 Successfully processed PDF!")
        st.info(f"📊 Found {len(products_data)} unique products across {len(transcript_files)} pages")
        
        # Show product summary
        with st.expander("📋 Product Summary"):
            for product_name, documents in products_data.items():
                st.write(f"**{product_name}**: {len(documents)} pages")
        
        return excel_path
    else:
        st.error("Failed to create Excel file")
        return None

def main():
    st.set_page_config(
        page_title="COA PDF to Excel Converter",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Certificate of Analysis (COA) PDF to Excel Converter")
    st.markdown("Upload a PDF containing Certificate of Analysis documents and get a structured Excel file with multiple tabs grouped by product.")
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        st.error("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF containing Certificate of Analysis documents"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"📁 Uploaded: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
        
        # Process button
        if st.button("🚀 Process PDF", type="primary"):
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # Save uploaded file
                pdf_path = os.path.join(temp_dir, uploaded_file.name)
                with open(pdf_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process the PDF
                excel_path = process_pdf_pipeline(pdf_path, temp_dir)
                
                if excel_path and os.path.exists(excel_path):
                    # Read the Excel file for download
                    with open(excel_path, "rb") as f:
                        excel_data = f.read()
                    
                    # Provide download button
                    st.download_button(
                        label="📥 Download Excel File",
                        data=excel_data,
                        file_name=f"coa_analysis_{uploaded_file.name.replace('.pdf', '')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    
                    st.balloons()
    
    # Instructions
    with st.expander("📖 How to use this app"):
        st.markdown("""
        ### Instructions:
        1. **Upload PDF**: Click "Choose a PDF file" and select your Certificate of Analysis PDF
        2. **Process**: Click "Process PDF" to start the analysis
        3. **Download**: Once processing is complete, click "Download Excel File" to get your results
        
        ### What this app does:
        - 🔄 Converts PDF pages to images
        - 🤖 Uses AI (OpenAI) to extract text from images
        - 📊 Groups pages by product name
        - 📁 Creates Excel file with multiple tabs (one per product)
        - 🎯 Preserves table structure using "|" separators
        
        ### Requirements:
        - Valid OpenAI API key (set as environment variable)
        - PDF containing Certificate of Analysis documents
        - Products should have "Product Name:" fields for proper grouping
        """)

if __name__ == "__main__":
    main() 