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
from structured_processor import StructuredCOAProcessor

def check_dependencies():
    """Check if required system dependencies are available."""
    try:
        import pdf2image
        # Try to check if poppler is available
        from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
        return True
    except ImportError as e:
        st.error(f"Missing Python dependency: {e}")
        return False
    except Exception as e:
        st.warning(f"Dependency check warning: {e}")
        return True  # Continue anyway

# Legacy functions removed - now using StructuredCOAProcessor

def create_structured_excel_from_transcripts(transcriptions_dir, temp_dir, excel_filename="structured_coa_analysis.xlsx"):
    """Create Excel file using structured templates from output_format."""
    
    # Initialize the structured processor
    output_format_path = os.path.join(os.path.dirname(__file__), "output_format")
    processor = StructuredCOAProcessor(templates_dir=output_format_path)
    
    # Process all transcripts using structured templates
    st.info("Processing transcripts with structured templates...")
    results = processor.process_transcripts_directory(transcriptions_dir)
    
    if not results:
        st.error("No structured data could be extracted from transcripts")
        return None
    
    # Create Excel file in temp directory
    excel_path = os.path.join(temp_dir, excel_filename)
    final_excel = processor.create_excel_output(results, excel_path)
    
    if final_excel:
        # Count processed results by template and transcripts
        template_counts = defaultdict(lambda: {'products': 0, 'transcripts': 0})
        for result in results.values():
            template = result['template']
            template_counts[template]['products'] += 1
            template_counts[template]['transcripts'] += result['transcript_count']
        
        return final_excel, template_counts
    else:
        return None

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
    
    try:
        os.chdir(temp_dir)
        
        # Convert PDF to images
        try:
            success = convert_to_images.main("./pdf_input")  # Use relative path
            
            if not success:
                st.error("Failed to convert PDF to images")
                return None
                
        except Exception as e:
            st.error(f"Error during PDF conversion: {str(e)}")
            st.error("This might be due to missing system dependencies. Make sure poppler-utils is installed.")
            return None
            
    finally:
        # Always restore the original working directory
        os.chdir(original_cwd)
    
    progress_bar.progress(30)
    
    # Debug information
    st.info(f"Temp directory structure:")
    for root, dirs, files in os.walk(temp_dir):
        level = root.replace(temp_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        st.text(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            st.text(f"{subindent}{file}")
    
    # Check if images were created
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))] if os.path.exists(images_dir) else []
    st.info(f"Images directory exists: {os.path.exists(images_dir)}")
    st.info(f"Found {len(image_files)} image files")
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
    
    # Step 3: Create Excel file using structured templates
    status_text.text("Step 3/3: Creating structured Excel file...")
    progress_bar.progress(90)
    
    result = create_structured_excel_from_transcripts(transcriptions_dir, temp_dir)
    
    if result:
        excel_path, template_counts = result
        progress_bar.progress(100)
        status_text.text("✅ Processing complete!")
        
        # Display results summary
        st.success(f"🎉 Successfully processed PDF using structured templates!")
        total_products = sum(counts['products'] for counts in template_counts.values())
        total_transcripts = sum(counts['transcripts'] for counts in template_counts.values())
        st.info(f"📊 Processed {total_products} products from {total_transcripts} transcript pages across {len(template_counts)} template types")
        
        # Show template summary
        with st.expander("📋 Processing Summary"):
            for template_name, counts in template_counts.items():
                st.write(f"**{template_name}**: {counts['products']} products from {counts['transcripts']} transcript pages")
        
        return excel_path
    else:
        st.error("Failed to create structured Excel file")
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
    
    # Check dependencies
    if not check_dependencies():
        st.error("⚠️ Required dependencies are not available. Please check the installation.")
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
        - 🤖 Uses AI (OpenAI) to extract text and analytical data from images
        - 🔍 Groups transcript pages by product name and batch number
        - 🎯 Uses structured templates to format data consistently
        - 📊 Automatically identifies product types (Pregabalin, Tenovamed, etc.)
        - 🧪 **Extracts actual test results** (assay values, impurities, microbial counts, etc.)
        - 📁 Creates Excel file with one row per product containing all analytical results
        - ✅ Ensures data matches exact pharmaceutical reporting standards
        
        ### Supported Product Types & Extracted Results:
        - **Pregabalin products**: Laurus Hartkapseln formulations
          - Assay, Water Content, Dissolution, Uniformity, Related Substances, Microbial Quality
        - **Tenovamed products**: Tenofovir Disoproxil Fumarate/Emtricitabine combinations  
          - Drug Content, Impurities, Dissolution profiles, Microbial testing
        - **Additional templates**: Can be added to the output_format directory
        
        ### Requirements:
        - Valid OpenAI API key (set as environment variable)
        - PDF containing Certificate of Analysis documents
        - Documents should contain identifiable product information
        """)

if __name__ == "__main__":
    main() 