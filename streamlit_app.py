import streamlit as st
import os
import zipfile
import tempfile
import shutil
import time
import datetime
import subprocess
from pathlib import Path
import pandas as pd
import sys
from typing import Tuple, List, Optional
import base64

# Import our processing scripts as modules
from template_processor import main as process_template
from convert_to_images import main as convert_pdfs
import extract_and_process_images

# Set page config
st.set_page_config(
    page_title="OCR Document Processing Workflow",
    page_icon="📄",
    layout="wide",
)

# Get the current project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Function to get a download link for a file
def get_download_link(file_path, file_name=None):
    if file_name is None:
        file_name = os.path.basename(file_path)
    
    with open(file_path, "rb") as f:
        data = f.read()
    
    b64 = base64.b64encode(data).decode("utf-8")
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_name}">Download {file_name}</a>'
    return href

# Function to create a workspace directory structure
def create_workspace(base_dir: str) -> str:
    """Create a workspace with the required directory structure."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    workspace_dir = os.path.join(base_dir, f"ocr_workspace_{timestamp}")
    
    # Create main directories
    os.makedirs(workspace_dir, exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(workspace_dir, "transcriptions"), exist_ok=True)
    
    return workspace_dir

# Function to extract a zip file
def extract_zip(zip_file, extract_to: str) -> List[str]:
    """Extract a zip file and return a list of extracted files."""
    extracted_files = []
    
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        for file in zip_ref.namelist():
            zip_ref.extract(file, extract_to)
            extracted_files.append(os.path.join(extract_to, file))
    
    return extracted_files

# Function to run the entire workflow
def run_workflow(workspace_dir: str, template_file: Optional[str] = None, model_name: str = 'local_llm') -> Tuple[bool, str]:
    """
    Run the entire OCR workflow.
    
    Args:
        workspace_dir: Base directory for the workspace
        template_file: Optional path to a Word template file
        
    Returns:
        Tuple of (success, message)
    """

    model_name = os.environ.get("MODEL_NAME", 'local_llm')
    # Step 1: Process the Word template if provided
    if template_file:
        st.write("### Step 1: Processing Word Template")
        with st.spinner("Generating question set and markdown template..."):
            json_path, md_path = process_template(template_file, workspace_dir, model_name)
            if not json_path or not md_path:
                return False, "Failed to process Word template."
            st.success("✅ Word template processed successfully.")
    else:
        st.write("### Step 1: Skipped (No template file provided)")
    
    # Step 2: Convert PDFs to images
    st.write("### Step 2: Converting PDFs to Images")
    with st.spinner("Recursively searching for PDFs and converting to images..."):
        data_dir = os.path.join(workspace_dir, "data")
        success = convert_pdfs(data_dir)
        if not success:
            return False, "No PDF files found or conversion failed."
        st.success("✅ PDFs converted to images successfully.")
    
    # Step 3: Extract and process images
    st.write("### Step 3: Extracting and Processing Images")
    with st.spinner("Extracting and processing images... This may take some time..."):
        extract_and_process_images.main(workspace_dir, model_name)
        st.success("✅ Images processed successfully.")
    
    # Final results
    pqr_output_path = os.path.join(workspace_dir, "transcriptions", "pqr_output.md")
    if os.path.exists(pqr_output_path):
        with open(pqr_output_path, 'r') as f:
            pqr_content = f.read()
            st.write("### Final PQR Output")
            st.markdown(pqr_content)
    
    return True, "Workflow completed successfully."

# Main Streamlit UI
st.title("OCR Document Processing Workflow")

st.markdown("""
This app processes PDF documents through an OCR pipeline:
1. Process a Word template to generate question sets and output templates
2. Recursively search for PDFs in all folders and convert them to images for OCR processing
3. Extract text from images and analyze the content
4. Generate a consolidated PQR report
""")

# Check for environment variables
env_vars_set = all([
    os.environ.get("HUGGINGFACE_KEY"),
    os.environ.get("HF_LLAMA_GGUF_MODEL_ENDPOINT"),
    os.environ.get("HF_QWEN2_5_VL_MODEL_ENDPOINT")
])
model_name = os.environ.get("MODEL_NAME", 'local_llm')
if not env_vars_set:
    st.error("""
    ⚠️ Missing environment variables. Please set the following environment variables:
    - HUGGINGFACE_KEY
    - HF_LLAMA_GGUF_MODEL_ENDPOINT
    - HF_QWEN2_5_VL_MODEL_ENDPOINT
    """)

# File upload section
with st.expander("Upload Files", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload ZIP File")
        st.info("The app will recursively search through all folders in the ZIP file for PDFs")
        zip_file = st.file_uploader("Upload a ZIP file containing PDFs (in any subdirectory)", type=["zip"])
    
    with col2:
        st.subheader("Upload Template (Optional)")
        template_file = st.file_uploader("Upload a Word template", type=["docx"])

# Create workspaces directory if it doesn't exist
workspaces_dir = os.path.join(PROJECT_DIR, "workspaces")
os.makedirs(workspaces_dir, exist_ok=True)

# Run workflow button
if st.button("Start Processing", disabled=not (zip_file and env_vars_set)):
    # Create a progress bar
    progress_bar = st.progress(0)
    
    # Create a workspace
    with st.spinner("Setting up workspace..."):
        workspace_dir = create_workspace(workspaces_dir)
        st.session_state["workspace_dir"] = workspace_dir
        progress_bar.progress(10)
    
    # Extract ZIP file if provided
    if zip_file:
        with st.spinner("Extracting ZIP file..."):
            # Save the uploaded zip to a file in the project directory
            zip_path = os.path.join(workspace_dir, "uploaded.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.getvalue())
            
            # Extract the ZIP file
            data_dir = os.path.join(workspace_dir, "data")
            extracted_files = extract_zip(zip_path, data_dir)
            st.info(f"Extracted {len(extracted_files)} files from ZIP. Will now recursively search for PDFs.")
            
            # Remove the uploaded zip file after extraction
            os.remove(zip_path)
            progress_bar.progress(20)
    
    # Save template file if provided
    template_path = None
    if template_file:
        with st.spinner("Saving template file..."):
            # Save the template to the workspace
            template_path = os.path.join(workspace_dir, "template.docx")
            with open(template_path, "wb") as f:
                f.write(template_file.getvalue())
            progress_bar.progress(30)
    
    # Run the workflow
    with st.spinner("Running OCR workflow..."):
        success, message = run_workflow(workspace_dir, template_path, model_name)
        progress_bar.progress(100)
    
    # Display results
    if success:
        st.success(message)
        
        # Provide download links for the results
        st.subheader("Download Results")
        
        # Add download links for important files
        results = []
        
        pqr_output_path = os.path.join(workspace_dir, "transcriptions", "pqr_output.md")
        if os.path.exists(pqr_output_path):
            results.append(("PQR Report (Markdown)", pqr_output_path, "pqr_report.md"))
        
        json_path = os.path.join(workspace_dir, "transcription_metadata.json")
        if os.path.exists(json_path):
            results.append(("Question Set (JSON)", json_path, "question_set.json"))
        
        template_path = os.path.join(workspace_dir, "pqr_template.md")
        if os.path.exists(template_path):
            results.append(("PQR Template (Markdown)", template_path, "pqr_template.md"))
        
        # Display download links
        for name, path, filename in results:
            st.markdown(f"**{name}**: {get_download_link(path, filename)}", unsafe_allow_html=True)
        
    else:
        st.error(message)

# Display workspace information if available
if "workspace_dir" in st.session_state:
    with st.expander("Workspace Information"):
        st.write(f"Workspace directory: {st.session_state['workspace_dir']}")
        
        # Count files in each directory
        data_files = list(Path(os.path.join(st.session_state["workspace_dir"], "data")).rglob("*"))
        pdf_files = list(Path(os.path.join(st.session_state["workspace_dir"], "data")).rglob("*.pdf"))
        image_files = list(Path(os.path.join(st.session_state["workspace_dir"], "images")).glob("*"))
        transcription_files = list(Path(os.path.join(st.session_state["workspace_dir"], "transcriptions")).glob("*"))
        
        st.write(f"Data files: {len(data_files)} (including {len(pdf_files)} PDF files)")
        st.write(f"Image files: {len(image_files)}")
        st.write(f"Transcription files: {len(transcription_files)}")
        
        # Add a button to open the workspace folder
        if st.button("Open Workspace Folder"):
            if os.name == 'nt':  # Windows
                os.startfile(st.session_state["workspace_dir"])
            elif os.name == 'posix':  # macOS/Linux
                subprocess.run(['open' if sys.platform == 'darwin' else 'xdg-open', st.session_state["workspace_dir"]]) 