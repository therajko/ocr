import os
import sys
import pdf2image
from pathlib import Path
from PIL import Image
import argparse
import glob

def find_pdfs_recursively(base_dir):
    """
    Find all PDF files recursively in the base directory.
    
    Args:
        base_dir (str): Base directory to search from
        
    Returns:
        list: List of Path objects to PDF files
    """
    print(f"Searching for PDFs recursively in {base_dir}...")
    pdf_files = list(Path(base_dir).rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")
    return pdf_files

def check_existing_images(pdf_path, images_dir):
    """
    Check if images for a PDF already exist.
    
    Args:
        pdf_path (Path): Path to the PDF file
        images_dir (Path): Directory where images are stored
        
    Returns:
        bool: True if all images for this PDF exist, False otherwise
    """
    pdf_name = pdf_path.stem
    rel_path = pdf_path.relative_to(pdf_path.parent.parent)
    rel_dir = str(rel_path.parent).replace('/', '_').replace('\\', '_')
    if rel_dir != '.':
        pdf_name = f"{rel_dir}_{pdf_name}"
    
    # Check if any image files exist for this PDF
    existing_images = list(images_dir.glob(f"{pdf_name}_page_*.jpg"))
    return len(existing_images) > 0

def convert_pdfs_to_images(data_path):
    # Create data directory if it doesn't exist
    data_dir = Path(data_path)
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        print(f"Created '{data_path}' directory. Please place your PDF files in this directory.")
        return False
    
    # Create images directory if it doesn't exist
    images_dir = data_dir.parent / "images"
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
        print(f"Created '{images_dir}' directory for storing converted images.")
    
    # Get all PDF files recursively in the data directory and its subdirectories
    pdf_files = find_pdfs_recursively(data_dir)
    
    if not pdf_files:
        print(f"No PDF files found in '{data_path}' or its subdirectories.")
        return False
    
    # Process each PDF file
    for pdf_path in pdf_files:
        # Check if images for this PDF already exist
        if check_existing_images(pdf_path, images_dir):
            print(f"Skipping {pdf_path.name} - images already exist")
            continue
        
        # Get the PDF name without extension to use as prefix
        pdf_name = pdf_path.stem
        
        # Include subdirectory in the name to avoid conflicts
        rel_path = pdf_path.relative_to(data_dir)
        rel_dir = str(rel_path.parent).replace('/', '_').replace('\\', '_')
        if rel_dir != '.':
            pdf_name = f"{rel_dir}_{pdf_name}"
        
        print(f"Processing {pdf_path.name} from {pdf_path.parent}...")
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            
            # Save each page as a JPEG with PDF name as prefix
            for i, image in enumerate(images):
                # Reduce image size by 50%
                width, height = image.size
                new_size = (width // 2, height // 2)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Create filename with PDF name as prefix
                image_path = images_dir / f"{pdf_name}_page_{i+1:03d}.jpg"
                image.save(image_path, "JPEG")
                print(f"  Saved page {i+1} as {image_path.name}")
            
            print(f"Completed processing {pdf_path.name}. {len(images)} pages converted.")
        
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
    
    return True

def main(data_path):
    success = convert_pdfs_to_images(data_path)
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert PDF files to images')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data directory containing PDF files')
    args = parser.parse_args()
    
    convert_pdfs_to_images(args.data_path)
