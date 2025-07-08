import sys
import os
import pandas as pd
import re
from collections import defaultdict

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import convert_to_images
from extract_and_process_images import process_images_with_model

def extract_markdown_content(content):
    """
    Extract content between ```markdown ``` blocks.
    """
    # Find all markdown code blocks
    markdown_pattern = r'```markdown\s*\n(.*?)\n```'
    matches = re.findall(markdown_pattern, content, re.DOTALL)
    
    if matches:
        # Join all markdown blocks with double newlines
        return '\n\n'.join(matches)
    else:
        # If no markdown blocks found, return original content
        return content

def extract_product_name(content):
    """
    Extract product name from the transcript content.
    """
    # Simple patterns for product name extraction
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
    """
    Convert content to Excel rows using "|" as separator and preserving newlines.
    """
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
    """
    Create Excel file with tabs grouped by product name.
    """
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return None
    
    # Group documents by product name
    products_data = defaultdict(list)
    
    # Process transcript files
    transcript_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    
    if not transcript_files:
        print("No transcript files found.")
        return None
    
    print(f"Processing {len(transcript_files)} files...")
    
    for file in transcript_files:
        try:
            with open(os.path.join(output_dir, file), 'r', encoding='utf-8') as f:
                raw_content = f.read()
            
            # Pre-process: Extract markdown content
            content = extract_markdown_content(raw_content)
            
            # Extract product name
            product_name = extract_product_name(content)
            print(f"File: {file} -> Product: {product_name or 'No product found'}")
            
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
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
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
    
    print(f"\nExcel file created: {excel_path}")
    print(f"Products found: {list(products_data.keys())}")
    
    return excel_path

def main(pdf_file_path=None):
    """
    Main function - simplified pipeline.
    Args:
        pdf_file_path (str, optional): Specific PDF file to process. If None, processes all PDFs in current directory.
    """
    images_local_path = "./images"
    output_dir = "./transcriptions"
    model_name = "openai"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Skip image processing if transcriptions already exist
    transcript_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')] if os.path.exists(output_dir) else []
    
    if not transcript_files:
        print("Step 1: Converting PDF to images...")
        
    
        # Process specific PDF file
        print(f"Processing specific PDF: {pdf_file_path}")
        
        # Create isolated directory for the specific PDF
        pdf_dir = "./pdf_input"
        os.makedirs(pdf_dir, exist_ok=True)
        
        # Copy the specific PDF to isolated directory
        import shutil
        pdf_basename = os.path.basename(pdf_file_path)
        target_pdf = os.path.join(pdf_dir, pdf_basename)
        shutil.copy2(pdf_file_path, target_pdf)
        
        # Process only the isolated directory
        convert_to_images.main(pdf_dir)
    
        
        print("Step 2: Processing images...")
        process_images_with_model(images_local_path, output_dir, model_name)
    else:
        print(f"Found {len(transcript_files)} existing transcriptions, skipping image processing...")
    
    # Create Excel file
    print("Step 3: Creating Excel file...")
    excel_path = create_excel_from_transcripts(output_dir)
    
    if excel_path:
        print(f"\nSuccess! Excel file saved to: {excel_path}")
    else:
        print("\nFailed to create Excel file.")

if __name__ == "__main__":
    main()
