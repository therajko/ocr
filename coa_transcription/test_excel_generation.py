import sys
import os
import pandas as pd
import re
from io import StringIO
from collections import defaultdict

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def extract_product_name(content):
    """
    Extract product name from the transcript content.
    Looks for patterns like "Product Name:" followed by the actual name.
    """
    # Look for Product Name pattern - now includes plain table format
    product_name_patterns = [
        # Bold format patterns
        r'\*\*Product Name:\*\*\s*([^\n\|]+)',
        r'\|\s*\*\*Product Name:\*\*\s*\|\s*([^\|]+)\s*\|',
        # Plain table format patterns (like | Product Name: | value |)
        r'\|\s*Product Name:\s*\|\s*([^\|]+)\s*\|',
        # Simple format
        r'Product Name:\s*([^\n\|]+)',
    ]
    
    for pattern in product_name_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            product_name = match.group(1).strip()
            # Clean up the product name - remove extra whitespace and markdown
            product_name = re.sub(r'\s+', ' ', product_name)
            product_name = product_name.replace('*', '').strip()
            if len(product_name) > 2:  # Valid product name should have some length
                return product_name
    
    # If no specific pattern found, try to extract from the beginning of the document
    lines = content.split('\n')
    for line in lines:
        if 'product' in line.lower() and ('name' in line.lower() or ':' in line):
            # Extract text after colon or similar delimiter
            if ':' in line:
                potential_name = line.split(':')[-1].strip()
                potential_name = potential_name.replace('*', '').replace('|', '').strip()
                if len(potential_name) > 3:  # Reasonable product name length
                    return potential_name
    
    # Last resort: look for any pharmaceutical-sounding terms
    pharma_keywords = ['capsules', 'tablets', 'mg', 'hartkapseln', 'film-coated']
    lines = content.split('\n')
    for line in lines:
        line_lower = line.lower()
        if any(keyword in line_lower for keyword in pharma_keywords):
            # Extract potential product name from this line
            cleaned_line = re.sub(r'[|*#]', ' ', line).strip()
            if len(cleaned_line) > 10 and len(cleaned_line) < 100:
                return cleaned_line[:50]  # Reasonable length
    
    return ""  # Return empty string so sheet naming can use source file

def parse_table_from_content(content):
    """
    Parse table data from markdown content.
    Returns a DataFrame with the parsed table data.
    """
    # Find markdown tables
    table_pattern = r'\|.*\|'
    table_lines = []
    
    lines = content.split('\n')
    in_table = False
    
    for line in lines:
        if re.match(table_pattern, line.strip()):
            # Skip separator lines (lines with only dashes and pipes)
            if not re.match(r'^\s*\|[\s\-\|]*\|\s*$', line):
                table_lines.append(line.strip())
                in_table = True
        elif in_table and line.strip() == '':
            # Empty line might indicate end of table, but continue for now
            continue
        elif in_table and not re.match(table_pattern, line.strip()) and line.strip():
            # Non-table line after table started, might be end of table
            break
    
    if not table_lines:
        return None
    
    # Parse the table lines
    parsed_rows = []
    headers = None
    
    for i, line in enumerate(table_lines):
        # Split by pipe and clean up
        cells = [cell.strip() for cell in line.split('|')]
        # Remove empty cells from start and end (markdown table format)
        if cells and cells[0] == '':
            cells = cells[1:]
        if cells and cells[-1] == '':
            cells = cells[:-1]
        
        if i == 0:
            headers = cells
        else:
            parsed_rows.append(cells)
    
    if not headers or not parsed_rows:
        return None
    
    # Create DataFrame
    try:
        # Ensure all rows have the same number of columns as headers
        max_cols = len(headers)
        normalized_rows = []
        
        for row in parsed_rows:
            # Pad or truncate row to match header length
            if len(row) < max_cols:
                row.extend([''] * (max_cols - len(row)))
            elif len(row) > max_cols:
                row = row[:max_cols]
            normalized_rows.append(row)
        
        if headers and normalized_rows:
            # Create DataFrame with data and then set column names
            df = pd.DataFrame(normalized_rows)
            df.columns = [str(h) for h in headers]
            return df
        else:
            return None
        
    except Exception as e:
        print(f"Error creating DataFrame: {e}")
        return None

def extract_metadata_from_content(content):
    """
    Extract metadata like batch number, expiry date, etc. from the content.
    """
    metadata = {}
    
    # Common metadata patterns for both bold and plain table formats
    metadata_patterns = {
        'internal_misom_no': [
            r'\*\*Internal Misom No[:\.]?\*\*\s*([^\n\|]+)',
            r'\|\s*Internal Misom No[:\.]?\s*\|\s*([^\|]+)\s*\|'
        ],
        'coa_no': [
            r'\*\*CoA No[:\.]?.*?\*\*\s*([^\n\|]+)',
            r'\|\s*CoA No[:\.]?.*?\s*\|\s*([^\|]+)\s*\|'
        ],
        'batch_no': [
            r'\*\*(?:Manufacturer )?Batch No[:\.]?\*\*\s*([^\n\|]+)',
            r'\|\s*(?:Manufacturer )?Batch No[:\.]?\s*\|\s*([^\|]+)\s*\|'
        ],
        'exp_date': [
            r'\*\*Exp[:\.]? Date[:\.]?\*\*\s*([^\n\|]+)',
            r'\|\s*Exp[:\.]? Date[:\.]?\s*\|\s*([^\|]+)\s*\|'
        ],
        'client_name': [
            r'\*\*Client Name[:\.]?\*\*\s*([^\n\|]+)',
            r'\|\s*Client Name[:\.]?\s*\|\s*([^\|]+)\s*\|'
        ],
        'generic_name': [
            r'\*\*Generic Name[:\.]?\*\*\s*([^\n\|]+)',
            r'\|\s*Generic Name[:\.]?\s*\|\s*([^\|]+)\s*\|'
        ]
    }
    
    for key, patterns in metadata_patterns.items():
        for pattern in patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metadata[key] = match.group(1).strip()
                break  # Found a match, move to next metadata field
    
    return metadata

def process_transcripts_to_excel(output_dir, excel_filename="combined_analysis.xlsx"):
    """
    Process all transcript files and create an Excel file with multiple tabs grouped by product name.
    """
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist.")
        return None
    
    # Dictionary to group documents by product name
    products_data = defaultdict(list)
    
    # Process each transcript file
    transcript_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    
    if not transcript_files:
        print("No transcript files found in the output directory.")
        return None
    
    print(f"Processing {len(transcript_files)} transcript files...")
    
    for file in transcript_files:
        file_path = os.path.join(output_dir, file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract product name
            product_name = extract_product_name(content)
            print(f"File: {file} -> Product: '{product_name}'")
            
            # Extract metadata
            metadata = extract_metadata_from_content(content)
            
            # Parse table data
            table_df = parse_table_from_content(content)
            
            # Create a comprehensive record for this document
            document_data = {
                'source_file': file,
                'product_name': product_name,
                'metadata': metadata,
                'table_data': table_df,
                'raw_content': content
            }
            
            products_data[product_name].append(document_data)
            
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Create Excel file with multiple tabs
    excel_path = os.path.join(output_dir, excel_filename)
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        
        # Create a summary sheet
        summary_data = []
        for product_name, documents in products_data.items():
            summary_data.append({
                'Product Name': product_name,
                'Number of Documents': len(documents),
                'Source Files': ', '.join([doc['source_file'] for doc in documents])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Create individual sheets for each product
        sheet_name_counter = {}  # To handle duplicate names
        for product_name, documents in products_data.items():
            # Handle empty or invalid product names
            if not product_name or product_name.strip() == "":
                # Use source file name as fallback
                source_files = [doc['source_file'] for doc in documents]
                if source_files:
                    base_name = source_files[0].replace('.txt', '').replace('_', ' ')
                    product_name = f"Document {base_name}"
                else:
                    product_name = "Unknown Document"
            
            # Clean product name for sheet name (Excel sheet names have restrictions)
            clean_sheet_name = re.sub(r'[^\w\s-]', '', product_name)[:28]  # Leave room for counter
            clean_sheet_name = clean_sheet_name.strip()
            
            # Ensure sheet name is not empty
            if not clean_sheet_name:
                clean_sheet_name = "Document"
            
            # Handle duplicate sheet names
            if clean_sheet_name in sheet_name_counter:
                sheet_name_counter[clean_sheet_name] += 1
                clean_sheet_name = f"{clean_sheet_name}_{sheet_name_counter[clean_sheet_name]}"
            else:
                sheet_name_counter[clean_sheet_name] = 0
            
            # Combine all table data for this product
            combined_tables = []
            metadata_summary = {}
            
            for doc in documents:
                # Add metadata to summary
                if doc['metadata']:
                    for key, value in doc['metadata'].items():
                        if key not in metadata_summary:
                            metadata_summary[key] = []
                        metadata_summary[key].append(value)
                
                # Add table data
                if doc['table_data'] is not None and not doc['table_data'].empty:
                    # Add source file column
                    table_copy = doc['table_data'].copy()
                    table_copy['Source_File'] = doc['source_file']
                    combined_tables.append(table_copy)
            
            # Create the sheet content
            sheet_data = []
            
            # Add product information header
            sheet_data.append(['Product Name', product_name])
            sheet_data.append(['Number of Documents', len(documents)])
            sheet_data.append([''])  # Empty row
            
            # Add metadata summary
            sheet_data.append(['=== METADATA SUMMARY ==='])
            for key, values in metadata_summary.items():
                unique_values = list(set([v for v in values if v]))  # Remove duplicates and empty values
                sheet_data.append([key.replace('_', ' ').title(), ', '.join(unique_values)])
            
            sheet_data.append([''])  # Empty row
            
            # Add combined table data
            if combined_tables:
                sheet_data.append(['=== TEST RESULTS ==='])
                sheet_data.append([''])  # Empty row
                
                # Combine all tables
                combined_df = pd.concat(combined_tables, ignore_index=True, sort=False)
                
                # Convert to list format for Excel
                sheet_data.append(combined_df.columns.tolist())  # Headers
                for _, row in combined_df.iterrows():
                    sheet_data.append(row.tolist())
            
            # Create DataFrame and write to Excel
            max_cols = max(len(row) for row in sheet_data) if sheet_data else 1
            normalized_sheet_data = []
            
            for row in sheet_data:
                if len(row) < max_cols:
                    row.extend([''] * (max_cols - len(row)))
                normalized_sheet_data.append(row)
            
            sheet_df = pd.DataFrame(normalized_sheet_data)
            sheet_df.to_excel(writer, sheet_name=clean_sheet_name, index=False, header=False)
    
    print(f"\nExcel file created: {excel_path}")
    print(f"Products found: {list(products_data.keys())}")
    
    return excel_path, products_data

if __name__ == "__main__":
    output_dir = "./transcriptions"
    result = process_transcripts_to_excel(output_dir)
    
    if result is not None:
        excel_path, products_data = result
        print(f"\nTest complete!")
        print(f"Excel file saved to: {excel_path}")
        print(f"Found {len(products_data)} unique products")
    else:
        print("\nTest failed - no results generated") 