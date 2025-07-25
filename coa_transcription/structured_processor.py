import os
import json
import pandas as pd
import openai
from pathlib import Path
import re
from collections import defaultdict
from rapidfuzz import fuzz, process

class StructuredCOAProcessor:
    """
    Process COA transcripts using structured templates to generate standardized Excel output.
    """
    
    def __init__(self, templates_dir="./output_format"):
        self.templates_dir = Path(templates_dir)
        self.templates = self.load_templates()
        self.client = None
    
    def load_templates(self):
        """Load CSV templates from the output_format directory."""
        templates = {}
        
        if not self.templates_dir.exists():
            print(f"Templates directory {self.templates_dir} not found")
            return templates
        
        for csv_file in self.templates_dir.glob("*.csv"):
            template_name = csv_file.stem.replace("Final format - ", "")
            try:
                # Read the raw CSV to preserve ALL rows including multi-level headers
                with open(csv_file, 'r', encoding='utf-8') as f:
                    raw_lines = f.readlines()
                
                # Parse CSV manually to preserve structure
                import csv
                all_rows = []
                with open(csv_file, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    for row in csv_reader:
                        all_rows.append(row)
                
                if not all_rows:
                    continue
                
                # Find the main header row (usually the last non-empty header row)
                header_row_idx = -1
                for i, row in enumerate(all_rows):
                    # Check if this row has meaningful headers (not mostly empty)
                    non_empty_cells = [cell.strip() for cell in row if cell.strip()]
                    if len(non_empty_cells) > 5:  # Assume main header has many columns
                        header_row_idx = i
                
                if header_row_idx == -1:
                    header_row_idx = 0  # Fallback to first row
                
                # Extract headers from the identified header row
                headers = all_rows[header_row_idx]
                
                # Clean up headers
                cleaned_headers = []
                for i, header in enumerate(headers):
                    if not header.strip():
                        cleaned_headers.append(f"Column_{i + 1}")
                    else:
                        cleaned_headers.append(header.strip())
                
                # Ensure all rows have the same length as headers
                max_cols = len(cleaned_headers)
                normalized_rows = []
                for row in all_rows:
                    normalized_row = row + [''] * (max_cols - len(row))  # Pad with empty strings
                    normalized_rows.append(normalized_row[:max_cols])  # Trim to max_cols
                
                templates[template_name] = {
                    'file': csv_file,
                    'headers': cleaned_headers,
                    'all_rows': normalized_rows,  # Keep all rows including headers
                    'header_row_index': header_row_idx,
                    'data_start_index': header_row_idx + 1,
                    'total_rows': len(normalized_rows)
                }
                print(f"Loaded template: {template_name} ({len(normalized_rows)} rows, {len(cleaned_headers)} columns)")
            except Exception as e:
                print(f"Error loading template {csv_file}: {e}")
        
        return templates
    
    def identify_product_template(self, transcript_content):
        """
        Identify which template to use based on transcript content using similarity scoring.
        """
        content_lower = transcript_content.lower()
        
        # Extract key terms from content for matching
        content_terms = self._extract_key_terms(content_lower)
        
        template_scores = {}
        
        for template_name, template_data in self.templates.items():
            # Extract key terms from template name
            template_terms = self._extract_key_terms(template_name.lower())
            
            # Calculate similarity scores using different strategies
            scores = []
            
            # 1. Direct fuzzy matching between template name and content
            direct_score = fuzz.partial_ratio(template_name.lower(), content_lower)
            scores.append(direct_score * 0.3)  # Weight: 30%
            
            # 2. Fuzzy matching between template terms and content terms
            if template_terms and content_terms:
                term_scores = []
                for template_term in template_terms:
                    # Find best match for this template term in content terms
                    best_match = process.extractOne(template_term, content_terms, scorer=fuzz.ratio)
                    if best_match and best_match[1] > 60:  # Minimum threshold
                        term_scores.append(best_match[1])
                
                if term_scores:
                    avg_term_score = sum(term_scores) / len(term_scores)
                    scores.append(avg_term_score * 0.4)  # Weight: 40%
            
            # 3. Product-specific similarity (dosage matching for pharmaceuticals)
            dosage_score = self._calculate_dosage_similarity(template_name.lower(), content_lower)
            scores.append(dosage_score * 0.3)  # Weight: 30%
            
            # Final weighted score
            final_score = sum(scores) if scores else 0
            
            if final_score > 20:  # Minimum threshold to consider a match
                template_scores[template_name] = final_score
        
        # Return the template with the highest score
        if template_scores:
            best_template = max(template_scores.keys(), key=lambda k: template_scores[k])
            print(f"Template matching scores: {template_scores}")
            print(f"Selected template: {best_template} (score: {template_scores[best_template]:.2f})")
            return best_template, self.templates[best_template]
        
        # Default to first available template
        if self.templates:
            first_template = list(self.templates.keys())[0]
            print(f"No specific match found, using default: {first_template}")
            return first_template, self.templates[first_template]
        
        return None, None
    
    def _extract_key_terms(self, text):
        """
        Extract key pharmaceutical terms from text for similarity matching.
        """
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'and', 'or', 'of', 'in', 'to', 'for', 'with', 'by', 'format', 'final'}
        
        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        
        # Filter out stop words and short words
        key_terms = [word.lower() for word in words 
                    if len(word) > 2 and word.lower() not in stop_words]
        
        # Also extract pharmaceutical-specific patterns
        # Drug names, dosages, etc.
        pharma_patterns = [
            r'\b[a-zA-Z]+\d+mg\b',  # e.g., pregabalin100mg
            r'\b\d+mg\b',           # e.g., 100mg
            r'\b[a-zA-Z]{5,}\b'     # Longer pharmaceutical names
        ]
        
        for pattern in pharma_patterns:
            matches = re.findall(pattern, text)
            key_terms.extend([match.lower() for match in matches])
        
        return list(set(key_terms))  # Remove duplicates
    
    def _calculate_dosage_similarity(self, template_name, content):
        """
        Calculate similarity based on dosage information for pharmaceutical products.
        """
        # Extract dosages from template name (e.g., "100mg", "150mg")
        template_dosages = re.findall(r'(\d+)\s*mg\b', template_name)
        
        # Extract dosages from content (various formats)
        content_dosages = re.findall(r'(\d+)\s*mg\b', content)
        content_dosages.extend(re.findall(r'(\d+)\s+mg\b', content))
        content_dosages.extend(re.findall(r'(\d+)mg', content))
        
        if not template_dosages:
            return 50  # Neutral score if no dosage in template
        
        if not content_dosages:
            return 30  # Lower score if no dosage in content
        
        # Convert to integers and filter reasonable dosage range
        template_dosages = [int(d) for d in template_dosages]
        content_dosages = [int(d) for d in content_dosages if 10 <= int(d) <= 2000]
        
        # Check for exact dosage matches
        for template_dose in template_dosages:
            if template_dose in content_dosages:
                return 100  # Perfect dosage match
        
        # Check for close dosage matches (within 20% difference)
        for template_dose in template_dosages:
            for content_dose in content_dosages:
                diff_ratio = abs(template_dose - content_dose) / template_dose
                if diff_ratio <= 0.2:  # Within 20%
                    return 80  # Close dosage match
        
        return 20  # No dosage match
    
    def create_extraction_prompt(self, template_name, template_data, transcript_content):
        """
        Create a detailed prompt for AI to extract data according to the template.
        """
        headers = template_data['headers']
        
        # Identify analytical result columns vs metadata columns
        analytical_columns = []
        metadata_columns = []
        
        for header in headers:
            header_lower = header.lower()
            if any(keyword in header_lower for keyword in [
                'assay', 'content', 'dissolution', 'uniformity', 'identification', 
                'impurity', 'substances', 'microbial', 'count', 'cfu', 'coli',
                'water content', 'related substances', '%', 'result'
            ]):
                analytical_columns.append(header)
            else:
                metadata_columns.append(header)
        
        prompt = f"""
You are a pharmaceutical data extraction specialist. Extract information from Certificate of Analysis (COA) content and format it according to the specified template structure.

TARGET TEMPLATE: {template_name}

COA CONTENT (may contain multiple pages/sections):
{transcript_content}

EXTRACTION PRIORITIES:

1. ANALYTICAL RESULTS (HIGHEST PRIORITY - Extract actual measured values):
{chr(10).join([f"   - {header}: Look for measured results, percentages, or test outcomes" for header in analytical_columns])}

2. PRODUCT METADATA (Extract identifying information):
{chr(10).join([f"   - {header}" for header in metadata_columns])}

CRITICAL INSTRUCTIONS FOR ANALYTICAL RESULTS:
- Extract ACTUAL MEASURED VALUES from test results tables
- Look for patterns like "Result: 2.1%", "Assay: 99.2%", "Water Content: 1.8%"
- For identification tests, extract results like "Complies", "Positive", "Confirmed"
- For microbial counts, extract values like "<10 CFU/g", "Not detected", specific numbers
- For impurities, extract percentages like "0.15%", "Not detected", "< 0.1%"
- For dissolution/uniformity, extract "Complies" or specific values
- If a test shows both result and specification, extract the RESULT value
- Combine results from multiple pages if the same test appears multiple times

GENERAL INSTRUCTIONS:
1. This content may come from multiple pages of the same COA document
2. For analytical columns, prioritize MEASURED RESULTS over specifications
3. For metadata columns, extract identifying information
4. If a value appears in multiple sections, use the most complete/accurate value
5. If no result is found for an analytical test, use "N/A"
6. Preserve units when they're part of the result (e.g., "2.1%", "<10 CFU/g")

OUTPUT FORMAT:
Provide the extracted data as a JSON object with keys matching the column headers exactly. Focus especially on populating the analytical result columns with actual measured values.

Example:
{{
    "Product Name": "Pregabalin Laurus 100mg Hartkapseln",
    "Water Content by KF (% m/m) (In-house)": "2.1%",
    "Assay by HPLC (%) (In-House)": "99.2%",
    "By HPLC (In House)": "Complies",
    "Total Impurities (In-House)": "0.15%",
    "Total Aerobic Microbial Count (CFU/g) (Ph. Eur)": "<10 CFU/g",
    ...
}}

Focus on extracting complete analytical results to populate the certificate data accurately.
"""
        return prompt
    
    def extract_data_with_ai(self, template_name, template_data, transcript_content):
        """
        Use OpenAI to extract structured data from transcript.
        """
        # Initialize OpenAI client when needed
        if self.client is None:
            self.client = openai.OpenAI()
            
        # Create a much simpler, more direct prompt
        headers = template_data['headers']
        
        simple_prompt = f"""
Extract data from this Certificate of Analysis content and format as JSON.

COA CONTENT:
{transcript_content}

REQUIRED FIELDS (extract exactly what you see in the content):
{chr(10).join([f'- {header}' for header in headers])}

INSTRUCTIONS:
1. Look for actual test results and values in the content
2. Extract product information like name, batch number, etc.
3. For test results, extract the actual measured values (not specifications)
4. If you can't find a value, use "N/A"
5. Return valid JSON with the exact field names provided

OUTPUT: Return ONLY a JSON object with the required fields.
"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a data extraction expert. Extract information from pharmaceutical certificates and return valid JSON."},
                    {"role": "user", "content": simple_prompt}
                ],
                temperature=0.1
            )
            
            extracted_text = response.choices[0].message.content
            if extracted_text:
                extracted_text = extracted_text.strip()
                print(f"Raw AI response: {extracted_text[:200]}...")
            else:
                print("AI returned empty content")
            
            # Extract JSON from response
            if extracted_text:
                # Try to find JSON in the response
                json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
                if json_match:
                    extracted_data = json.loads(json_match.group())
                    print(f"Successfully extracted {len(extracted_data)} fields")
                    return extracted_data
                else:
                    print("No JSON found in AI response")
                    print(f"Full response: {extracted_text}")
                    return None
            else:
                print("Empty response from AI")
                return None
                
        except Exception as e:
            print(f"Error in AI extraction: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def identify_product_from_transcript(self, content):
        """
        Extract product identification information from transcript content.
        """
        content_lower = content.lower()
        
        # Updated patterns to match actual transcript content
        product_patterns = [
            r'\*\*Product Name:\*\*\s*([^\n\|]+)',                    # **Product Name:** value
            r'Product Name:\s*\|\s*([^\|]+)\s*\|',                    # | Product Name: | value |
            r'\|\s*\*\*Product Name:\*\*\s*\|\s*([^\|]+)\s*\|',       # | **Product Name:** | value |
            r'Product Name:\s*([^\n\|]+)',                            # Product Name: value
        ]
        
        batch_patterns = [
            r'\*\*Manufacturer Batch No\.?:\*\*\s*([^\n\|]+)',        # **Manufacturer Batch No:** value
            r'Manufacturer Batch No\.?:\s*\|\s*([^\|]+)\s*\|',        # | Manufacturer Batch No: | value |
            r'\|\s*\*\*Manufacturer Batch No\.?:\*\*\s*\|\s*([^\|]+)\s*\|',  # | **Manufacturer Batch No:** | value |
            r'Manufacturer Batch No\.?:\s*([^\n\|]+)',                # Manufacturer Batch No: value
            r'Manufacture Batch No\.?:\s*\|\s*([^\|]+)\s*\|',         # | Manufacture Batch No: | value | (typo in source)
        ]
        
        product_info = {}
        
        # Extract product name
        for pattern in product_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                product_name = match.group(1).strip()
                # Clean up the product name
                product_name = re.sub(r'[*|]', '', product_name).strip()
                if len(product_name) > 3:
                    product_info['product_name'] = product_name
                    print(f"Found product name: '{product_name}'")
                    break
        
        # Extract batch number
        for pattern in batch_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                batch_number = match.group(1).strip()
                # Clean up the batch number
                batch_number = re.sub(r'[*|]', '', batch_number).strip()
                if len(batch_number) > 2:
                    product_info['batch_number'] = batch_number
                    print(f"Found batch number: '{batch_number}'")
                    break
        
        # If no product name found, use fallback
        if 'product_name' not in product_info:
            if any(term in content_lower for term in ['tenovamed', 'emtricitabine', 'tenofovir']):
                product_info['product_name'] = 'TENOVAMED INOVAMED'
                print("Using fallback: TENOVAMED INOVAMED")
            else:
                product_info['product_name'] = 'Unknown Product'
                print("No product name found, using Unknown Product")
        
        if 'batch_number' not in product_info:
            print("No batch number found")
            product_info['batch_number'] = 'Unknown'
        
        # Return tuple for compatibility with calling code
        return product_info.get('product_name', 'Unknown'), product_info.get('batch_number', 'Unknown')
    
    def _are_product_names_similar(self, name1, name2):
        """
        Check if two product names are similar enough to be considered the same product.
        """
        if not name1 or not name2:
            return False
        
        # Normalize names for comparison
        norm1 = name1.lower().strip()
        norm2 = name2.lower().strip()
        
        # Exact match
        if norm1 == norm2:
            return True
        
        # Check if one is a substring of the other (for partial matches)
        if norm1 in norm2 or norm2 in norm1:
            return True
        
        # Check for common pharmaceutical name patterns
        # Remove common variations and compare core names
        variations_to_remove = ['product', 'unknown', 'hartkapseln', 'capsules', 'tablets']
        
        clean1 = norm1
        clean2 = norm2
        
        for variation in variations_to_remove:
            clean1 = clean1.replace(variation, '').strip()
            clean2 = clean2.replace(variation, '').strip()
        
        # If cleaned names are similar
        if clean1 and clean2 and (clean1 in clean2 or clean2 in clean1):
            return True
        
        return False
    
    def group_transcripts_by_product(self, transcripts_dir):
        """
        Group transcript files by product name.
        Returns: dict with product names as keys and list of file info as values
        """
        if not os.path.exists(transcripts_dir):
            return {}
        
        transcript_files = [f for f in os.listdir(transcripts_dir) 
                           if f.endswith('.txt') and os.path.isfile(os.path.join(transcripts_dir, f))]
        
        if not transcript_files:
            return {}
        
        # Sort files to ensure consistent processing order
        transcript_files.sort()
        
        products = {}
        
        print(f"DEBUG - Processing {len(transcript_files)} transcript files...")
        
        for filename in transcript_files:
            filepath = os.path.join(transcripts_dir, filename)
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            product_name, batch_number = self.identify_product_from_transcript(content)
            
            # Create standardized product key for grouping
            if product_name:
                # Normalize product name for grouping
                if "TENOVAMED" in product_name.upper():
                    product_key = "TENOVAMED INOVAMED"
                elif "PREGABALIN" in product_name.upper() and "100" in product_name:
                    product_key = "Pregabalin Laurus 100mg"
                elif "PREGABALIN" in product_name.upper() and "150" in product_name:
                    product_key = "Pregabalin Laurus 150mg"
                else:
                    product_key = product_name[:50]  # Use first 50 chars as fallback
            else:
                # Use filename as fallback
                product_key = f"Unknown_Product_{filename.replace('.txt', '')}"
            
            print(f"DEBUG - File: {filename} -> Product Key: '{product_key}'")
            
            if product_key not in products:
                products[product_key] = []
            
            products[product_key].append({
                'filename': filename,
                'filepath': filepath,
                'content': content,
                'product_name': product_name,
                'batch_number': batch_number
            })
        
        print(f"DEBUG - Found {len(products)} product groups:")
        for product_key, files in products.items():
            print(f"  - {product_key}: {len(files)} files")
        
        return products
    
    def process_transcripts_directory(self, transcripts_dir):
        """
        Process transcript files grouped by product.
        """
        # Group transcripts by product
        product_groups = self.group_transcripts_by_product(transcripts_dir)
        
        if not product_groups:
            print("No product groups found")
            return {}
        
        results = {}
        
        # Process each product group
        for product_key, transcripts in product_groups.items():
            print(f"\nProcessing product group: {product_key}")
            print(f"Combining {len(transcripts)} transcripts...")
            
            # Combine all transcript content for this product
            combined_content = []
            source_files = []
            
            for transcript in transcripts:
                combined_content.append(f"=== Content from {transcript['filename']} ===")
                combined_content.append(transcript['content'])
                combined_content.append("\n" + "="*50 + "\n")
                source_files.append(transcript['filename'])
            
            full_content = "\n".join(combined_content)
            
            # Identify appropriate template based on combined content
            template_name, template_data = self.identify_product_template(full_content)
            
            if not template_name:
                print(f"No template found for product group {product_key}")
                continue
            
            print(f"Using template: {template_name}")
            
            # Extract structured data from combined content
            extracted_data = self.extract_data_with_ai(template_name, template_data, full_content)
            
            if extracted_data:
                results[product_key] = {
                    'template': template_name,
                    'data': extracted_data,
                    'source_files': source_files,
                    'transcript_count': len(transcripts),
                    'product_info': {
                        'product_name': transcripts[0]['product_name'],
                        'batch_number': transcripts[0]['batch_number']
                    }
                }
                print(f"Successfully extracted data for {product_key}")
            else:
                print(f"Failed to extract data for {product_key}")
        
        return results
    
    def create_excel_output(self, processed_results, output_file="structured_coa_analysis.xlsx"):
        """
        Create Excel file with properly formatted sheets based on templates.
        Each row represents one product (potentially combining multiple transcript pages).
        """
        if not processed_results:
            print("No processed results to create Excel file")
            return None
        
        # Group results by template type
        template_groups = defaultdict(list)
        for product_key, result in processed_results.items():
            template_groups[result['template']].append(result)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            
            # Create summary sheet
            summary_data = []
            total_transcripts = 0
            
            for template_name, results in template_groups.items():
                transcript_count = sum(r['transcript_count'] for r in results)
                total_transcripts += transcript_count
                
                summary_data.append({
                    'Template Type': template_name,
                    'Number of Products': len(results),
                    'Total Transcripts': transcript_count,
                    'Products': ', '.join([r.get('product_info', {}).get('product_name', 'Unknown') for r in results])
                })
            
            # Add overall summary
            summary_data.insert(0, {
                'Template Type': 'TOTAL',
                'Number of Products': sum(len(results) for results in template_groups.values()),
                'Total Transcripts': total_transcripts,
                'Products': f"{len(processed_results)} unique products processed"
            })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create sheets for each template type
            for template_name, results in template_groups.items():
                
                # Get template structure
                if template_name in self.templates:
                    template_data = self.templates[template_name]
                    headers = template_data['headers']
                    all_template_rows = template_data['all_rows']
                    data_start_index = template_data['data_start_index']
                    
                    # Start with the complete template structure
                    output_rows = []
                    
                    # Add all template rows (including multi-level headers and template rows)
                    for row in all_template_rows:
                        output_rows.append(row[:])  # Copy the row
                    
                    # Add extracted data rows for each product
                    for result in results:
                        extracted = result['data']
                        source_files = result['source_files']
                        
                        # Create data row matching template structure
                        data_row = [''] * len(headers)
                        
                        # Map extracted data to correct columns
                        for i, header in enumerate(headers):
                            if header in extracted:
                                data_row[i] = str(extracted[header])
                            elif header.strip() == '':
                                data_row[i] = ''  # Keep empty for empty headers
                            else:
                                data_row[i] = 'N/A'  # Default for missing data
                        
                        # Add source file info to first empty column or append as comment
                        source_info = f"Sources: {'; '.join(source_files)}"
                        
                        # Try to find a good place for source info
                        for i, header in enumerate(headers):
                            if 'source' in header.lower() or header.strip() == '':
                                data_row[i] = source_info
                                break
                        
                        output_rows.append(data_row)
                    
                    # Create DataFrame preserving exact structure
                    # Use column indices as column names to avoid header conflicts
                    if output_rows:
                        output_df = pd.DataFrame(output_rows)
                    else:
                        output_df = pd.DataFrame()
                    
                    # Clean sheet name
                    sheet_name = re.sub(r'[^\w\s-]', '', template_name)[:31]
                    if not sheet_name.strip():
                        sheet_name = "COA_Data"
                    
                    # Write to Excel without headers (since template includes its own headers)
                    output_df.to_excel(writer, sheet_name=sheet_name, index=False, header=False)
                    
                    print(f"Created sheet '{sheet_name}' with complete template structure:")
                    print(f"  - Original template rows: {len(all_template_rows)}")
                    print(f"  - Added data rows: {len(results)}")
                    print(f"  - Total rows: {len(output_rows)}")
                    print(f"  - Source transcripts: {sum(r['transcript_count'] for r in results)}")
        
        print(f"Excel file created: {output_file}")
        print(f"Final result: {len(processed_results)} products processed from multiple transcript pages")
        return output_file


def main():
    """
    Main function to process COA transcripts using structured templates.
    """
    # Initialize processor
    processor = StructuredCOAProcessor()
    
    # Process transcripts
    transcripts_dir = "./transcriptions"
    results = processor.process_transcripts_directory(transcripts_dir)
    
    if results:
        # Create Excel output
        excel_file = processor.create_excel_output(results)
        print(f"Processing complete! Results saved to: {excel_file}")
    else:
        print("No results to process")


if __name__ == "__main__":
    main() 