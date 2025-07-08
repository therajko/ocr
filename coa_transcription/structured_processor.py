import os
import json
import pandas as pd
import openai
from pathlib import Path
import re
from collections import defaultdict

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
                # Try reading with multi-row headers first (pharmaceutical COA format)
                try:
                    df = pd.read_csv(csv_file, header=[0, 1])
                    
                    # Combine multi-level headers into single level
                    if isinstance(df.columns, pd.MultiIndex):
                        combined_headers = []
                        for col in df.columns:
                            # Join non-unnamed parts
                            parts = [str(part) for part in col if not str(part).startswith('Unnamed')]
                            if parts:
                                combined_headers.append(' | '.join(parts))
                            else:
                                # If all parts are unnamed, use position-based naming
                                combined_headers.append(f"Column_{len(combined_headers) + 1}")
                        df.columns = combined_headers
                except:
                    # Fallback to single header row
                    df = pd.read_csv(csv_file)
                    
                    # Clean up any unnamed columns 
                    cleaned_headers = []
                    for i, col in enumerate(df.columns):
                        if str(col).startswith('Unnamed'):
                            cleaned_headers.append(f"Column_{i + 1}")
                        else:
                            cleaned_headers.append(str(col))
                    df.columns = cleaned_headers
                
                templates[template_name] = {
                    'file': csv_file,
                    'headers': df.columns.tolist(),
                    'structure': df.to_dict('records')[0] if len(df) > 0 else {},
                    'dataframe': df
                }
                print(f"Loaded template: {template_name}")
            except Exception as e:
                print(f"Error loading template {csv_file}: {e}")
        
        return templates
    
    def identify_product_template(self, transcript_content):
        """
        Identify which template to use based on transcript content.
        """
        content_lower = transcript_content.lower()
        
        # Product-specific keywords for template matching
        template_keywords = {
            'pregabalin': ['pregabalin', 'hartkapseln', 'laurus'],
            'tenovamed': ['tenofovir', 'emtricitabine', 'tenovamed', 'inovamed'],
        }
        
        scores = {}
        for template_type, keywords in template_keywords.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                scores[template_type] = score
        
        if scores:
            best_match = max(scores.keys(), key=lambda k: scores[k])
            
            # Find matching template
            for template_name, template_data in self.templates.items():
                if best_match in template_name.lower():
                    return template_name, template_data
        
        # Default to first available template
        if self.templates:
            first_template = list(self.templates.keys())[0]
            return first_template, self.templates[first_template]
        
        return None, None
    
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
            
        prompt = self.create_extraction_prompt(template_name, template_data, transcript_content)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a pharmaceutical data extraction specialist. Extract data accurately and format it as requested."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            extracted_text = response.choices[0].message.content
            
            # Extract JSON from response
            if extracted_text:
                json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
            else:
                json_match = None
            if json_match:
                extracted_data = json.loads(json_match.group())
                return extracted_data
            else:
                print("No JSON found in AI response")
                return None
                
        except Exception as e:
            print(f"Error in AI extraction: {e}")
            return None
    
    def identify_product_from_transcript(self, content):
        """
        Extract product identification information from transcript content.
        """
        content_lower = content.lower()
        
        # Product name extraction patterns
        product_patterns = [
            r'product\s*name[:\s]*([^\n\|]+)',
            r'produktname[:\s]*([^\n\|]+)',  
            r'produkt[:\s]*([^\n\|]+)',
            r'product[:\s]*([^\n\|]+)',  # Added pattern for "Product:"
            r'preparation[:\s]*([^\n\|]+)',
            r'drug[:\s]*([^\n\|]+)',
        ]
        
        # Batch number patterns  
        batch_patterns = [
            r'batch\s*(?:no|number)[:\s]*([^\n\|]+)',
            r'batch[:\s]*([^\n\|]+)',  # Added pattern for "Batch:"
            r'lot\s*(?:no|number)[:\s]*([^\n\|]+)',
            r'charge[:\s]*([^\n\|]+)',
        ]
        
        product_info = {}
        
        # Extract product name
        for pattern in product_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                product_name = match.group(1).strip()
                # Clean up
                product_name = re.sub(r'[|*#]', ' ', product_name).strip()
                if len(product_name) > 3:
                    product_info['product_name'] = product_name[:100]  # Limit length
                    break
        
        # Extract batch number
        for pattern in batch_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                batch_no = match.group(1).strip()
                batch_no = re.sub(r'[|*#]', ' ', batch_no).strip()
                if len(batch_no) > 2:
                    product_info['batch_number'] = batch_no[:50]
                    break
        
        # Fallback: use keywords to identify product type
        if 'product_name' not in product_info:
            if any(term in content_lower for term in ['pregabalin', 'hartkapseln']):
                product_info['product_name'] = 'Pregabalin Product'
            elif any(term in content_lower for term in ['tenofovir', 'emtricitabine', 'tenovamed']):
                product_info['product_name'] = 'Tenovamed Product'
            else:
                product_info['product_name'] = 'Unknown Product'
        
        return product_info
    
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
        Group transcript files by product identity.
        """
        transcripts_path = Path(transcripts_dir)
        if not transcripts_path.exists():
            print(f"Transcripts directory {transcripts_dir} not found")
            return {}
        
        transcript_files = list(transcripts_path.glob("*.txt"))
        print(f"Found {len(transcript_files)} transcript files")
        
        # Read all transcripts and extract product info
        transcript_data = {}
        for transcript_file in transcript_files:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            product_info = self.identify_product_from_transcript(content)
            transcript_data[transcript_file] = {
                'content': content,
                'product_info': product_info
            }
        
        # Group by product identity with fuzzy matching
        product_groups = defaultdict(list)
        
        for transcript_file, data in transcript_data.items():
            product_info = data['product_info']
            
            # Create a product key for grouping
            product_name = product_info.get('product_name', 'Unknown')
            batch_number = product_info.get('batch_number', 'Unknown')
            
            # Check if this transcript should be grouped with an existing product
            matched_group = None
            
            # Look for existing groups with similar product name and same batch
            for existing_key, existing_transcripts in product_groups.items():
                existing_info = existing_transcripts[0]['product_info']
                existing_name = existing_info.get('product_name', '')
                existing_batch = existing_info.get('batch_number', '')
                
                # Match if batch numbers are the same and product names are similar
                if (batch_number != 'Unknown' and batch_number == existing_batch and
                    self._are_product_names_similar(product_name, existing_name)):
                    matched_group = existing_key
                    break
            
            if matched_group:
                # Add to existing group
                product_groups[matched_group].append({
                    'file': transcript_file,
                    'content': data['content'],
                    'product_info': product_info
                })
            else:
                # Create new group
                product_key = f"{product_name}_{batch_number}".replace(' ', '_')
                product_groups[product_key].append({
                    'file': transcript_file,
                    'content': data['content'],
                    'product_info': product_info
                })
        
        print(f"Grouped transcripts into {len(product_groups)} products:")
        for product_key, transcripts in product_groups.items():
            print(f"  - {product_key}: {len(transcripts)} transcripts")
        
        return product_groups
    
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
                combined_content.append(f"=== Content from {transcript['file'].name} ===")
                combined_content.append(transcript['content'])
                combined_content.append("\n" + "="*50 + "\n")
                source_files.append(transcript['file'].name)
            
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
                    'product_info': transcripts[0]['product_info']  # Use first transcript's product info
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
                    headers = self.templates[template_name]['headers']
                    
                    # Create data rows from extracted results (one row per product)
                    data_rows = []
                    for result in results:
                        row_data = {}
                        extracted = result['data']
                        
                        # Map extracted data to template columns
                        for header in headers:
                            row_data[header] = extracted.get(header, "N/A")
                        
                        # Add metadata about source files (in a comment or additional column if space)
                        if 'Source Files' not in headers and len(headers) < 50:  # Avoid too many columns
                            source_files_str = '; '.join(result['source_files'])
                            row_data['Source Files'] = source_files_str
                        
                        data_rows.append(row_data)
                    
                    # Create DataFrame with template structure + data
                    if data_rows:
                        # Determine final columns
                        final_headers = headers.copy()
                        if 'Source Files' in data_rows[0] and 'Source Files' not in final_headers:
                            final_headers.append('Source Files')
                        
                        output_df = pd.DataFrame(data_rows, columns=final_headers)
                        
                        # Clean sheet name
                        sheet_name = re.sub(r'[^\w\s-]', '', template_name)[:31]
                        if not sheet_name.strip():
                            sheet_name = "COA_Data"
                        
                        output_df.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"Created sheet '{sheet_name}' with {len(data_rows)} products from {sum(r['transcript_count'] for r in results)} transcripts")
        
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