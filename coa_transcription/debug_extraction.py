#!/usr/bin/env python3

import os
import sys
from structured_processor import StructuredCOAProcessor

def test_extraction():
    """
    Test the extraction process step by step to debug issues.
    """
    print("=== DEBUGGING COA EXTRACTION ===")
    
    # Initialize processor
    print("\n1. Initializing processor...")
    processor = StructuredCOAProcessor(templates_dir="./output_format")
    print(f"Found {len(processor.templates)} templates:")
    for name in processor.templates.keys():
        print(f"  - {name}")
    
    # Test transcript grouping
    print("\n2. Testing transcript grouping...")
    transcripts_dir = "./transcriptions"
    groups = processor.group_transcripts_by_product(transcripts_dir)
    
    if not groups:
        print("ERROR: No groups found!")
        return
    
    # Test extraction for first group
    print("\n3. Testing data extraction...")
    first_group_key = list(groups.keys())[0]
    first_group = groups[first_group_key]
    
    print(f"Testing with group: {first_group_key}")
    print(f"Number of transcripts: {len(first_group)}")
    
    # Combine content
    combined_content = []
    for transcript in first_group:
        combined_content.append(f"=== Content from {transcript['file'].name} ===")
        combined_content.append(transcript['content'])
        combined_content.append("\n" + "="*50 + "\n")
    
    full_content = "\n".join(combined_content)
    print(f"Combined content length: {len(full_content)} characters")
    
    # Identify template
    template_name, template_data = processor.identify_product_template(full_content)
    print(f"Selected template: {template_name}")
    
    if template_data:
        print(f"Template has {len(template_data['headers'])} columns")
        print("First 10 columns:")
        for i, header in enumerate(template_data['headers'][:10]):
            print(f"  {i+1}. {header}")
    
    # Test AI extraction
    print("\n4. Testing AI extraction...")
    if template_name and template_data:
        # Show a sample of the content being sent to AI
        print("Sample content being sent to AI:")
        print("-" * 50)
        print(full_content[:500] + "..." if len(full_content) > 500 else full_content)
        print("-" * 50)
        
        extracted_data = processor.extract_data_with_ai(template_name, template_data, full_content)
        
        if extracted_data:
            print(f"✓ Successfully extracted {len(extracted_data)} fields!")
            print("Sample extracted data:")
            for key, value in list(extracted_data.items())[:10]:
                print(f"  {key}: {value}")
        else:
            print("✗ Extraction failed!")
    else:
        print("No template found!")

if __name__ == "__main__":
    test_extraction() 