from pinecone import Pinecone, ServerlessSpec
import os
import time
import re
import json

pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

project_name = "atv_reports_v2"
cleaned_project_name = project_name.replace('_','-')
index_name = f"{cleaned_project_name}-transcripts"
transcript_location = f"{project_name}/data/transcriptions_cleaned"
pinecone_upsert_batch_size = 60
name_space = "all_text_transcripts_page_embeddings"

def clean_page_content(content):
    """Clean the entire page content at once."""
    # Split into lines and remove empty lines
    lines = [line.strip() for line in content.split('\n')]
    lines = [line for line in lines if line]
    
    # Remove lines that are just separators or empty cells
    cleaned_lines = []
    for line in lines:
        # Skip separator lines (|---|---|)
        if re.match(r'^\s*\|[\s\-\|]*\|\s*$', line):
            continue
        # Skip lines that are just empty table cells
        if re.match(r'^\s*\|(\s*\|\s*)*$', line):
            continue
        # Skip lines that are just whitespace or punctuation
        if not re.sub(r'[\s\|\-\:\.]', '', line):
            continue
        cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)

def process_files():
    """Process all files and create embeddings."""
    if not pc.has_index(index_name):
        print(f"Creating new index: {index_name}")
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"}
            }
        )
    
    records = []
    print(f"Processing files from: {transcript_location}")
    
    # Walk through all directories
    for root, _, files in os.walk(transcript_location):
        for file_name in files:
            if file_name.endswith(".txt"):
                print(f"Processing file: {file_name}")
                file_path = os.path.join(root, file_name)
                cleaned_file_name = file_name.replace(".txt", "").replace('-','_').replace(' ','_').lower()
                
                with open(file_path, "r") as f:
                    content = f.read()
                    cleaned_content = clean_page_content(content)
                    
                    if cleaned_content:  # Only create record if page has content
                        records.append({
                            "_id": cleaned_file_name,
                            "chunk_text": json.dumps({
                                "file_name": file_name,
                                "content": cleaned_content
                            })
                        })
                        print(f"Processed page: {file_name}")
    
    print(f"Total records to be embedded: {len(records)}")
    return records

def batch_upsert_to_pinecone(records, batch_size=60):
    """Batch upsert records to Pinecone with proper error handling and logging."""
    dense_index = pc.Index(index_name)
    total_batches = (len(records) + batch_size - 1) // batch_size
    
    print(f"Starting to upsert {len(records)} records in {total_batches} batches")
    
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        try:
            print(f"Upserting batch {i//batch_size + 1}/{total_batches} with {len(batch)} records")
            dense_index.upsert_records(name_space, batch)
            # Small delay to avoid rate limiting
            time.sleep(0.5)
        except Exception as e:
            print(f"Error upserting batch {i//batch_size + 1}: {str(e)}")
            # Log the problematic records for debugging
            print(f"Problematic records in batch: {json.dumps(batch, indent=2)}")
            raise

def main():
    try:
        records = process_files()
        if records:
            batch_upsert_to_pinecone(records, pinecone_upsert_batch_size)
            print("Successfully completed upserting all records")
        else:
            print("No records found to process")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
      

