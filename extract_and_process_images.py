import re
import os
import base64
import json
import pandas as pd
from openai import OpenAI
import openai
import pandas as pd
import io
from markdown_it import MarkdownIt
from create_batch_prompt import create_batch_prompt
import argparse
import os
import json
import time
from template_processor import get_model_response

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_image_urls(html_file_path):
    """
    Extract image URLs from the HTML file.
    
    Args:
        html_file_path (str): Path to the HTML file containing image tags
        
    Returns:
        list: List of image URLs
    """
    with open(html_file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Extract URLs using regex
    pattern = r'<img src="(https://[^"]+)"'
    urls = re.findall(pattern, content)
    
    return urls

def process_images_with_model(images_local_path, output_dir, model_name,text_instructions = "Transcribe this document in markdown"):
    """
    Process images with the model API and save transcriptions.
    
    Args:
        images_local_path (str): Path to the directory containing local images
        output_dir (str): Directory to save transcriptions
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize OpenAI client
    if model_name == 'openai':
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
    else:
        client = OpenAI(
            base_url=os.environ.get("HF_QWEN2_5_VL_MODEL_ENDPOINT"),
            api_key=os.environ.get("HUGGINGFACE_KEY")
        )
        
    # Get list of local images
    images = [f'{images_local_path}/{i}' for i in os.listdir(images_local_path) if i.endswith(".jpg") or i.endswith(".png") or i.endswith(".jpeg")]
    total_images = len(images)
    
    # Process each image
    for idx, img_path in enumerate(images, 1):
        # Check if transcription already exists
        filename = os.path.basename(img_path).replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
        output_file = os.path.join(output_dir, filename)
        
        if os.path.exists(output_file):
            print(f"Skipping {img_path} - transcription already exists")
            continue
            
        print(f"Processing image {idx}/{total_images}: {img_path}")
        
        base64_image = encode_image(img_path)
        prompt = [
            {
                "role": "user", 
                "content": [
                    {
                        "type": "input_image",
                        "image_url":  f"data:image/jpg;base64,{base64_image}",
                    },
                    {
                        "type": "input_text",
                        "text": text_instructions
                    },
                ],
            }
        ]
        
        # Get model response using the imported function
        content = get_model_response(client, model_name, prompt)
        time.sleep(3)
        
        # Save transcribed text to file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"\nTranscription saved to {output_file}\n")

def combine_transcripts(output_dir):
    """Combine all transcript files into a single string"""
    combined_content = ""
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            with open(os.path.join(output_dir, filename), 'r') as f:
                combined_content += f.read() + "\n\n---\n\n"
    return combined_content

def merge_unique_values(values):
    """Merge values keeping unique items"""
    if not values:
        return ""
    
    # Handle case where values is a list of strings
    if all(isinstance(v, str) for v in values):
        unique_values = set()
        for v in values:
            if v and str(v).strip() and str(v).strip() != "Not specified":
                unique_values.add(str(v).strip())
        return sorted(unique_values) if unique_values else []
    
    # Handle case where values is a list of dictionaries
    if all(isinstance(v, dict) for v in values):
        merged = {}
        for d in values:
            for k, v in d.items():
                if k not in merged:
                    merged[k] = []
                merged[k].append(v)
        return {k: merge_unique_values(v) for k, v in merged.items()}
    
    # Handle case where values is a list of lists
    if all(isinstance(v, list) for v in values):
        merged = []
        for lst in values:
            merged.extend(lst)
        
        # Check if the merged list contains dictionaries
        if merged and all(isinstance(item, dict) for item in merged):
            # Get all keys from all dictionaries
            all_keys = set()
            for item in merged:
                all_keys.update(item.keys())
            
            # Create a new dictionary with merged values for each key
            result = {}
            for key in all_keys:
                key_values = [item.get(key) for item in merged if key in item]
                result[key] = merge_unique_values(key_values)
            return result
        else:
            # If not dictionaries, just return unique values
            return list(set(merged))
    
    # If none of the above, return the first value
    return values[0]

def combine_analysis_results(output_dir, questionnaire):
    """Combine all analysis JSONs into a master dataframe"""
    all_results = []
    
    # Read and clean all analysis JSONs
    for filename in os.listdir(output_dir):
        if filename.endswith('_analysis.json'):
            with open(os.path.join(output_dir, filename), 'r') as f:
                analysis = json.load(f)
                
                
                # Add filename as a column for reference
                analysis['source_file'] = filename
                all_results.append(analysis)
    
    if not all_results:
        print("No analysis files found in the output directory.")
        return None
    
    # Flatten nested dictionaries and create a list of records
    flattened_records = []
    
    for result in all_results:
        record = {}
        record['source_file'] = result.get('source_file', '')
        
        # Process each top-level key
        for key, value in result.items():
            if key == 'source_file':
                continue
                
            if isinstance(value, dict):
                # Flatten nested dictionaries
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, dict):
                        # Handle double-nested dictionaries
                        for double_nested_key, double_nested_value in nested_value.items():
                            column_name = f"{key}_{nested_key}_{double_nested_key}"
                            record[column_name] = double_nested_value
                    else:
                        column_name = f"{key}_{nested_key}"
                        record[column_name] = nested_value
            else:
                # Direct values
                record[key] = value
        
        flattened_records.append(record)
    
    # Create a dataframe from the flattened records
    df = pd.DataFrame(flattened_records)
    
    # Save the dataframe to CSV
    csv_file = os.path.join(output_dir, 'combined_analysis.csv')
    df.to_csv(csv_file, index=False)
    
    # Save the dataframe to Excel for better formatting
    excel_file = os.path.join(output_dir, 'combined_analysis.xlsx')
    df.to_excel(excel_file, index=False)
    
    print(f"\nCombined analysis saved to {csv_file} and {excel_file}")
    return df

def process_transcripts_with_llm(output_dir, metadata_file, questionnaire,base_url,model_name):
    """Process each transcript with LLM to extract structured information"""
    # Initialize OpenAI client
    if model_name == 'openai':
        client = OpenAI(
         
            api_key=os.environ.get("OPENAI_API_KEY")
        )
    else:
        client = OpenAI(
            base_url=base_url,
            api_key=os.environ.get("HUGGINGFACE_KEY")
        )

    results = []
    transcripts = []
    current_batch = ""
    current_batch_size = 0
    
    # First pass - collect and batch transcripts
    for filename in os.listdir(output_dir):
        if filename.endswith('.txt'):
            # Check if analysis already exists
            analysis_file = os.path.join(output_dir, f'batch_{len(transcripts) + 1}_analysis.json')
            if os.path.exists(analysis_file):
                print(f"Skipping batch {len(transcripts) + 1} - analysis already exists")
                with open(analysis_file, 'r') as f:
                    try:
                        results.append(json.load(f))
                    except:
                        pass
                continue
                
            with open(os.path.join(output_dir, filename), 'r') as f:
                transcript = f.read()
                # Rough estimate of token count (4 chars ≈ 1 token)
                token_estimate = len(transcript) // 4
                
                if current_batch_size + token_estimate > 12000:  # Leave room for prompt
                    transcripts.append(current_batch)
                    current_batch = transcript
                    current_batch_size = token_estimate
                else:
                    if current_batch:
                        current_batch += "\n\n---\n\n" + transcript
                    else:
                        current_batch = transcript
                    current_batch_size += token_estimate
    
    if current_batch:
        transcripts.append(current_batch)

    # Process each batch
    for batch_idx, batch in enumerate(transcripts):
        print(f"\nProcessing batch {batch_idx + 1}/{len(transcripts)}")
        
        prompt = f"""Based on the following batch of documents, please extract pharmaceutical information for a Product Quality Review (PQR) report. Important:

                    1. Return ONLY a valid JSON object with non-empty values
                    2. For each field in the template:
                    - Extract information ONLY if it EXACTLY matches the field definition
                    - Skip any field where information is missing, uncertain, or doesn't meet criteria
                    - Do NOT include empty/null fields in the output JSON
                    3. If multiple candidates exist for a field, select all of them
                    4. Combine and consolidate information from all documents in the batch
                    5. For batch sizes and pack styles, include all unique values found
                    6. For dates, use the most recent one if multiple exist
                    7. For specifications, use the most stringent values

                    Template fields to extract (ONLY include fields where you find valid matching information):
                {questionnaire}
                
                Documents:  {batch}
                It is not necessary to include all the fields in the questionnaire, only include the ones that are present in the documents. Skip the key value pair for which you are not able to find any information in the documents. The idea is to extract the most relevant information from the documents."""
        
    
        retries=0
        while retries<3:
            try:
                # Get model response using the imported function
                content = get_model_response(client, model_name, prompt)
                time.sleep(3)

                matches = re.search(r"\{.*\}", content, re.DOTALL)
                answers = matches.group(0) if matches else ""

                
                # Save batch results
                result_file = os.path.join(output_dir, f'batch_{batch_idx + 1}_analysis.json')
                
                
                try:
                    json_data = json.loads(answers)
                
                    with open(result_file, 'w') as f:
                        json.dump(json_data, f, indent=2)
                except:
                    with open(result_file,'w') as f:
                        f.write(answers)    
                results.append(json_data)
                print(f"Batch {batch_idx + 1} analysis saved to {result_file}")
                break
            except Exception as e:
                retries += 1
                print(f"Error processing batch {batch_idx + 1}: {e}. Retrying...{retries} times")

    return results


def reprocess_batches_with_llm(output_dir, questionnaire, group_size,base_url):
    """Reprocess batch-level outputs into higher-level grouped batches"""
    client = OpenAI(
        base_url=base_url,
        api_key=os.environ.get("HUGGINGFACE_KEY")
    )
    
    # Load all batch result files
    batch_files = sorted([
        f for f in os.listdir(output_dir)
        if f.startswith('batch_') and f.endswith('_analysis.json')
    ])

    grouped_results = []
    for i in range(0, len(batch_files), group_size):
        group = batch_files[i:i+group_size]
        group_file = os.path.join(output_dir, f'group_{i//group_size + 1}_consolidated.json')
        
        # Check if group consolidation already exists
        if os.path.exists(group_file):
            print(f"Skipping group {i//group_size + 1} - consolidation already exists")
            with open(group_file, 'r') as f:
                try:
                    grouped_results.append(json.load(f))
                except:
                    pass
            continue
            
        print(f"\nProcessing group {i//group_size + 1} with batches: {group}")

        combined_data = []
        for file in group:
            with open(os.path.join(output_dir, file), 'r') as f:
                try:
                    data = json.load(f)
                    combined_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"Skipping file {file}: {e}")
        
        if not combined_data:
            continue
        
        prompt = f"""You are given a list of extracted JSON objects representing pharmaceutical PQR information, each from a separate batch. Your task is to consolidate them into a single, unified JSON by:

1. Merging non-conflicting values (for fields like `batch_sizes`, `pack_styles`, include all unique values across batches)
2. Selecting the most recent date where multiple dates exist
3. Choosing the most stringent specification values if conflicting
4. Ignoring any empty or null fields
5. Outputting a single clean JSON object with only valid, non-empty fields

Template (for reference): {questionnaire}

Extracted JSONs from batches:
{json.dumps(combined_data, indent=2)}"""

        retries = 0
        while retries < 3:
            try:
                response = client.responses.create(
                    model='tgi',
                    input=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.2
                )
                time.sleep(3)
                merged = response.output_text
                matches = re.search(r"\{.*\}", merged, re.DOTALL)
                merged_content = matches.group(0) if matches else ""
                try:
                    merged_json = json.loads(merged_content)
                except:
                    merged_json = merged_content

                with open(group_file, 'w') as f:
                    json.dump(merged_json, f, indent=2)

                grouped_results.append(merged_json)
                print(f"Group {i//group_size + 1} consolidated output saved to {group_file}")
                break

            except Exception as e:
                retries += 1
                print(f"Error consolidating group {i//group_size + 1}: {e}. Retrying... ({retries}/3)")

    return grouped_results


def create_batch_analysis_prompt(output_dir, metadata_file,pqr_report_template_file):
    """
    Create a prompt from batch analysis files using the create_batch_prompt function.
    
    Args:
        output_dir (str): Directory containing batch analysis files
        metadata_file (str): Path to the metadata file containing the question set
        
    Returns:
        str: Path to the generated prompt file
    """
    # Load question set
    with open(metadata_file, 'r') as f:
        question_set = json.load(f)
    
    with open(pqr_report_template_file, 'r') as f:
        pqr_report_template = f.read()
    
    # Get all batch analysis files
    batch_analysis_files = []
    for file in os.listdir(output_dir):
        if file.endswith('_analysis.json'):
            batch_analysis_files.append(os.path.join(output_dir, file))
    
    # Create prompt
    prompt = create_batch_prompt(batch_analysis_files, question_set,pqr_report_template)
    
    # Save prompt
    output_path = os.path.join(output_dir, "group_consolidated_analysis_prompt.txt")
    with open(output_path, 'w') as f:
        f.write(prompt)
    
    print(f"Batch analysis prompt saved to {output_path}")
    return output_path

def get_pqr_output(model_url,pqr_prompt_file,output_dir,model_name):
    """
    Get the PQR output from the batch analysis prompt.
    """
    with open(pqr_prompt_file, 'r') as f:
        pqr_output = f.read()
    
    if model_name == 'openai':
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )
    else:
        client = OpenAI(
            base_url=model_url,
            api_key=os.environ.get("HUGGINGFACE_KEY")
        )
    
    # Get model response using the imported function
    pqr_output = get_model_response(client, model_name, pqr_output)
    time.sleep(3)
    
    with open(os.path.join(output_dir, "pqr_output.md"), 'w') as f:
        f.write(pqr_output)
    return pqr_output

def main(base_path, model_name):
    # Output directory for transcriptions
    output_dir = os.path.join(base_path, "transcriptions")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load questionnaire if it exists
    metadata_file = os.path.join(base_path, "transcription_metadata.json")
    
    if os.path.exists(metadata_file):
        with open(metadata_file, "r") as f:
            questionnaire = f.read()
    else:
        questionnaire = "{}"  # Default empty JSON if file doesn't exist
    
    # Path to local images
    images_local_path = os.path.join(base_path, "images")
    
    # Output path for combined document
    combined_output_path = os.path.join(base_path, "transcriptions/combined_document.md")
    
    # Path to PQR report template file
    pqr_report_template_file = os.path.join(base_path, "pqr_template.md")
    
    # Check if all transcriptions exist
    all_transcriptions_exist = True
    for img_file in os.listdir(images_local_path):
        if img_file.endswith(('.jpg', '.png', '.jpeg')):
            txt_file = img_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
            if not os.path.exists(os.path.join(output_dir, txt_file)):
                all_transcriptions_exist = False
                break
    
    if not all_transcriptions_exist:
        # Process images with model
        print("\nProcessing images with model API...")
        process_images_with_model(images_local_path, output_dir, model_name)
    else:
        print("\nAll transcriptions already exist. Skipping image processing.")
    
    # Check if all batch analyses exist
    all_batch_analyses_exist = True
    batch_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    for i in range(1, len(batch_files) + 1):
        if not os.path.exists(os.path.join(output_dir, f'batch_{i}_analysis.json')):
            all_batch_analyses_exist = False
            break
    
    if not all_batch_analyses_exist:
        # Process transcripts with LLM
        print("\nProcessing transcripts with LLM...")
        summarization_model_url = os.environ.get("HF_LLAMA_GGUF_MODEL_ENDPOINT")
        group_size = 2
        analysis_results = process_transcripts_with_llm(output_dir, metadata_file, questionnaire, summarization_model_url,model_name)
    else:
        print("\nAll batch analyses already exist. Skipping transcript processing.")
    
    # Check if all group consolidations exist
    all_group_consolidations_exist = True
    batch_files = [f for f in os.listdir(output_dir) if f.startswith('batch_') and f.endswith('_analysis.json')]
    for i in range(0, len(batch_files), group_size):
        if not os.path.exists(os.path.join(output_dir, f'group_{i//group_size + 1}_consolidated.json')):
            all_group_consolidations_exist = False
            break
    
    # if not all_group_consolidations_exist:
    #     reprocessed_batches = reprocess_batches_with_llm(output_dir, questionnaire, group_size, summarization_model_url)
    # else:
    #     print("\nAll group consolidations already exist. Skipping reprocessing.")
    
    # Check if final results exist
    # if not os.path.exists(os.path.join(output_dir, 'combined_analysis.csv')):
    #     # Combine and clean analysis results
    #     print("\nCombining analysis results...")
    #     final_results = combine_analysis_results(output_dir, questionnaire)
    # else:
    #     print("\nCombined analysis already exists. Skipping combination.")
    
    # Check if PQR template exists and if PQR output needs to be generated
    if os.path.exists(pqr_report_template_file):
        pqr_output_path = os.path.join(output_dir, "pqr_output.md")
        if not os.path.exists(pqr_output_path):
            # Create batch analysis prompt
            print("\nCreating batch analysis prompt...")
            batch_prompt_path = create_batch_analysis_prompt(output_dir, metadata_file, pqr_report_template_file)
            print(f"Batch analysis prompt created at: {batch_prompt_path}")
            
            pqr_prompt_file = os.path.join(output_dir, "group_consolidated_analysis_prompt.txt")
            get_pqr_output(summarization_model_url, pqr_prompt_file, output_dir, model_name)
        else:
            print("\nPQR output already exists. Skipping PQR generation.")
    else:
        print(f"PQR template file not found at {pqr_report_template_file}. Skipping PQR generation.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and process images')
    parser.add_argument('--base_path', type=str, required=True, help='Path to the base directory')
    parser.add_argument('--model_name', type=str, required=False, default='local_llm', help='Model name')
    args = parser.parse_args()
    
    main(args.base_path, args.model_name)