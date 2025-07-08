import json
import os
from typing import List, Dict, Any
import argparse
def create_batch_prompt(batch_analysis_files: List[str], question_set: Dict[str, Any],pqr_report_template:str) -> str:
    """
    Takes batch analysis JSON files and creates a prompt from them without cleaning or prettifying.
    
    Args:
        batch_analysis_files: List of paths to batch analysis JSON files
        question_set: Dictionary containing the question set structure
        
    Returns:
        String containing the prompt with all batch analysis data
    """
    # Start with the question set
    prompt = "Create a Product Quality Review (PQR) report and update the PQR report template with the following batch analysis data:\n\n"
    prompt += pqr_report_template
    prompt += "Based on the following batch analysis data, please answer the questions in the question set:\n\n"
    prompt += "QUESTION SET:\n"
    prompt += json.dumps(question_set, indent=2)
    prompt += "\n\nBATCH ANALYSIS DATA:\n\n"
    
    # Add each batch analysis file
    for i, file_path in enumerate(batch_analysis_files):
        try:
            with open(file_path, 'r') as f:
                batch_data = json.load(f)
                
                # Add file name as a header
                file_name = os.path.basename(file_path)
                prompt += f"=== BATCH ANALYSIS {i+1}: {file_name} ===\n"
                prompt += json.dumps(batch_data, indent=2)
                prompt += "\n\n"
        except Exception as e:
            prompt += f"Error loading file {file_path}: {str(e)}\n\n"
    
    return prompt

def main(base_path):
    # Example usage
    data_dir = os.path.join(base_path, "transcriptions")
    question_set_path = os.path.join(base_path, "transcription_metadata.json")
    
    # Load question set
    with open(question_set_path, 'r') as f:
        question_set = json.load(f)
    
    # Get all batch analysis files
    batch_analysis_files = []
    for file in os.listdir(data_dir):
        if file.endswith('_analysis.json'):
            batch_analysis_files.append(os.path.join(data_dir, file))
    
    # Create prompt
    prompt = create_batch_prompt(batch_analysis_files, question_set)
    
    # Save prompt
    output_path = os.path.join(base_path, "batch_prompt.txt")
    with open(output_path, 'w') as f:
        f.write(prompt)
    
    print(f"Batch prompt saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create batch prompt')
    parser.add_argument('--base_path', type=str, required=True, help='Path to the base directory')
    args = parser.parse_args()
    
    main(args.base_path) 