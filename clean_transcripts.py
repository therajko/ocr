import os
import re
project_name = "atv_reports_v2"
transcript_dir = "/data/transcriptions"
cleaned_project_name = project_name.replace('_','-')
index_name = f"{cleaned_project_name}-transcripts"
transcript_location = f"{project_name}{transcript_dir}"
transcript_location_cleaned = f"{project_name}{transcript_dir}_cleaned"

all_transcripts = [i for i in os.listdir(transcript_location) if i.endswith(".txt")]

def clean_repeated_patterns(text):
    """Clean up repeated patterns like 'x x x x' or '| | |' that might appear from OCR artifacts."""
    # First, clean up any long sequences of 'x' with or without spaces
    cleaned = re.sub(r'(?:\s*x\s*){3,}', '', text)
    # Then clean up any remaining sequences of just 'x' characters
    cleaned = re.sub(r'x{5,}', '', cleaned)
    # Clean up repeated pipe characters (outside of valid table rows)
    cleaned = re.sub(r'(?:\s*\|\s*){3,}(?!\s*\w)', '', cleaned)
    # Clean up any resulting extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

def clean_page_content(content):
    """
    Clean and normalize page content for better RAG performance while preserving structure.
    
    Args:
        content (str): Raw page content
        
    Returns:
        str: Cleaned and normalized content
    """
    import re
    
    def normalize_whitespace(text):
        """Normalize whitespace in text while preserving markdown."""
        # Preserve bold markers
        text = text.replace('** ', '**')
        text = text.replace(' **', '**')
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Clean any repeated x patterns
        text = clean_repeated_patterns(text)
        return text.strip()
    
    def clean_table_row(row):
        """Clean a table row and check if it contains actual content."""
        # Skip if row is just repeated pipes
        if re.match(r'^[\s|]*$', row):
            return None
        
        # Remove leading/trailing pipes and split into cells
        cells = [cell.strip() for cell in row.split('|')[1:-1]]
        # Remove empty cells
        cells = [cell for cell in cells if cell]
        
        # Check if any cell has actual content (not just separators or whitespace)
        has_content = any(bool(re.sub(r'[\s\-\|:]+', '', cell)) for cell in cells)
        
        if has_content and cells:
            # Preserve bold markers in cells and clean repeated patterns
            cells = [clean_repeated_patterns(cell.replace('** ', '**').replace(' **', '**')) for cell in cells]
            
            # If it's a single cell with timestamps, format it properly
            if len(cells) > 1 and all(re.match(r'^\d{8}\s+\d{2}[:.]\d{2}$', cell) for cell in cells):
                return cells[0] + ' - ' + cells[-1]
            
            return '| ' + ' | '.join(cells) + ' |'
        return None

    def is_separator_row(line):
        """Check if line is a table separator."""
        return bool(re.match(r'^\s*\|[\s\-\|:]*\|\s*$', line))

    # Split content into lines
    lines = content.split('\n')
    cleaned_lines = []
    current_table_lines = []
    in_table = False
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines unless they're needed for structure
        if not line:
            if in_table:
                # Process the collected table
                if current_table_lines:
                    cleaned_lines.extend(current_table_lines)
                    cleaned_lines.append('')  # Add single spacing after table
                current_table_lines = []
                in_table = False
            elif cleaned_lines and cleaned_lines[-1]:  # Add single line between sections
                cleaned_lines.append('')
            i += 1
            continue
        
        # Handle standalone text (non-table content)
        if not line.startswith('|'):
            if in_table:
                # Process any collected table content first
                if current_table_lines:
                    cleaned_lines.extend(current_table_lines)
                    cleaned_lines.append('')
                current_table_lines = []
                in_table = False
            
            # Clean and add non-table text
            cleaned_text = normalize_whitespace(line)
            if cleaned_text:
                cleaned_lines.append(cleaned_text)
            i += 1
            continue
        
        # Handle table content
        if line.startswith('|'):
            if not in_table:
                in_table = True
                current_table_lines = []
            
            # Skip separator rows
            if is_separator_row(line):
                i += 1
                continue
            
            cleaned_row = clean_table_row(line)
            if cleaned_row:
                current_table_lines.append(cleaned_row)
            i += 1
            continue
        
        i += 1
    
    # Process any remaining table content
    if current_table_lines:
        cleaned_lines.extend(current_table_lines)
    
    # Remove any trailing empty lines
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()
    
    # Join lines with single newlines
    cleaned_content = '\n'.join(line for line in cleaned_lines if line.strip())
    
    return cleaned_content

# Create cleaned directory if it doesn't exist
if not os.path.exists(transcript_location_cleaned):
    os.makedirs(transcript_location_cleaned)

# Process all transcripts
for transcript in all_transcripts:
    with open(f"{transcript_location}/{transcript}", "r") as f:
        content = f.read()
        cleaned_content = clean_page_content(content)
        
        with open(f"{transcript_location_cleaned}/{transcript}", "w") as f:
            f.write(cleaned_content)