import base64
import os
from pathlib import Path

def image_to_base64(image_path):
    """
    Convert a local image file to a base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded string of the image
    """
    # Check if file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Read the image file
    with open(image_path, "rb") as image_file:
        # Encode the image as base64
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
    
    return encoded_string

def prepare_image_for_api(image_path):
    """
    Prepare an image for sending to an API that accepts base64 images.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary with the image data in the format expected by most APIs
    """
    base64_image = image_to_base64(image_path)
    
    # Format for OpenAI's API (and many others)
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
        }
    }

# Example usage
if __name__ == "__main__":
    # Example: Convert an image from the data folder
    try:
        # Assuming you have an image in the transcribe_pdf/data folder
        image_path = "transcribe_pdf/data/example.jpg"
        image_data = prepare_image_for_api(image_path)
        print("Image successfully converted to base64")
        print(f"Base64 string length: {len(image_data['image_url']['url'])}")
    except Exception as e:
        print(f"Error: {e}") 