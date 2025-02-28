import os
import requests
import json
from PIL import Image
from io import BytesIO
import base64
import csv
import re

ollama_url = "http://localhost:11434/api/generate"
model_name = "llava:latest"
prompt = """Analyze the provided image and determine the construction progress for each stage.
give the output only in JSON format.
For EACH stage, identify which specific phase it's in:
Yet to Start (1 if true, 0 if false)
In Progress (1 if true, 0 if false)
Completed (1 if true, 0 if false)
Stages to analyze:

Block Work{
Yet to Start: Beam and slab are in progress, no brick work is done, and no brick work is visible.
In Progress: Partial walls built; some parts show partial brick work.
Completed: Full wall height achieved with no ongoing block laying.
}
Cement Plastering{
Yet to Start: Bare masonry with no plaster materials; no scaffolding/ladder for plaster.
In Progress: Partial plaster coverage, wet or unfinished surfaces; scaffolding in use, and tools/materials actively used.
Completed: Uniform plaster surface that is dry, with indications of the next stage.
}
Windows & Ventilators Fixing{
Yet to Start: No physical work has started; no window or ventilation materials are visible.
In Progress: Frames are aligned and marked for proper placement; openings are cleaned and adjusted; frames fixed using screws, fasteners, or adhesives.
Completed: All frames and panels are securely fixed, aligned, and sealed.
}
Electrical Switches & Sockets Installation{
Yet to Start: No switches, sockets, or lighting fixtures are installed; no electrical tools or equipment are in use.
In Progress: Workers are visible installing switches, lights, or appliances; some fixtures are installed but not all areas are complete; testing devices may be visible.
Completed: Fully installed and functional switches, sockets, and lighting fixtures with a clean, finished appearance.
}
Flooring{
Yet to Start: No even floor surface; visible cement layer or subfloor with debris.
In Progress: Flooring work has started, though not all images may show completion.
Completed: All flooring materials are securely installed with a clear pattern (e.g., tile, granite/marble, or wooden flooring).
}
Provide the output only in JSON format like this:
{
"image_id": "image_1",
"Block Work": {
"yet_to_start": 0 | 1,
"in_progress": 0 | 1,
"completed": 0 | 1
},
"Cement Plastering": {
"yet_to_start": 0 | 1,
"in_progress": 0 | 1,
"completed": 0 | 1
},
"Windows & Ventilators Fixing": {
"yet_to_start": 0 | 1,
"in_progress": 0 | 1,
"completed": 0 | 1
},
"Electrical Switches & Sockets Installation": {
"yet_to_start": 0 | 1,
"in_progress": 0 | 1,
"completed": 0 | 1
},
"Flooring": {
"yet_to_start": 0 | 1,
"in_progress": 0 | 1,
"completed": 0 | 1
}
}
Important: give the output only in the above format in json."""

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    
def query_ollama(image_base64):
    payload = {
        "model": model_name,
        "prompt": prompt,
        "image": [image_base64],
    }
    response = requests.post(ollama_url, json=payload)
    return response

def parse_json_response(response_text):
    """
    Try multiple methods to extract valid JSON from the response.
    """
    # Clean the response text by removing escape characters
    cleaned_text = re.sub(r'\\', '', response_text)
    
    # Method 1: Try to directly parse the cleaned text
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    
    # Method 2: Look for markdown code blocks and fix incomplete JSON
    try:
        # Extract content between markdown code blocks
        if "```json" in cleaned_text and "```" in cleaned_text.split("```json")[1]:
            json_part = cleaned_text.split("```json")[1].split("```")[0].strip()
            
            # Check if JSON is missing closing brace
            if json_part.count('{') > json_part.count('}'):
                json_part += "}"
                
            return json.loads(json_part)
    except (IndexError, json.JSONDecodeError):
        pass
    
    # Method 3: Fix incomplete JSON when code block is incomplete
    try:
        if "```json" in cleaned_text:
            json_part = cleaned_text.split("```json")[1].strip()
            
            # Check if JSON is missing closing brace
            if json_part.count('{') > json_part.count('}'):
                json_part += "}"
                
            return json.loads(json_part)
    except (IndexError, json.JSONDecodeError):
        pass
    
    # Method 4: Try to find JSON-like structure with regex
    try:
        json_match = re.search(r'(\{.*?\"Flooring\".*?\})', cleaned_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            # Check if JSON is missing closing brace
            if json_str.count('{') > json_str.count('}'):
                json_str += "}"
            return json.loads(json_str)
    except (AttributeError, json.JSONDecodeError):
        pass
    
    # Method 5: More aggressive pattern matching, specifically for the structure we know we should have
    try:
        # Look for pattern with all expected keys
        pattern = r'.*?image_id.*?Block Work.*?Cement Plastering.*?Windows & Ventilators Fixing.*?Electrical Switches & Sockets Installation.*?Flooring.*?'
        if re.match(pattern, cleaned_text, re.DOTALL):
            # Try to reconstruct valid JSON
            json_str = re.sub(r'```json|```', '', cleaned_text).strip()
            
            # Ensure proper JSON structure
            if not json_str.endswith('}'):
                # Add missing closing brace for the last element (Flooring)
                if '"completed": ' in json_str and not json_str.endswith('}'):
                    last_completed = json_str.rfind('"completed": ')
                    end_idx = json_str.find('\n', last_completed)
                    if end_idx == -1:  # If no newline found, use the end of string
                        end_idx = len(json_str)
                    
                    json_str = json_str[:end_idx] + "}" + json_str[end_idx:]
                
                # Add missing closing brace for the root object
                json_str += "}"
                
            return json.loads(json_str)
    except (json.JSONDecodeError):
        pass
    
    # Method 6: Manual parsing as a last resort
    try:
        # Find keys and values manually
        image_id_match = re.search(r'"image_id":\s*"([^"]+)"', cleaned_text)
        
        if image_id_match:
            result = {"image_id": image_id_match.group(1)}
            
            # Extract stages data
            stages = ["Block Work", "Cement Plastering", "Windows & Ventilators Fixing", 
                     "Electrical Switches & Sockets Installation", "Flooring"]
            
            for stage in stages:
                stage_data = {}
                
                # Look for the three possible states for each stage
                states = ["yet_to_start", "in_progress", "completed"]
                for state in states:
                    pattern = f'"{re.escape(stage)}".*?"{state}":\s*(\d)'
                    state_match = re.search(pattern, cleaned_text, re.DOTALL)
                    if state_match:
                        stage_data[state] = int(state_match.group(1))
                    else:
                        # Default value if not found
                        stage_data[state] = 0
                
                result[stage] = stage_data
            
            return result
    except Exception:
        pass
    
    # If all methods fail, raise an exception
    raise ValueError(f"Could not extract valid JSON from: {cleaned_text}")

def get_processed_images(csv_file):
    """
    Get a list of image IDs that have already been processed.
    """
    processed_images = set()
    
    if not os.path.exists(csv_file):
        return processed_images
        
    try:
        with open(csv_file, 'r', newline='') as f:
            reader = csv.reader(f)
            # Skip header row
            next(reader, None)
            for row in reader:
                if row and len(row) > 0:
                    # Extract image number from image ID (e.g., "image_5" -> 5)
                    match = re.search(r'image_(\d+)', row[0])
                    if match:
                        image_number = int(match.group(1))
                        processed_images.add(image_number)
    except Exception as e:
        print(f"Error reading existing CSV: {str(e)}")
    
    return processed_images

# Define output CSV file
output_file = "construction_progress_results.csv"

# Define headers for CSV
headers = [
    "image_id",
    "block_work_yet_to_start", "block_work_in_progress", "block_work_completed",
    "cement_plastering_yet_to_start", "cement_plastering_in_progress", "cement_plastering_completed",
    "windows_&_ventilators_fixing_yet_to_start", "windows_&_ventilators_fixing_in_progress", "windows_&_ventilators_fixing_completed",
    "electrical_switches_&_sockets_installation_yet_to_start", "electrical_switches_&_sockets_installation_in_progress", "electrical_switches_&_sockets_installation_completed",
    "flooring_yet_to_start", "flooring_in_progress", "flooring_completed"
]

# Check if CSV exists and get processed images
processed_images = get_processed_images(output_file)
print(f"Found {len(processed_images)} already processed images")

# Create new CSV with headers if it doesn't exist
if not os.path.exists(output_file):
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

# Process each image
image_paths = [f"D:/Office/images/train/400/{i}.jpg" for i in range(1, 61)]

for i, image_path in enumerate(image_paths):
    # Extract image number from path
    image_num = i + 1
    
    # Skip if already processed
    if image_num in processed_images:
        print(f"Skipping already processed image {image_num}")
        continue
    
    try:
        print(f"Processing image {image_num}/{len(image_paths)}: {image_path}")
        
        # Encode image
        image_base64 = encode_image(image_path)
        
        # Query Ollama
        response = query_ollama(image_base64)
        
        # Collect full response
        collected_response = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                collected_response += data["response"]
        
        # Save raw response for debugging
        with open(f"raw_response_{image_num}.txt", "w") as f:
            f.write(collected_response)
        
        # Parse JSON from response
        try:
            json_data = parse_json_response(collected_response)
            print(f"Successfully parsed JSON for image {image_num}")
            
            # Extract data for CSV
            image_id = f"image_{image_num}"  # Ensure consistent image_id format
            
            block_work = json_data.get("Block Work", {})
            cement_plastering = json_data.get("Cement Plastering", {})
            windows_ventilators_fixing = json_data.get("Windows & Ventilators Fixing", {})
            electrical_switches_sockets_installation = json_data.get("Electrical Switches & Sockets Installation", {})
            flooring = json_data.get("Flooring", {})
            
            # Write row to CSV
            with open(output_file, "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    image_id,
                    block_work.get("yet_to_start", ""), block_work.get("in_progress", ""), block_work.get("completed", ""),
                    cement_plastering.get("yet_to_start", ""), cement_plastering.get("in_progress", ""), cement_plastering.get("completed", ""),
                    windows_ventilators_fixing.get("yet_to_start", ""), windows_ventilators_fixing.get("in_progress", ""), windows_ventilators_fixing.get("completed", ""),
                    electrical_switches_sockets_installation.get("yet_to_start", ""), electrical_switches_sockets_installation.get("in_progress", ""), electrical_switches_sockets_installation.get("completed", ""),
                    flooring.get("yet_to_start", ""), flooring.get("in_progress", ""), flooring.get("completed", "")
                ])
            
            # Add to processed images to handle case where script runs multiple images in one execution
            processed_images.add(image_num)
                
        except ValueError as e:
            print(f"Error processing image {image_num}: {e}")
            # Log the error to a file
            with open("error_log.txt", "a") as error_file:
                error_file.write(f"Error on image {image_path}: {str(e)}\n")
                error_file.write(f"Response: {collected_response}\n\n")
            
    except Exception as e:
        print(f"Unexpected error processing image {image_num}: {str(e)}")
        # Log the error to a file
        with open("error_log.txt", "a") as error_file:
            error_file.write(f"Unexpected error on image {image_path}: {str(e)}\n\n")

print(f"Processing complete. Results saved to {output_file}")