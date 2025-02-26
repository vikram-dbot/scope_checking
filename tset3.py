import os
import torch
import csv
import json
import re
import gc
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import pandas as pd
from PIL import Image

# Define the mapping of expected outputs to binary values
expected_phrases = {"Not Found": 0, "Yet to Start": 0, "Work in Progress": 1, "Completed": 1}

def load_model_and_processor():
    """Load model with memory optimization settings"""
    # Set up GPU memory optimization configurations
    torch.cuda.empty_cache()
    
    # Load model with memory efficiency options
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct-AWQ", 
        torch_dtype=torch.float16,  # Use fp16 precision to reduce memory usage
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    return model, processor

def parse_json_output(output_text):
    """Extract and parse JSON from the model output that might have prefixes or formatting."""
    try:
        # If the output already starts with a valid JSON character
        if output_text.strip().startswith('{'):
            return json.loads(output_text)
        
        # Look for json followed by a JSON object
        json_match = re.search(r'json\s*\n*\s*(\{.*\})', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
            
        # If we still can't find it, look for any valid JSON object in the text
        json_match = re.search(r'(\{.*\})', output_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            return json.loads(json_str)
            
        # If all else fails, raise an exception
        raise ValueError(f"Could not extract valid JSON from: {output_text}")
    
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw output: {output_text}")
        # Return a default structure if parsing fails
        return {
            "image_id": "error",
            "Block Work": 0,
            "Cement Plastering": 0,
            "Windows & Ventilators Fixing": 0,
            "Electrical Switches & Sockets Installation": 0,
            "Flooring": 0
        }

def preprocess_image(image_path, max_size=512):
    """Preprocess image to reduce memory usage"""
    try:
        img = Image.open(image_path)
        # Resize large images to reduce memory consumption
        if max(img.size) > max_size:
            # Calculate new dimensions while maintaining aspect ratio
            ratio = max_size / max(img.size)
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            img = img.resize(new_size, Image.LANCZOS)
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {str(e)}")
        return None

def generate_analysis(model, processor, prompt, image_path, image_index, processed_images, output_file):
    image_filename = os.path.basename(image_path)
    
    # Skip if already processed
    if image_filename in processed_images:
        print(f"Skipping already processed image: {image_filename}")
        return None
    
    try:
        # Preprocess image to reduce memory usage
        image = preprocess_image(image_path)
        if image is None:
            raise ValueError(f"Failed to preprocess image {image_path}")
        
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Aggressive memory cleanup before inference
        torch.cuda.empty_cache()
        gc.collect()
        
        # Process inputs with memory management
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
            inputs = processor(
                text=text,
                images=image_inputs,
                video=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(model.device)  # Use model's device instead of hardcoding "cuda"
            
            # Generate with careful memory settings
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,  # Reduced from 250 to save memory
                do_sample=False,     # Deterministic generation uses less memory
                use_cache=True       # Enable KV cache for efficiency
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
        
        # Clean up GPU memory immediately after inference
        del inputs, outputs, generated_ids_trimmed
        torch.cuda.empty_cache()
        gc.collect()
        
        # Parse output with our custom parser
        parsed_output = parse_json_output(output_text)
        
        # Make sure image_id is set correctly
        parsed_output["image_id"] = f"image_{image_index}"
        
        # Ensure all keys exist with valid binary values
        standardized_output = {
            "Image": image_filename,
            "Block Work": int(parsed_output.get("Block Work", 0)),
            "Cement Plastering": int(parsed_output.get("Cement Plastering", 0)),
            "Windows & Ventilators Fixing": int(parsed_output.get("Windows & Ventilators Fixing", 0)),
            "Electrical Switches & Sockets Installation": int(parsed_output.get("Electrical Switches & Sockets Installation", 0)),
            "Flooring": int(parsed_output.get("Flooring", 0))
        }
        
        # Append to the CSV file (not overwriting)
        with open(output_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(standardized_output.values())
        
        # Add to processed images set
        processed_images.add(image_filename)
        
        return standardized_output
    
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        # Return default values in case of error
        default_output = {
            "Image": image_filename,
            "Block Work": 0,
            "Cement Plastering": 0,
            "Windows & Ventilators Fixing": 0,
            "Electrical Switches & Sockets Installation": 0,
            "Flooring": 0
        }
        
        # Still write the default output to CSV
        with open(output_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(default_output.values())
            
        return default_output

def main():
    # Path to the images
    image_dir = r"D:\Office\dataset_new\400"
    image_paths = [os.path.join(image_dir, f"{i}.jpg") for i in range(1, 400)]
    
    # Standardized prompt
    prompt = """Analyze the provided image and determine the construction progress for each stage. Use binary values: 1 if the stage is present, 0 otherwise.
    Stages:
    - Block Work{Yet to Start: Beam and slab are in progress, no brick work is done, and no brick work is visible.Work in Progress: Partial walls built; some parts show partial brick work.Completed: Full wall height achieved with no ongoing block laying.}
    - Cement Plastering{Yet to Start: Bare masonry with no plaster materials; no scaffolding/ladder for plaster.Work in Progress: Partial plaster coverage, wet or unfinished surfaces; scaffolding in use, and tools/materials actively used.Completed: Uniform plaster surface that is dry, with indications of the next stage.}
    - Windows & Ventilators Fixing{Yet to Start: No physical work has started; no window or ventilation materials are visible.Work in Progress: Frames are aligned and marked for proper placement; openings are cleaned and adjusted; frames fixed using screws, fasteners, or adhesives.Completed: All frames and panels are securely fixed, aligned, and sealed.}
    - Electrical Switches & Sockets Installation{Yet to Start: No switches, sockets, or lighting fixtures are installed; no electrical tools or equipment are in use.Work in Progress: Workers are visible installing switches, lights, or appliances; some fixtures are installed but not all areas are complete; testing devices may be visible.Completed: Fully installed and functional switches, sockets, and lighting fixtures with a clean, finished appearance.}
    - Flooring{Yet to Start: No even floor surface; visible cement layer or subfloor with debris.Work in Progress: Flooring work has started, though not all images may show completion.Completed: All flooring materials are securely installed with a clear pattern (e.g., tile, granite/marble, or wooden flooring).}

    Provide the output in JSON format like this:
    {
      "image_id": "image_1",
      "Block Work": 0 | 1,
      "Cement Plastering": 0 | 1,
      "Windows & Ventilators Fixing": 0 | 1,
      "Electrical Switches & Sockets Installation": 0 | 1,
      "Flooring": 0 | 1
    }
    """

    # Check if the output file exists, if not create it with headers
    output_file = "output1.csv"
    file_exists = os.path.isfile(output_file)

    # Load existing data to avoid processing duplicates
    processed_images = set()
    if file_exists:
        try:
            existing_data = pd.read_csv(output_file)
            processed_images = set(existing_data['Image'].tolist())
        except Exception as e:
            print(f"Error reading existing CSV file: {str(e)}")
            # If there's an error reading the file, we'll create a new one
            file_exists = False

    # Create or open the CSV file with proper mode
    if not file_exists:
        with open(output_file, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Image", "Block Work", "Cement Plastering", "Windows & Ventilators Fixing",
                           "Electrical Switches & Sockets Installation", "Flooring"])
    
    # Determine where to start processing
    start_index = 0  # Default start index
    if processed_images:
        # Find the highest numbered image already processed
        last_processed = 0
        for img in processed_images:
            try:
                # Extract the number from filenames like "1.jpg"
                num = int(img.split('.')[0])
                last_processed = max(last_processed, num)
            except (ValueError, IndexError):
                continue
        
        start_index = last_processed
        print(f"Resuming from image index {start_index + 1}")
    
    # Load model and processor just once
    model, processor = load_model_and_processor()
    
    # Process images starting from where we left off
    batch_size = 5  # Reduced batch size to prevent OOM errors
    
    try:
        for batch_start in range(start_index, len(image_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(image_paths))
            print(f"Processing batch: images {batch_start+1} to {batch_end}")
            
            for i in range(batch_start, batch_end):
                image_index = i + 1  # 1-based indexing for image_id
                result = generate_analysis(model, processor, prompt, image_paths[i], image_index, processed_images, output_file)
                if result:
                    print(f"Processed image {i+1}/{len(image_paths)}: {image_paths[i]}")
                    print(f"Result: {result}")
                
                # Clear memory after each image
                torch.cuda.empty_cache()
                gc.collect()
            
            print(f"Completed batch {batch_start+1} to {batch_end}")
            # Extra cleanup between batches
            torch.cuda.empty_cache()
            gc.collect()
    
    finally:
        # Clean up resources
        del model, processor
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f"Processing complete. Data saved to {output_file}")

if __name__ == "__main__":
    main()