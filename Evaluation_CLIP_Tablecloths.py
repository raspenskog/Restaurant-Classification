import os
import json
import torch
import logging
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Paths
base_dir = r'path_to_dataset'
log_dir = os.path.join(base_dir, 'tablecloth_logs')
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log file name based on the current timestamp
log_filename = f"tablecloth_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file_path = os.path.join(log_dir, log_filename)

class_dirs = {
    'Class_1-2': os.path.join(base_dir, 'Class_1-2'),
    'Class_3': os.path.join(base_dir, 'Class_3')
}
annotations_file_path = os.path.join(base_dir, 'annotations.json')

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Prompt candidates for tuning
prompt_candidates = {
    "tablecloth": [
        "A dining table with a tablecloth.",
        "Dining tables with tablecloths.",
        "A vintage table covered with a tablecloth.",
        "Vintage tables covered with tablecloths.",
        "A table draped with a tablecloth.",
        "Tables draped with tablecloths.",
        "A table covered with a tablecloth.",
        "Tables covered with tablecloths.",
        "A dining room table covered with a tablecloth.",
        "Dining room tables covered with tablecloths.",
        "A table in a dining setting with a tablecloth.",
        "Tables in a dining setting with tablecloths.",
        "A table in a restaurant with a white tablecloth.",
        "Tables in a restaurant with white tablecloths.",
        "A table set for dinner with a tablecloth.",
        "Tables set for dinner with tablecloths.",
        "A table with cloth.",
        "Tables with cloth.",
        "A vintage table with a tablecloth.",
        "Vintage tables with tablecloths.",
        "A table set for a banquet with a tablecloth.",
        "Tables set for a banquet with tablecloths.",
        "A vintage table with a white tablecloth.",
        "Vintage tables with white tablecloths.",
        "A retro table with a tablecloth.",
        "Retro tables with tablecloths.",
        "A vintage table covered with a tablecloth.",
        "Vintage tables covered with tablecloths.",
        "A retro table covered with a tablecloth.",
        "Retro tables covered with tablecloths."
    ],
    "bare_table": [
        "A dining table without a tablecloth.",
        "Dining tables without tablecloths.",
        "A vintage table without a tablecloth.",
        "Vintage tables without tablecloths.",
        "A rustic table with no tablecloth.",
        "Rustic tables with no tablecloths.",
        "A table with no covering.",
        "Tables with no coverings.",
        "A dining table with no covering.",
        "Dining tables with no coverings.",
        "A dining table without a tablecloth.",
        "Dining tables without tablecloths.",
        "A wooden table with no tablecloth.",
        "Wooden tables with no tablecloths.",
        "A simple table with an uncovered surface.",
        "Simple tables with uncovered surfaces.",
        "A table without a tablecloth, ready to be set.",
        "Tables without tablecloths, ready to be set.",
        "A rustic table with no tablecloth.",
        "Rustic tables with no tablecloths.",
        "A plain dining table.",
        "Plain dining tables.",
        "A simple dining table.",
        "Simple dining tables.",
        "A tavern table.",
        "Tavern tables.",
        "A table in a simple tavern.",
        "Tables in simple taverns.",
        "A modest table with no tablecloth.",
        "Modest tables with no tablecloths."
    ]
}

# Log the prompt candidates being used
logging.info("Starting prompt tuning with the following candidates:")
logging.info(f"Tablecloth prompts: {prompt_candidates['tablecloth']}")
logging.info(f"Bare table prompts: {prompt_candidates['bare_table']}")

# Load the JSON data from the annotations file
with open(annotations_file_path, 'r') as file:
    data = json.load(file)

# Create a set of filenames labeled as "tablecloth"
tablecloth_files = {
    os.path.basename(item['data']['image'])
    for item in data
    if any(
        "Tablecloth" in result['value']['choices']
        for annotation in item.get('annotations', [])
        for result in annotation.get('result', [])
    )
}

# Load the CLIP model and processor
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to process a single image with given prompts
def process_image_with_prompts(image_path, prompt_tablecloth, prompt_bare_table):
    try:
        image = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, FileNotFoundError):
        return None  # Skip if image cannot be opened

    image_inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs_tablecloth = model(**processor(text=[prompt_tablecloth], images=image, return_tensors="pt", padding=True).to(device))
        logits_tablecloth = outputs_tablecloth.logits_per_image.item()

        outputs_bare_table = model(**processor(text=[prompt_bare_table], images=image, return_tensors="pt", padding=True).to(device))
        logits_bare_table = outputs_bare_table.logits_per_image.item()

    return logits_tablecloth > logits_bare_table

# Initialize best result trackers
best_prompts = None
best_accuracy = 0

# Total combinations of prompts
total_combinations = len(prompt_candidates["tablecloth"]) * len(prompt_candidates["bare_table"])

# Prompt tuning loop with overall progress bar
with tqdm(total=total_combinations, desc="Overall Progress", ncols=100) as overall_progress:
    for prompt_tablecloth in prompt_candidates["tablecloth"]:
        for prompt_bare_table in prompt_candidates["bare_table"]:
            # Initialize counters for the current pair of prompts
            results = {
                'total_images': 0,
                'correct_predictions': 0,
                'incorrect_predictions': 0,
                'per_class': {
                    'Class_1-2': {'total': 0, 'correct': 0, 'incorrect': 0},
                    'Class_3': {'total': 0, 'correct': 0, 'incorrect': 0}
                },
                'incorrect_details': []
            }

            # Process each image with the current prompts
            for class_name, class_dir in class_dirs.items():
                image_files = [
                    os.path.join(class_dir, fname)
                    for fname in os.listdir(class_dir)
                    if os.path.isfile(os.path.join(class_dir, fname))
                ]

                for image_file in tqdm(image_files, desc=f"Processing {class_name} | {prompt_tablecloth} | {prompt_bare_table}", unit="image", ncols=80, leave=False):
                    predicted_contains_tablecloth = process_image_with_prompts(image_file, prompt_tablecloth, prompt_bare_table)
                    actual_contains_tablecloth = os.path.basename(image_file) in tablecloth_files
                    is_correct = predicted_contains_tablecloth == actual_contains_tablecloth

                    # Update counters
                    results['total_images'] += 1
                    if is_correct:
                        results['correct_predictions'] += 1
                        results['per_class'][class_name]['correct'] += 1
                    else:
                        results['incorrect_predictions'] += 1
                        results['per_class'][class_name]['incorrect'] += 1
                        results['incorrect_details'].append({
                            "image": image_file,
                            "predicted_contains_tablecloth": predicted_contains_tablecloth,
                            "actual_contains_tablecloth": actual_contains_tablecloth
                        })
                    results['per_class'][class_name]['total'] += 1

            # Log incorrect predictions in detail
            if results['incorrect_predictions'] > 0:
                logging.info(f"Incorrect Predictions for Prompts - Tablecloth: {prompt_tablecloth} | Bare Table: {prompt_bare_table}")
                for detail in results['incorrect_details']:
                    logging.info(f"Image: {detail['image']}, Predicted: {detail['predicted_contains_tablecloth']}, Actual: {detail['actual_contains_tablecloth']}")

            # Calculate accuracy for the current prompt pair
            accuracy = results['correct_predictions'] / results['total_images']
            logging.info(f"Tested Prompts - Tablecloth: {prompt_tablecloth} | Bare Table: {prompt_bare_table} | Accuracy: {accuracy:.4f}")

            # Update best prompts if current accuracy is better
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_prompts = (prompt_tablecloth, prompt_bare_table)

            # Update overall progress bar
            overall_progress.update(1)

# Log the best prompts and their accuracy
logging.info(f"Best Prompts - Tablecloth: {best_prompts[0]} | Bare Table: {best_prompts[1]} | Accuracy: {best_accuracy:.4f}")
