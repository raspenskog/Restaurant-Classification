import os
import json
import time
from tqdm import tqdm
from datetime import datetime
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch

# Paths
base_dir = r'path_do_dataset'
log_dir = os.path.join(base_dir, 'tablecloth_logs')
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log file name based on the current timestamp
log_filename = f"VQA_ViLT_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_file_path = os.path.join(log_dir, log_filename)

class_dirs = {
    'Class_1-2': os.path.join(base_dir, 'Class_1-2'),
    'Class_3': os.path.join(base_dir, 'Class_3')
}
annotations_file_path = os.path.join(base_dir, 'annotations.json')

# Initialize the processor and model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Load the cleaned JSON data from the annotations file
with open(annotations_file_path, 'r') as f:
    annotations = json.load(f)

# Define the question for the VQA model
question = "Is there any cloth on the table?"

# Create a dictionary to map filenames to expected answers based on annotations
expected_answers = {}

for item in annotations:
    image_filename = item['data']['image']
    expected_answer = "Yes" if any(
        "Tablecloth" in result['value']['choices']
        for annotation in item.get('annotations', [])
        for result in annotation.get('result', [])
    ) else "No"
    expected_answers[image_filename] = expected_answer

# Function to process a single image and question
def process_single_image(image_path, question):
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure image is in RGB format
        inputs = processor(images=image, text=question, return_tensors="pt")

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            answer_idx = logits.argmax(-1).item()

        # Convert the answer index to the actual answer text
        answer = model.config.id2label[answer_idx]
        return answer

    except Exception as e:
        print(f"Failed to process {image_path}: {e}")
        return None

# Prepare to log the results
total_images = 0
correct_predictions = 0
incorrect_predictions = 0
log_entries = {'Class_1-2': [], 'Class_3': [], 'Incorrect': []}

start_time = time.time()

# Iterate over the images in the directories and check against annotations
for class_name, class_dir in class_dirs.items():
    for image_filename in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}", unit="image"):
        image_path = os.path.join(class_dir, image_filename)
        
        # Check if the image is in the annotations
        if image_filename in expected_answers:
            expected_answer = expected_answers[image_filename]
            model_answer = process_single_image(image_path, question)
            
            if model_answer is not None:
                total_images += 1
                # Case-insensitive comparison
                if model_answer.lower() == expected_answer.lower():
                    correct_predictions += 1
                else:
                    incorrect_predictions += 1
                    # Log incorrect predictions in detail
                    log_entries['Incorrect'].append(f"Image: {image_filename} | Expected: {expected_answer} | Model: {model_answer}")

            # Append log entry to the correct class group
            log_entries[class_name].append(f"Image: {image_filename} | Expected: {expected_answer} | Model: {model_answer}")

# Calculate performance
accuracy = correct_predictions / total_images * 100 if total_images > 0 else 0
eta = time.time() - start_time

# Writing the log file
with open(log_file_path, 'w') as log_file:
    log_file.write(f"Question: {question}\n")
    log_file.write(f"Total images processed: {total_images}\n")
    log_file.write(f"Correct predictions: {correct_predictions} ({accuracy:.2f}%)\n")
    log_file.write(f"Incorrect predictions: {incorrect_predictions}\n")
    log_file.write(f"Total time: {eta:.2f} seconds\n")
    
    for class_name in ['Class_1-2', 'Class_3']:
        log_file.write(f"\nDetailed Results for {class_name}:\n")
        log_file.write("\n".join(log_entries[class_name]))

    log_file.write("\n\nIncorrect Predictions:\n")
    log_file.write("\n".join(log_entries['Incorrect']))

print(f"Log file saved to {log_file_path}")
