import os
import json
import time
from tqdm import tqdm
from datetime import datetime
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch

# Paths
base_dir = r'path_to_root_directory'
log_dir = os.path.join(base_dir, 'tablecloth_logs')
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log file name based on the current timestamp
log_filename = f"VQA_ViLT_Performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

# Question variations to be tested
question_variations = [
    "Are there dining tables with tablecloths?",
    "Is there a dining table with a tablecloth?",
    "Are there vintage tables covered with tablecloths?",
    "Is there a vintage table covered with a tablecloth?",
    "Are there tables set for a banquet with tablecloths?",
    "Is there a table set for a banquet with a tablecloth?",
    "Are there tables draped with tablecloths?",
    "Is there a table draped with a tablecloth?",
    "Are there tables covered with tablecloths?",
    "Is there a table covered with a tablecloth?",
    "Are there dining room tables covered with tablecloths?",
    "Is there a dining room table covered with a tablecloth?",
    "Are there tables in a dining setting with tablecloths?",
    "Is there a table in a dining setting with a tablecloth?",
    "Are there tables in a restaurant with white tablecloths?",
    "Is there a table in a restaurant with a white tablecloth?",
    "Are there tables set for dinner with tablecloths?",
    "Is there a table set for dinner with a tablecloth?",
    "Are there tables with cloth?",
    "Is there any cloth on the table?",
    "Are there vintage tables with tablecloths?",
    "Is there a vintage table with a tablecloth?",
    "Are there vintage tables with white tablecloths?",
    "Is there a vintage table with a white tablecloth?",
    "Are there retro tables with tablecloths?",
    "Is there a retro table with a tablecloth?",
    "Are there vintage tables covered with tablecloths?",
    "Is there a vintage table covered with a tablecloth?",
    "Are there retro tables covered with tablecloths?",
    "Is there a retro table covered with a tablecloth?"
]

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

# Test all questions and log performance
question_performance = []

for question in question_variations:
    total_images = 0
    correct_predictions = 0
    start_time = time.time()

    # Iterate over the images in the directories and check against annotations
    for class_name, class_dir in class_dirs.items():
        for image_filename in tqdm(os.listdir(class_dir), desc=f"Processing {class_name} with question '{question}'", unit="image"):
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

    # Calculate performance for this question
    accuracy = correct_predictions / total_images * 100 if total_images > 0 else 0
    eta = time.time() - start_time

    question_performance.append((question, accuracy, total_images, correct_predictions, eta))

# Find the best-performing question
best_question = max(question_performance, key=lambda x: x[1])

# Log wrong predictions for the best question
wrong_predictions = []

# Process images again for the best-performing question to find wrong predictions
for class_name, class_dir in class_dirs.items():
    for image_filename in tqdm(os.listdir(class_dir), desc=f"Processing {class_name} for wrong predictions", unit="image"):
        image_path = os.path.join(class_dir, image_filename)

        if image_filename in expected_answers:
            expected_answer = expected_answers[image_filename]
            model_answer = process_single_image(image_path, best_question[0])

            if model_answer is not None and model_answer.lower() != expected_answer.lower():
                wrong_predictions.append((image_filename, expected_answer, model_answer))

# Writing the log file
with open(log_file_path, 'w') as log_file:
    log_file.write(f"Performance Summary for Each Question:\n")
    for q, acc, total, correct, time_taken in question_performance:
        log_file.write(f"Question: {q}\n")
        log_file.write(f"Accuracy: {acc:.2f}% ({correct}/{total})\n")
        log_file.write(f"Time Taken: {time_taken:.2f} seconds\n\n")
    
    log_file.write(f"\nBest Question: {best_question[0]}\n")
    log_file.write(f"Accuracy: {best_question[1]:.2f}%\n")
    log_file.write(f"Total Images: {best_question[2]}\n")
    log_file.write(f"Correct Predictions: {best_question[3]}\n")
    log_file.write(f"Time Taken: {best_question[4]:.2f} seconds\n")

    log_file.write(f"\nIncorrect Predictions for Best-Performing Question:\n")
    for image_filename, expected_answer, model_answer in wrong_predictions:
        log_file.write(f"Image: {image_filename}\n")
        log_file.write(f"Expected: {expected_answer}\n")
        log_file.write(f"Model Answer: {model_answer}\n\n")

print(f"Log file saved to {log_file_path}")
print(f"Best-performing question: {best_question[0]} with accuracy {best_question[1]:.2f}%")
print(f"Incorrect predictions logged in {log_file_path}")
