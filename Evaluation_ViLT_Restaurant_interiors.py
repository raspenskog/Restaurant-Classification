import os
import time
from tqdm import tqdm
from datetime import datetime
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch

# Paths
base_dir = r'path_to_ground_truth_data'
log_dir = os.path.join(base_dir, 'evaluation_logs')
os.makedirs(log_dir, exist_ok=True)

# Generate a unique log file name based on the current timestamp
log_filename = f"VQA_ViLT_{datetime.now().strftime('%Y%m%d_%H%M%S')}_best_question.log"
log_file_path = os.path.join(log_dir, log_filename)

# Initialize the processor and model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# Define the directories for evaluation
directories = {
    'restaurant_interiors': os.path.join(base_dir, 'restaurant_interiors'),
    'non_restaurant_interiors': os.path.join(base_dir, 'non_restaurant_interiors'),
    'non_restaurant_exteriors': os.path.join(base_dir, 'non_restaurant_exteriors')
}

# Define the question variations
question_variations = [
    "Is this a restaurant interior?",
    "Is this picture taken inside a restaurant?",
    "Does this image show the inside of a restaurant?",
    "Is this photo taken inside a restaurant?",
    "Is this an interior of a restaurant?",
    "Does this image depict a restaurant interior?",
    "Is this a picture of a restaurant's interior?",
    "Is this an image of a restaurant's interior?",
    "Is this photograph showing the interior of a restaurant?",
    "Is this a restaurant?",
    "Does this image depict a restaurant?",
    "Is this an image of a restaurant?",
    "Is this a restaurant scene?",
    "Does this picture show the inside of a restaurant?",
    "Is this photo showing a restaurant interior?"
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

# Prepare to log the results
best_accuracy = 0
best_question = ""
total_images = 0

overall_correct_predictions = 0

start_time = time.time()

# Iterate over each question variation
for question in question_variations:
    correct_predictions = {key: 0 for key in directories.keys()}
    incorrect_predictions = {key: 0 for key in directories.keys()}
    log_entries = {key: [] for key in directories.keys()}

    question_total_images = 0
    question_correct_predictions = 0

    # Iterate over the images in the directories and process them
    for category, dir_path in directories.items():
        for image_filename in tqdm(os.listdir(dir_path), desc=f"Processing {category} with question '{question}'", unit="image"):
            image_path = os.path.join(dir_path, image_filename)
            
            # Process the image and get the model's answer
            model_answer = process_single_image(image_path, question)
            
            if model_answer is not None:
                question_total_images += 1
                total_images += 1
                expected_answer = "yes" if category == "restaurant_interiors" else "no"
                
                if model_answer.lower() == expected_answer:
                    correct_predictions[category] += 1
                    overall_correct_predictions += 1
                    question_correct_predictions += 1
                else:
                    incorrect_predictions[category] += 1
                
                log_entries[category].append(f"Image: {image_filename} | Expected: {expected_answer} | Model: {model_answer}")

    # Calculate accuracy for this question
    question_accuracy = question_correct_predictions / question_total_images * 100 if question_total_images > 0 else 0

    # Check if this is the best-performing question
    if question_accuracy > best_accuracy:
        best_accuracy = question_accuracy
        best_question = question

    # Write log for this question
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"\nQuestion: {question}\n")
        log_file.write(f"Total images processed: {question_total_images}\n")
        log_file.write(f"Correct predictions: {question_correct_predictions}\n")
        log_file.write(f"Accuracy: {question_accuracy:.2f}%\n")

        for category in directories.keys():
            total_category_images = correct_predictions[category] + incorrect_predictions[category]
            category_accuracy = correct_predictions[category] / total_category_images * 100 if total_category_images > 0 else 0
            log_file.write(f"\nSummary for {category}:\n")
            log_file.write(f"  Total images: {total_category_images}\n")
            log_file.write(f"  Correct predictions: {correct_predictions[category]}\n")
            log_file.write(f"  Incorrect predictions: {incorrect_predictions[category]}\n")
            log_file.write(f"  Accuracy: {category_accuracy:.2f}%\n")

        for category in directories.keys():
            log_file.write(f"\nDetailed Results for {category}:\n")
            log_file.write("\n".join(log_entries[category]))
        log_file.write("\n" + "="*50 + "\n")

# Calculate overall accuracy
overall_accuracy = overall_correct_predictions / total_images * 100 if total_images > 0 else 0

# Final log entry with the best question
eta = time.time() - start_time
with open(log_file_path, 'a') as log_file:
    log_file.write(f"\nBest Question: {best_question}\n")
    log_file.write(f"Best Accuracy: {best_accuracy:.2f}%\n")
    log_file.write(f"Overall Accuracy: {overall_accuracy:.2f}%\n")
    log_file.write(f"Total time: {eta:.2f} seconds\n")

print(f"Log file saved to {log_file_path}")
