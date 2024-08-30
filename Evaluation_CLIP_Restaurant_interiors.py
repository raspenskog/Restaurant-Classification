# Evaluating the performance of CLIP in the task of finding images depicting restaurant interiors.

import os
import time
from PIL import Image, ImageFile
import torch
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
import numpy as np

# Suppress future warnings and enable truncated images
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize paths and model details
source_dir = "path_to_images"
model_name = "openai/clip-vit-large-patch14"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model and processor initialization
model = CLIPModel.from_pretrained(model_name).to(device)
processor = CLIPProcessor.from_pretrained(model_name, do_rescale=False)

# Image directories
restaurant_dir = os.path.join(source_dir, "restaurant_interiors")
non_restaurant_dirs = [
    os.path.join(source_dir, "non_restaurant_interiors"), # Path to subdirectory of non-restaurant interior images
    os.path.join(source_dir, "exteriors") # Path to subdirectory of exterior images
]

# Collecting image paths
restaurant_files = [os.path.join(restaurant_dir, file) for file in os.listdir(restaurant_dir) if file.lower().endswith('.png')]
non_restaurant_files = [os.path.join(d, file) for d in non_restaurant_dirs for file in os.listdir(d) if file.lower().endswith('.png')]

# Combining and labeling images
image_files = [(path, 1) for path in restaurant_files] + [(path, 0) for path in non_restaurant_files]
np.random.shuffle(image_files)

# Define data augmentation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# Specify a single prompt
prompt = "A vintage black and white photo of a historical 20th century dining room interior. The photograph features time-appropriate vintage furniture and decor that reflects the atmosphere of 20th century eating and drinking culture."

# Specify the logit threshold to test
logit_threshold = 20  # Set your desired threshold here

# Function to process a single image
def process_image(image_path, label, logit_threshold, text_description):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0)  # Apply data augmentation and add batch dimension
        inputs = processor(text=[text_description], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logit_value = outputs.logits_per_image.squeeze().item()
        return logit_value >= logit_threshold, label
    except Exception as e:
        return False, label

# Initialize counters for metrics
true_positives = 0
false_positives = 0
false_negatives = 0
true_negatives = 0

# For logging
true_positive_list = []
false_positive_list = []
false_negative_list = []
true_negative_list = []

# Start timer
start_time = time.time()

# Evaluate the entire dataset
print(f"Evaluating prompt: {prompt}")
for path, true_label in image_files:
    prediction, predicted_label = process_image(path, true_label, logit_threshold, prompt)
    if prediction and true_label == 1:
        true_positives += 1
        true_positive_list.append(path)
    elif prediction and true_label == 0:
        false_positives += 1
        false_positive_list.append(path)
    elif not prediction and true_label == 1:
        false_negatives += 1
        false_negative_list.append(path)
    elif not prediction and true_label == 0:
        true_negatives += 1
        true_negative_list.append(path)

# End timer
end_time = time.time()
elapsed_time = end_time - start_time
minutes, seconds = divmod(int(elapsed_time), 60)

# Total images processed
total_images = true_positives + false_positives + false_negatives + true_negatives
processed_images = total_images

# Calculate percentages
true_positive_percentage = (true_positives / total_images) * 100
false_positive_percentage = (false_positives / total_images) * 100
false_negative_percentage = (false_negatives / total_images) * 100
true_negative_percentage = (true_negatives / total_images) * 100
accuracy = ((true_positives + true_negatives) / total_images) * 100

# Summary
summary = (
    f"Finished processing {total_images} images in {int(minutes)} minutes and {int(seconds)} seconds.\n"
    f"Text description: '{prompt}'\n"
    f"Processed {processed_images} images.\n"
    f"Logit threshold: {logit_threshold}\n"
    f"True Positives: {true_positives} ({true_positive_percentage:.2f}%)\n"
    f"False Positives: {false_positives} ({false_positive_percentage:.2f}%)\n"
    f"False Negatives: {false_negatives} ({false_negative_percentage:.2f}%)\n"
    f"True Negatives: {true_negatives} ({true_negative_percentage:.2f}%)\n"
    f"Accuracy: {accuracy:.2f}%\n"
)

print(summary)

# Write the summary and lists to the log file
log_file_name = "evaluation_log.txt"  # Set the log file name
log_path = os.path.join(source_dir, log_file_name)
with open(log_path, 'w') as log_file:
    log_file.write(summary + '\n')
    log_file.write("True Positives:\n" + "\n".join(true_positive_list) + '\n')
    log_file.write("False Positives:\n" + "\n".join(false_positive_list) + '\n')
    log_file.write("False Negatives:\n" + "\n".join(false_negative_list) + '\n')
    log_file.write("True Negatives:\n" + "\n".join(true_negative_list) + '\n')
