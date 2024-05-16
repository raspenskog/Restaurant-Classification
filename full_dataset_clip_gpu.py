import os
from PIL import Image, ImageFile
import shutil
import torch
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
import time
import requests  # Import the requests library for sending HTTP requests

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Logit threshold for copying images
logit_threshold = 20

# Start the timer
start_time = time.time()

# Check CUDA availability and initialize CLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

source_dir = "C:/images_full"
question = "The interior of a restaurant with several tables"
formatted_question = question.replace(' ', '_').replace('?', '')
today_date = datetime.now().strftime('%Y-%m-%d')
target_dir = os.path.join(f'FULL_DATASET_{today_date}_{formatted_question}_CLIP_Large_Logit_{logit_threshold}_GPU')
os.makedirs(target_dir, exist_ok=True)

total_images = len([name for name in os.listdir(source_dir) if name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))])
processed_images = 0

for filename in os.listdir(source_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(source_dir, filename)
        try:
            image = Image.open(image_path)
            inputs = processor(text=[question], images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logit_value = outputs.logits_per_image.squeeze().item()
            
            processed_images += 1
            percent_complete = (processed_images / total_images) * 100
            
            if logit_value > logit_threshold:
                shutil.copy(image_path, os.path.join(target_dir, filename))
                print(f"Copied '{filename}' to {target_dir} with {logit_value:.2f} logit value. ({percent_complete:.2f}%)")
            else:
                print(f"Skipped '{filename}' with {logit_value:.2f} logit value. ({percent_complete:.2f}%)")
        except Exception as e:
            print(f"Failed to process {filename} due to an error: {e}. ({percent_complete:.2f}%)")

end_time = time.time()
total_time_seconds = end_time - start_time
minutes, seconds = divmod(total_time_seconds, 60)
summary = f"Finished iteration on {total_images} images after {int(minutes)} minutes {int(seconds)} seconds, with {processed_images} images processed."

print(summary)
with open(os.path.join(target_dir, 'log.txt'), 'w') as log_file:
    log_file.write(summary + '\n')

# Sending a webhook upon completion
webhook_url = "https://ha.raspenskog.se/api/webhook/python"
response = requests.post(webhook_url)
print(f"Webhook sent, status code: {response.status_code}")
