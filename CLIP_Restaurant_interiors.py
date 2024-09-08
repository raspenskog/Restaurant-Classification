import os
from PIL import Image, ImageFile
import torch
from transformers import CLIPProcessor, CLIPModel
from datetime import datetime
import time
import warnings
from torchvision import transforms
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress FutureWarning from huggingface_hub
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Logit threshold for copying images
logit_threshold = 19

# Input text description
text_description = "A vintage black and white photo of a historical 20th-century restaurant dining room interior. The photograph features time-appropriate vintage dining furniture and decor that reflects the atmosphere of 20th-century eating and drinking culture."

# Define source and destination directories
source_dir = "path_to_full_dataset"
destination_base_dir = "path_to_new_dataset"

# Create a unique subdirectory for storing copied images
date_str = datetime.now().strftime("%Y_%m_%d")
subfolder_name = f"{date_str}_{logit_threshold}"
destination_dir = os.path.join(destination_base_dir, subfolder_name)
os.makedirs(destination_dir, exist_ok=True)

# Copy the current script to the new directory
script_name = os.path.basename(__file__)
shutil.copy(__file__, os.path.join(destination_dir, script_name))

# Start the timer
start_time = time.time()

# Check CUDA availability and initialize CLIP model and processor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Collecting image paths
image_files = [os.path.join(source_dir, filename) for filename in os.listdir(source_dir) if filename.lower().endswith(('.png', '.jpg', '.jpeg'))]

total_images = len(image_files)

# Define your preprocessing transformations
def preprocess_image(image):
    """ Apply necessary preprocessing for images. """
    image = image.convert('RGB')
    image = image.resize((224, 224))
    return image

def process_image(image_path):
    try:
        image = Image.open(image_path)
        image = preprocess_image(image)

        inputs = processor(text=[text_description], images=image, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        logit_value = outputs.logits_per_image.squeeze().item()

        if logit_value >= logit_threshold:
            filename = os.path.basename(image_path)
            destination_path = os.path.join(destination_dir, filename)
            shutil.copy(image_path, destination_path)
        return True
    except Exception as e:
        tqdm.write(f"Failed to process image: {os.path.basename(image_path)} due to error: {e}")
        return False

# Initialize the progress bar
with tqdm(total=total_images, desc="Processing Images", unit="image", ncols=100, leave=True) as pbar:
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_path) for image_path in image_files]
        
        for future in as_completed(futures):
            future.result()
            pbar.update(1)

end_time = time.time()
total_time_seconds = end_time - start_time
minutes, seconds = divmod(total_time_seconds, 60)

# Summary
summary = (
    f"Finished processing {total_images} images in {int(minutes)} minutes and {int(seconds)} seconds.\n"
    f"Images that passed the logit threshold of {logit_threshold} have been copied to {destination_dir}.\n"
)

print(summary)
