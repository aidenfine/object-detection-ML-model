from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, UnidentifiedImageError
import requests
import argparse
from termcolor import colored, cprint

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Object detection using DETR model.")
parser.add_argument("-f", "--file_path", type=str, required=False, help="File path of the image")
parser.add_argument("-u", "--image_url", type=str, required=False, help="URL of the image to process")
args = parser.parse_args()

# Check if either file path or image URL is provided
if args.file_path:
    # Load image from file path
    try:
        image = Image.open(args.file_path).convert("RGB")
    except UnidentifiedImageError:
        cprint("Error: Image file cannot be identified.", "red")
        exit(1)
elif args.image_url:
    # Load image from URL
    try:
        image = Image.open(requests.get(args.image_url, stream=True).raw).convert("RGB")
    except UnidentifiedImageError:
        cprint("Error: Image URL cannot be read. Make sure the URL ends with .png, .jpeg, .jpg, etc...", "red")
        exit(1)
else:
    # If neither file path nor image URL is provided, print an error message
    cprint("Please use -f or -u to specify a file path or URL link", "red")
    exit(1)

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])

try:
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    # print("results:", results)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected {model.config.id2label[label.item()]}"
            f" -->{(round(score.item(), 3)) * 100}% correct at location {box}"
        )

except Exception as e:
    # Handle other exceptions during object detection
    cprint(f"Error during object detection: {e}", "red")
