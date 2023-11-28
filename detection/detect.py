from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image
import requests

##  activate virtual env -> source .env/bin/activate


url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQVKGiOS-GoCuHLPeg5IuVxgM9yOEYvHx3ibDXmT_Tf0zzqQB0r"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")




# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
            f"Detected {model.config.id2label[label.item()]} with confidence "
            f"{round(score.item(), 3)} at location {box}"
    )

