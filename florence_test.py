import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model and processor
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
model.to(device)  # Move model to GPU if available

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

prompt = "<OCR>"

# Load the image
picture = "axiom_screenshot.png"
image = Image.open(picture)

# Preprocess the inputs
inputs = processor(text=prompt, images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print("Viewing image...")
# Generate the output
generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=True,
    temperature=1,
    num_beams=10
)

print("Processing...")
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(generated_text)
