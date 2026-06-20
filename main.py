#%%
from transformers import AutoProcessor, AutoModelForImageTextToText, TextStreamer
from transformers.image_utils import load_image
from urllib.parse import urlparse
import os

# Function to load image from URL or local file
def load_image_from_source(source):
    if urlparse(source).scheme in ("http", "https"):  # Check if source is a URL
        return load_image(source)
    elif os.path.isfile(source):  # Check if source is a valid local file
        return load_image(source)
    else:
        raise ValueError("Invalid source. Provide a valid URL or local file path.")

# Load model and processor
model_id = "LiquidAI/LFM2.5-VL-1.6B"
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    device_map="auto",
    dtype="bfloat16"
)
processor = AutoProcessor.from_pretrained(model_id)
streamer = TextStreamer(processor, skip_prompt=True, skip_special_tokens=True)
#%% 
# Specify the image source (URL or local file path)
#image_source = "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg"
image_source="ticket_caisse.jpeg"

# Load the image
image = load_image_from_source(image_source)

#%%

# Prompts for testing
prompt="décris cette image?"
#prompt="what is in the image?"
#prompt="est ce qu'il y a de l'eau sur cette image?"
#prompt="Quel est le total TTC sur ce ticket de caisse?"
#prompt="Quel est le magasin de ce ticket de caisse?"


# Create conversation
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    },
]

# Generate Answer
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
    tokenize=True,
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=64, streamer=streamer)
#%%