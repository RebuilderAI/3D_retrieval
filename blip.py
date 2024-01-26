from PIL import Image
import requests
from transformers import AutoProcessor, Blip2ForConditionalGeneration, BitsAndBytesConfig
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
        )
processor = AutoProcessor.from_pretrained('Salesforce/blip2-opt-2.7b', cache_dir='/data/ansh941/.cache', load_in_4bit=True, quantization_config=bnb_config)

model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto", load_in_4bit=True, cache_dir='/data/ansh941/.cache', quantization_config=bnb_config) # load in int4

class_prompt = 'Question: what does it look like? Answer:'

def create_caption(image):
    global processor, model, class_prompt

    inputs = processor(image, text=class_prompt, return_tensors='pt').to(device, torch.float16)
    
    generated_ids = model.generate(**inputs, max_new_tokens=30)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    return generated_text