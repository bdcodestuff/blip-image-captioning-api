import logging
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)

def load_model(model_name):
    try:
        # Load the pre-trained BLIP model for image-to-text captioning
        processor = Blip2Processor.from_pretrained(model_name)
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
        model.to(device)
        logger.info(f"Loaded model: {model_name}")
        return model, processor
    except Exception as e:
        logger.exception(f"Error loading model: {model_name}")
        raise e

def generate_caption(model, processor, image):

    try:
        # Prepare the inputs for captioning
        text = "this is a picture of"
        inputs = processor(image, text=text, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"Generated caption: {generated_text}")

        return generated_text
    except Exception as e:
        logger.exception("Error during caption generation")
        raise e
