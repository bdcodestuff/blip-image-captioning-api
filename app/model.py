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
        logger.info(f"Loaded model: {model_name}")
        return model, processor
    except Exception as e:
        logger.exception(f"Error loading model: {model_name}")
        raise e

def generate_caption(model, processor, image, text=None):
    try:
        # Prepare the inputs for captioning
        if text is not None:
            # Conditional image captioning
            inputs = processor(images=image, text=text, return_tensors="pt").to(device=device, dtype=torch.float16)
        else:
            # Unconditional image captioning
            inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        # Generate the caption
        out = model.generate(**inputs)
        caption = processor.batch_decode(out, skip_special_tokens=True)[0].strip()
        logger.info(f"Generated caption: {caption}")

        return caption
    except Exception as e:
        logger.exception("Error during caption generation")
        raise e
