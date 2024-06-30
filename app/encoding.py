"""
This module contains the image encoding logic
"""

import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    DetrImageProcessor,
    DetaForObjectDetection
)
from PIL import Image

def get_caption(image, blip_model_name, device_type):
    """"
    Return BLIP caption for the image
    """
    processor = BlipProcessor.from_pretrained(blip_model_name)
    model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(device_type)
    inputs = processor(images= image, return_tensors= 'pt').to(device_type)
    output = model.generate(**inputs, max_new_tokens= 20)
    caption = processor.decode(output[0], skip_special_tokens= True)
    return caption

def detect_objects(image, detr_model_name, device_type, threshold= 0.9):
    """
    Returns the BLIP detected objects in image
    """
    processor = DetrImageProcessor.from_pretrained(detr_model_name)
    model = DetaForObjectDetection.from_pretrained(detr_model_name)
    inputs = processor(images= image, return_tensors= 'pt')
    output = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(output, target_sizes=target_sizes)[0]
    
    detections = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detection = {
            "box": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
            "class_name": model.config.id2label[int(label)],
            "confidence_score": float(score)
        }
        detections.append(detection)
    return detections