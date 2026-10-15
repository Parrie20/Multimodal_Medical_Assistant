import torch
from core.schemas import ImageResult
from processing.medical_image import MedicalImageProcessor
from model.biomedclip import (
    biomedclip_preprocess,
    biomedclip_model,
    biomedclip_tokenizer
)

class ImagePipeline:

    def __init__(self,image_processor=None):
        self.image_processor=(image_processor or MedicalImageProcessor())

    def process(self,image_file):
        image=self.image_processor.load(image_file)
        image_tensor=(biomedclip_preprocess(image).unsqueeze(0))
        labels=["a normal medical image", "an abnormal medical image"]
        text_tokens=(biomedclip_tokenizer(labels))
        with torch.no_grad():
            image_features=(biomedclip_model.encode_image(image_tensor))
            text_features=(biomedclip_model.encode_text(text_tokens))
            image_features=(image_features/image_features.norm(dim=-1,keepdim=True))
            text_features=(text_features/text_features.norm(dim=-1,keepdim=True))
            similarity=[image_features @ text_features.T]
            probabilities=torch.softmax(similarity,dim=-1)
        predicted_index=[probabilities.argmax(dim=-1).item()]
        predictions=labels[predicted_index]
        return ImageResult(
            filename=image_file.filename,
            prediction=prediction,
            scores={
                labels[0]: round(
                    probabilities[0][0].item(),
                    4
                ),
                labels[1]: round(
                    probabilities[0][1].item(),
                    4
                )
            }
        )