import torch

from model.biomedclip import (
    biomedclip_preprocess,
    biomedclip_model,
    biomedclip_tokenizer
)

from utils.medical_image_util import load_medical_image


def analyze_medical_image(image_file):

    # LOAD MEDICAL IMAGE
    img = load_medical_image(image_file)

    image_tensor = biomedclip_preprocess(
        img
    ).unsqueeze(0)

    # LABELS FOR BIOMEDCLIP
    labels = [
        "a normal medical image",
        "an abnormal medical image"
    ]

    text_tokens = biomedclip_tokenizer(labels)


    # BIOMEDCLIP INFERENCE
    with torch.no_grad():

        image_features = biomedclip_model.encode_image(
            image_tensor
        )

        text_features = biomedclip_model.encode_text(
            text_tokens
        )

        # Normalize image features
        image_features = image_features / image_features.norm(
            dim=-1,
            keepdim=True
        )

        # Normalize text features
        text_features = text_features / text_features.norm(
            dim=-1,
            keepdim=True
        )

        # Calculate similarity
        similarity = image_features @ text_features.T

        probabilities = torch.softmax(
            similarity,
            dim=-1
        )

    # GET PREDICTION
    predicted_index = probabilities.argmax(
        dim=-1
    ).item()

    prediction = labels[predicted_index]

    # RETURN RESULTS
    return {
        "filename": image_file.filename,
        "prediction": prediction,
        "scores": {
            labels[0]: round(
                probabilities[0][0].item(),
                4
            ),
            labels[1]: round(
                probabilities[0][1].item(),
                4
            )
        }
    }