from fastapi import APIRouter, UploadFile, File
import torch
from utils.img_utils import load_image
from model.biomedclip import biomedclip_preprocess,biomedclip_model,biomedclip_tokenizer

router=APIRouter()

@router.post("/api/v1/image/analyze")
async def analyze_img(file: UploadFile=File(...)):
    file_path=f"uploads/{file.filename}"
    with open(file_path,"wb") as buffer:
        buffer.write(await file.read())

    image=load_image(file_path)
    image_tensor=biomedclip_preprocess(image).unsqueeze(0)
    medical_prompts = [
        "A normal medical image",
        "An abnormal medical image",
        "Evidence of pneumonia",
        "Evidence of lung disease"
    ]
    text_tokens=biomedclip_tokenizer(medical_prompts)
    with torch.no_grad():

        # Create image embedding
        image_features = biomedclip_model.encode_image(image_tensor)

        # Create text embeddings
        text_features = biomedclip_model.encode_text(text_tokens)
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


        # Convert scores into probabilities
        probabilities = torch.softmax(similarity, dim=-1)

    results = {}
    for prompt, probability in zip(medical_prompts, probabilities[0]):
        results[prompt] = round(float(probability) * 100, 2)


    return {
    "filename": file.filename,
    "results": results
}  