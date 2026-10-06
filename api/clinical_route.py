from fastapi import APIRouter
from pydantic import BaseModel
import torch

from model.clinical import clinical_tokenizer, clinical_model

router = APIRouter()


class ClinicalTextRequest(BaseModel):
    text: str


@router.post("/api/v1/clinical/analyze")
async def analyze_clinical_text(request: ClinicalTextRequest):

    tokens = clinical_tokenizer(
        request.text,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = clinical_model(**tokens)

    token_embeddings = outputs.last_hidden_state

    attention_mask = tokens["attention_mask"]

    mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    summed_embeddings = torch.sum(
        token_embeddings * mask_expanded,
        dim=1
    )

    summed_mask = torch.clamp(
        mask_expanded.sum(dim=1),
        min=1e-9
    )

    text_embedding = summed_embeddings / summed_mask

    return {
        "text": request.text,
        "embedding_shape": list(text_embedding.shape)
    }