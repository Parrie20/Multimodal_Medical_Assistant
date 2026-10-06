import torch
import torch.nn.functional as F
from model.clinical import (
    clinical_model,
    clinical_tokenizer,
    DEVICE
) 

class ClinicalEmbeddingsService:

    def __init__(self):
        self.model=clinical_model
        self.tokenizer=clinical_tokenizer
        self.device=DEVICE

    def get_embeddings(self,text: str):
        input=clinical_tokenizer(
            text,
            return_tensors="pt",
            truncation="True",
            padding="True",
            max_length=512
        )
        inputs={
            key: value.to(self.device)
            for key,value in inputs.items()
        }
        with torch.no_grad():
            outputs=self.model(**inputs)

        token_embeddings=outputs.last_hidden_state
        attention_mask=inputs["attention_mask"]

        mask_expanded=attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings=torch.sum(token_embeddings*mask_expanded,dim=1)
        sum_mask=torch.clamp(mask_expanded.sum(dim=1),min=1e-9)

        embeddings=sum_embeddings/sum_mask
        embeddings=F.normalize(embeddings,p=2,dim=1)

        return embeddings