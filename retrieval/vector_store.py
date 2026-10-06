from pathlib import Path
import torch
from retrieval.embeddings import ClinicalEmbeddingsService

class ClinicalVectorStore:

    def __init__(self,embedding__service=None):

        if embedding__service:
            self.embedding_service=embedding__service
        else:
            self.embedding_service=ClinicalEmbeddingsService()

        self.documents:list[str]=[]
        self.embeddings:list[torch.Tensor]=[]

    def load_documents(self,filepath:str | Path)->list[str]:
        filepath=Path(filepath)
        with open(filepath,"r",encoding="utf-8") as file:
            content=file.read()
        documents=content.split("---")
        documents=[document.strip() for document in documents if document.strip()]
        self.documents=documents
        return self.documents

    def build_embeddings(self)->list[torch.Tensor]:
        self.embeddings=[]
        for document in self.documents:
            embeddings=self.embedding_service.get_embeddings()
            self.embeddings.append(embeddings)

        return self.embeddings

    def get_documents(self)->list[str]:
        return self.documents

    def get_embeddingsVS(self)->list[torch.Tensor]:
        return self.embeddings
        