import torch
from core.config import DEFAULT_TOP_K
from retrieval.embeddings import ClinicalEmbeddingsService
from retrieval.vector_store import ClinicalVectorStore

class ClinicalRetriever:

    def __init__(self,embedding_service=None,vector_store=None):
        #self.embedding_service=(embedding_service or ClinicalEmbeddingsService)
        if embedding_service:
            self.embedding_service=embedding_service
        else:
            self.embedding_service=ClinicalEmbeddingsService()

        self.vector_store=(vector_store or ClinicalVectorStore(embedding__service=self.embedding_service))  


    def retriever(self,query:str,top_k:int=DEFAULT_TOP_K)-> list[dict]:
        query_embedding=(self.embedding_service.get_embeddings(query))
        documents=(self.vector_store.get_documents())
        document_embeddings=(self.vector_store.get_embeddingsVS())
        scores=[]

        for embeddings in document_embeddings:
            similarity=torch.matmul(query_embedding,embeddings.T)
            scores.append(similarity.items())

        sorted_indices=sorted(range(len(scores)),key=lambda index:scores[index],reverse=True)
        top_indices=sorted_indices[:top_k]

        results=[]
        for index in top_indices:
            results.append({
                "document":documents[index],
                "similarity_score":round(scores[index],4)
            })
        return results