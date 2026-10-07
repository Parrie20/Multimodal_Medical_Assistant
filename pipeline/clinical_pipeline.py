from pathlib import Path
from core.config import DEFAULT_TOP_K, KNOWLEDGAE_BASE_DIR
from core.schemas import ClinicalResult, EvidenceResult,EvidenceItem

from retrieval.embeddings import ClinicalEmbeddingsService
from Codebase.retrieval.Clinical_retriever import ClinicalRetriever
from retrieval.vector_store import ClinicalVectorStore

class ClinicalPipeline:

    def __init__(self,embedding_service=None,vector_store=None,retriever=None):
        self.embedding_service=(embedding_service or ClinicalEmbeddingsService())
        self.vector_store=(vector_store or ClinicalVectorStore(embedding_service=self.embedding_service))
        self.retriever=(retriever or ClinicalRetriever(embedding_service=self.embedding_service))
        self.initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        knowledge_file=(Path(KNOWLEDGAE_BASE_DIR)/"clinical_data.txt")
        self.vector_store.load_documents(knowledge_file)
        self.vector_store.build_embeddings()    

    def process(self,text: str,top_k: int = DEFAULT_TOP_K):

        clinical_embedding = (
            self.embedding_service.embed(text)
        )

        retrieved_documents = (
            self.retriever.retrieve(
                query=text,
                top_k=top_k
            )
        )

        evidence_items = [
            EvidenceItem(
                document=item["document"],
                similarity_score=item["similarity_score"]
            )
            for item in retrieved_documents
        ]

        return ClinicalResult(
            text=text,
            embedding=clinical_embedding,
            metadata={
                "top_k": top_k
            }
        ), EvidenceResult(
            query=text,
            items=evidence_items
        )