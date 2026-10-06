from fastapi import APIRouter, UploadFile, File, Form
from typing import Optional

from services.multimodal_service import generate_medical_response
from services.image_service import analyze_medical_image

from utils.clinical_rag import (
    load_knowledge_base,
    create_knowledge_embeddings,
    retrieve_relevant_documents
)


router = APIRouter()

# LOAD CLINICAL KNOWLEDGE BASE
print("Loading clinical knowledge base...")

documents = load_knowledge_base()

print("Creating knowledge base embeddings...")

document_embeddings = create_knowledge_embeddings(documents)

print("Clinical RAG system ready.")

# MULTIMODAL API
@router.post("/api/v1/multimodal/analyze")
async def multimodal_analyze(

    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)

):

    # CLINICAL RAG
    clinical_results = None

    if text:

        clinical_results = retrieve_relevant_documents(
            query=text,
            documents=documents,
            document_embeddings=document_embeddings,
            top_k=2
        )

    # IMAGE ANALYSIS
    image_results = None

    if image:

        image_results = analyze_medical_image(image)

    # GENERATE AI RESPONSE
    ai_response = generate_medical_response(
        text=text,
        clinical_results=clinical_results,
        image_results=image_results
    )

    # FINAL RESPONSE
    return {
        "clinical_text": text,
        "retrieved_clinical_knowledge": clinical_results,
        "image_analysis": image_results,
        "ai_response": ai_response
    }