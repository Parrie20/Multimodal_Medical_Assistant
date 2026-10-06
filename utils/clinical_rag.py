import torch
import torch.nn.functional as F

from model.clinical import (
    clinical_model,
    clinical_tokenizer,
    DEVICE
)


# CREATE TEXT EMBEDDING
def get_embedding(text):

    inputs = clinical_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = clinical_model(**inputs)

    # ATTENTION-MASK-AWARE MEAN POOLING
    token_embeddings = outputs.last_hidden_state

    attention_mask = inputs["attention_mask"]

    mask_expanded = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    sum_embeddings = torch.sum(
        token_embeddings * mask_expanded,
        dim=1
    )

    sum_mask = torch.clamp(
        mask_expanded.sum(dim=1),
        min=1e-9
    )

    embeddings = sum_embeddings / sum_mask

    # Normalize embedding
    embeddings = F.normalize(
        embeddings,
        p=2,
        dim=1
    )

    return embeddings

# LOAD KNOWLEDGE BASE
def load_knowledge_base():

    with open(
        "knowledge_base/clinical_data.txt",
        "r"
    ) as file:

        content = file.read()

    # Split documents using ---
    documents = content.split("---")

    # Remove empty spaces
    documents = [
        document.strip()
        for document in documents
        if document.strip()
    ]

    return documents

# CREATE KNOWLEDGE BASE EMBEDDINGS
def create_knowledge_embeddings(documents):

    embeddings = []

    for document in documents:

        embedding = get_embedding(document)

        embeddings.append(embedding)

    return embeddings

# RETRIEVE MOST RELEVANT DOCUMENTS
def retrieve_relevant_documents(
    query,
    documents,
    document_embeddings,
    top_k=2
):

    query_embedding = get_embedding(query)

    scores = []

    for embedding in document_embeddings:

        similarity = torch.matmul(
            query_embedding,
            embedding.T
        )

        scores.append(similarity.item())

    # Get indices sorted by similarity
    sorted_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True
    )

    top_indices = sorted_indices[:top_k]

    results = []

    for index in top_indices:

        results.append({
            "document": documents[index],
            "similarity_score": round(
                scores[index],
                4
            )
        })

    return results