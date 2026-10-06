from langchain_ollama import ChatOllama


# LOAD LLAMA MODEL
llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

# MULTIMODAL ORCHESTRATION
def generate_medical_response(
    text,
    clinical_results,
    image_results
):

    prompt = f"""
You are a medical AI assistant.

User clinical information:
{text if text else "No clinical text was provided."}

Relevant clinical knowledge retrieved from the knowledge base:
{clinical_results if clinical_results else "No relevant clinical knowledge was retrieved."}

Image analysis results:
{image_results if image_results else "No medical image was provided."}

Based only on the provided clinical information, retrieved knowledge,
and image analysis results, give a concise and helpful analysis.

Important:
- Do not claim certainty.
- Do not present retrieved information as a confirmed diagnosis.
- Clearly state that this is AI-generated information.
- Recommend consulting a qualified healthcare professional for diagnosis.
"""

    response = llm.invoke(prompt)

    return response.content