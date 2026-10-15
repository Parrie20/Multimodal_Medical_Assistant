import os

from pipeline.clinical_pipeline import ClinicalPipeline
from pipeline.image_pipeline import ImagePipeline
from dataclasses import asdict
from services.multimodal_service import generate_medical_response


print("CLI Clinical RAG system ready.")

clinical_pipeline=ClinicalPipeline()
image_pipeline = ImagePipeline()

# CREATE FILE OBJECT FOR CLI IMAGE
class CLIImageFile:

    def __init__(self, image_path):

        self.filename = os.path.basename(image_path)

        self.file = open(image_path, "rb")


# CLINICAL TEXT PROCESSING
def process_clinical_text(text):

    print("\nProcessing clinical information...")

    clinical_result,evidence_result=clinical_pipeline.process(text,top_k=2)

    print("\nRetrieved Clinical Knowledge:\n")

    for result in evidence_result.items:

        print(
            f"Similarity Score: "
            f"{result['similarity_score']}"
        )

        print(result["document"])

        print("\n-------------------------")

    print("\nGenerating AI response...\n")

    ai_response = generate_medical_response(
        text=text,
        clinical_results=clinical_result,
        image_results=None
    )

    print("========================================")
    print("AI ANALYSIS")
    print("========================================\n")

    print(ai_response)

    print("\n========================================\n")


# IMAGE PROCESSING

def process_image(image_path):

    image_path = os.path.expanduser(
        image_path.strip()
    )

    if not os.path.isfile(image_path):
        print("\nError: Image file not found.")
        print(f"Path received: {repr(image_path)}\n")
        return

    print("\nProcessing medical image...")

    image_file = CLIImageFile(image_path)

    try:
        # Run the new image pipeline.
        image_result = image_pipeline.process(image_file)

        # Convert the dataclass into a dictionary for
        # compatibility with the existing response service.
        image_results = asdict(image_result)

    finally:
        image_file.file.close()

    print("\nImage Analysis Results:\n")
    print(image_results)

    print("\nGenerating AI response...\n")

    ai_response = generate_medical_response(
        text=None,
        clinical_results=None,
        image_results=image_results
    )

    print("========================================")
    print("AI ANALYSIS")
    print("========================================\n")

    print(ai_response)

    print("\n========================================\n")


# MULTIMODAL PROCESSING
def process_multimodal(text, image_path):

    clinical_result = None
    evidence_result = None
    image_results = None

    # -----------------------------------
    # CLINICAL RAG
    # -----------------------------------

    if text:

        print("\nProcessing clinical information...")

        clinical_result, evidence_result = (
            clinical_pipeline.process(
                text,
                top_k=2
            )
        )

    # -----------------------------------
    # IMAGE ANALYSIS
    # -----------------------------------

    image_results = None

    if image_path:
        image_path = os.path.expanduser(
            image_path.strip()
        )

        if not os.path.isfile(image_path):
            print("\nError: Image file not found.")
            print(f"Received path: {repr(image_path)}\n")
            return

        print("\nProcessing medical image...")

        image_file = CLIImageFile(image_path)

        try:
            image_result = image_pipeline.process(image_file)
            image_results = asdict(image_result)

        finally:
            image_file.file.close()


    # -----------------------------------
    # GENERATE FINAL RESPONSE
    # -----------------------------------

    print(
        "\nGenerating multimodal AI response...\n"
    )

    ai_response = generate_medical_response(
        text=text,
        clinical_results=(
            evidence_result.items
            if evidence_result
            else None
        ),
        image_results=image_results
    )

    # -----------------------------------
    # PRINT RESULTS
    # -----------------------------------

    print("========================================")
    print("MULTIMODAL AI ANALYSIS")
    print("========================================\n")

    if evidence_result:

        print(
            "RETRIEVED CLINICAL KNOWLEDGE:\n"
        )

        for result in evidence_result.items:

            print(
                f"Similarity Score: "
                f"{result.similarity_score}"
            )

            print(result.document)

            print(
                "\n-------------------------\n"
            )

    if image_results:

        print("IMAGE ANALYSIS:\n")

        print(image_results)

        print(
            "\n-------------------------\n"
        )

    print("AI RESPONSE:\n")

    print(ai_response)

    print(
        "\n========================================\n"
    )