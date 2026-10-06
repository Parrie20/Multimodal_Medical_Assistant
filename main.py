from fastapi import FastAPI

from api.doc_upload import router as upload_router
from api.img_routes import router as image_router
from api.clinical_route import router as clinical_router
from api.llama_route import router as llama_router
from api.multimodal_route import router as multimodal_router
from services.cli_service import (
    process_clinical_text,
    process_image,
    process_multimodal
)


# -----------------------------------
# FASTAPI APPLICATION
# -----------------------------------

app = FastAPI(
    title="Multimodal Medical Assistant",
    version="0.1.0"
)


app.include_router(upload_router)
app.include_router(image_router)
app.include_router(clinical_router)
app.include_router(llama_router)
app.include_router(multimodal_router)


@app.get("/health")
def health_check():

    return {
        "status": "healthy"
    }

def get_multiline_input(prompt):

    print(prompt)

    print("Type END on a new line when finished.\n")

    lines = []

    while True:

        line = input()

        if line.strip().upper() == "END":
            break

        lines.append(line)

    return "\n".join(lines).strip()

# -----------------------------------
# TERMINAL CLI
# -----------------------------------

def run_cli():

    print("\n========================================")
    print("   MULTIMODAL MEDICAL AI ASSISTANT")
    print("========================================\n")

    while True:

        print("Choose an option:\n")

        print("1. Clinical Text Analysis")
        print("2. Medical Image Analysis")
        print("3. Text + Medical Image Analysis")
        print("4. Exit")

        choice = input("\nEnter your choice (1-4): ").strip()


        # -----------------------------------
        # OPTION 1 — CLINICAL TEXT
        # -----------------------------------

        if choice == "1":

            text = get_multiline_input("\nEnter clinical information:")

            if not text:
                print("\nNo clinical text provided.\n")
                continue

            process_clinical_text(text)


        # -----------------------------------
        # OPTION 2 — MEDICAL IMAGE
        # -----------------------------------

        elif choice == "2":

            image_path = input(
                "\nEnter the full image path:\n"
            ).strip()

            process_image(image_path)


        # -----------------------------------
        # OPTION 3 — TEXT + IMAGE
        # -----------------------------------

        elif choice == "3":

            text = get_multiline_input("\nEnter clinical information:")

            image_path = input(
                "\nEnter the full image path:\n"
            ).strip()

            process_multimodal(
                text,
                image_path
            )


        # -----------------------------------
        # EXIT
        # -----------------------------------

        elif choice == "4":

            print(
                "\nExiting Multimodal Medical Assistant..."
            )

            break


        else:

            print(
                "\nInvalid choice. Please select 1-4.\n"
            )


# -----------------------------------
# START CLI
# -----------------------------------

if __name__ == "__main__":

    run_cli()