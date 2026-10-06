import open_clip

# LOADING BIOMEDCLIP MODEL
print("Loading BiomedCLIP...")

biomedclip_model, _, biomedclip_preprocess = open_clip.create_model_and_transforms(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

biomedclip_tokenizer = open_clip.get_tokenizer(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

biomedclip_model.eval()

print("BiomedCLIP loaded successfully.")