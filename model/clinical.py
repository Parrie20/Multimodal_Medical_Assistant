import torch
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


print("Loading Bio_ClinicalBERT...")

clinical_tokenizer = AutoTokenizer.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT"
)

clinical_model = AutoModel.from_pretrained(
    "emilyalsentzer/Bio_ClinicalBERT"
)

clinical_model.to(DEVICE)

clinical_model.eval()

print("Bio_ClinicalBERT loaded successfully.")
print("Device:", DEVICE)