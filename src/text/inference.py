import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

# Load trained model and tokenizer
model_path = "./saved_model"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = RobertaForSequenceClassification.from_pretrained(model_path)
model.eval()

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Function: classify text
def classify_essay(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze()
        confidence = float(torch.max(probs)) * 100
        predicted_class = torch.argmax(probs).item()
        
        label = "Human" if predicted_class == 0 else "AI"
        return label, confidence

# Example usage
if __name__ == "__main__":
    print("🔍 Enter an essay to analyze. Paste full essay and press Enter twice:\n")
    essay = ""
    while True:
        line = input()
        if line == "":
            break
        essay += line + "\n"

    result, score = classify_essay(essay)
    print(f"\n🧠 Prediction: {result}")
    print(f"✅ Confidence: {score:.2f}%")
