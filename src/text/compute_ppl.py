import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from math import exp
from tqdm import tqdm

def compute_perplexities(csv_in, csv_out, device="cuda"):
    # Load data
    df = pd.read_csv(csv_in)
    texts = df["text"].tolist()

    # Load GPT-2
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    model.eval()

    perps = []
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing PPL"):
            enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            # shift so labels==input_ids
            labels = enc["input_ids"]
            outputs = model(**enc, labels=labels)
            loss = outputs.loss.item()
            perps.append(exp(loss))

    # Attach and save
    df["ppl_gpt2"] = perps
    df.to_csv(csv_out, index=False)
    print(f"Saved with PPL feature: {csv_out}")

if __name__ == "__main__":
    compute_perplexities(
        csv_in="./data/AI_Human.csv",
        csv_out="./data/AI_Human_with_ppl.csv"
    )
