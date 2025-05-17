# inference_ppl.py
import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    GPT2Tokenizer,
    GPT2LMHeadModel,
)

torch.cuda.empty_cache()

class PPLFusionModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        hidden = base_model.config.hidden_size
        self.classifier = nn.Linear(hidden + 1, base_model.config.num_labels)

    def forward(self, input_ids, attention_mask, ppl):
        # get the [CLS] token embedding from the base model
        out = self.base.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        cls_emb = out.last_hidden_state[:, 0]  # (batch_size, hidden)

        # ---- broadcast your single PPL scalar to (batch_size,1) ----
        if ppl.dim() == 0:
            # from torch.tensor(3.2)  →  shape []
            ppl = ppl.unsqueeze(0).repeat(input_ids.size(0))
        if ppl.dim() == 1:
            # from (batch_size,) → (batch_size,1)
            ppl = ppl.unsqueeze(1)

        fused = torch.cat([cls_emb, ppl.to(cls_emb.device)], dim=1)  # (batch_size, hidden+1)
        logits = self.classifier(fused)                              # (batch_size, num_labels)
        return logits

def compute_ppl(text, tok, model, device):
    # tokenize & get loss
    enc = tok(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    # perplexity == exp(loss)
    return torch.exp(loss).cpu()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo_dir = "/home/pooh555/coding/pyproj/ai-text-detector/saved_model_derbata_ppl"
    print(f"Loading fusion model from {repo_dir} on {device}…")

    # 1) load your fine-tuned DeBERTa sequence‐classifier (base_model)
    base = AutoModelForSequenceClassification.from_pretrained(repo_dir).to(device)

    # 2) wrap it in PPLFusionModel and load the fusion head
    model = PPLFusionModel(base).to(device)
    head_path = os.path.join(repo_dir, "classifier_head.pth")
    if os.path.isfile(head_path):
        sd = torch.load(head_path, map_location=device)
        model.classifier.load_state_dict(sd)
    model.eval()

    # 3) GPT-2 for PPL scoring
    ppl_tok   = GPT2Tokenizer.from_pretrained("gpt2")
    ppl_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    ppl_model.eval()

    print("✅ Ready—just paste an essay (empty to quit):")
    while True:
        txt = input("\nEssay> ").strip()
        if not txt:
            break

        # compute PPL scalar
        ppl = compute_ppl(txt, ppl_tok, ppl_model, device)

        # tokenize for your DeBERTa classifier
        cls_tok = AutoTokenizer.from_pretrained(repo_dir)
        enc     = cls_tok(
            txt,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        ).to(device)

        # forward through fusion model
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"], ppl)
        probs      = torch.softmax(logits, dim=-1)       # shape (1,2)
        conf, pred = probs.max(dim=-1)                   # both shape (1,)
        label      = "AI-generated" if pred.item()==1 else "Human-written"
        print(f"\n→ {label}   (conf={conf.item():.1%},  GPT-2 PPL={ppl:.1f})\n")

    print("👋 Goodbye!")

if __name__ == "__main__":
    main()
