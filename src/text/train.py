import warnings
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", message=".*TypedStorage is deprecated.*", category=UserWarning)

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# ─────────────────────────────────────────────────────────────────────────────
class PPLStreamingDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_mask, labels, ppls):
        self.input_ids     = input_ids
        self.attention_mask= attention_mask
        self.labels        = labels
        self.ppls          = ppls

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels":         self.labels[idx],
            "ppl":            self.ppls[idx],
        }

# ─────────────────────────────────────────────────────────────────────────────
class PPLFusionModel(nn.Module):
    def __init__(self, base_model: AutoModel):
        super().__init__()
        self.base = base_model
        hidden_size = self.base.config.hidden_size
        self.classifier = nn.Linear(hidden_size + 1, 2)

    def forward(self, input_ids, attention_mask, labels=None, ppl=None):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        cls_emb = outputs.last_hidden_state[:, 0]
        fused   = torch.cat([cls_emb, ppl.unsqueeze(1)], dim=1)
        logits  = self.classifier(fused)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        self.base.save_pretrained(save_directory)
        torch.save(
            self.classifier.state_dict(),
            os.path.join(save_directory, "classifier_head.pth")
        )

# ─────────────────────────────────────────────────────────────────────────────
def main():
    # a) Load & normalize
    df = pd.read_csv("./data/AI_Human_with_ppl.csv")
    if "generated" not in df.columns:
        for c in ("label","class"):
            if c in df.columns:
                df = df.rename(columns={c: "generated"}); break
    if df["generated"].dtype == object:
        df["generated"] = df["generated"].map({"Human":0,"human":0,"AI":1,"ai":1})

    # b) Filter & dedupe
    df = (
        df[df["text"].str.split().str.len() >= 150]
          .drop_duplicates("text")
          .reset_index(drop=True)
    )

    # c) Balance
    ai    = df[df.generated==1]
    hum   = df[df.generated==0]
    n     = min(len(ai), len(hum))
    df = pd.concat([
            ai.sample(n=n, random_state=42),
            hum.sample(n=n, random_state=42)
        ]).sample(frac=1, random_state=42).reset_index(drop=True)

    # d) Split
    texts, labs, ppls = df["text"].tolist(), df["generated"].tolist(), df["ppl_gpt2"].tolist()
    tr_txt, vl_txt, tr_lab, vl_lab, tr_ppl, vl_ppl = train_test_split(
        texts, labs, ppls, test_size=0.2, random_state=42
    )

    # e) Pre-tokenize (256 tokens)
    MAX_LEN = 256
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
    tr_enc = tokenizer(tr_txt, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    vl_enc = tokenizer(vl_txt, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")

    # f) Datasets
    train_ds = PPLStreamingDataset(
        tr_enc.input_ids, tr_enc.attention_mask,
        torch.tensor(tr_lab, dtype=torch.long),
        torch.tensor(tr_ppl, dtype=torch.float),
    )
    val_ds = PPLStreamingDataset(
        vl_enc.input_ids, vl_enc.attention_mask,
        torch.tensor(vl_lab, dtype=torch.long),
        torch.tensor(vl_ppl, dtype=torch.float),
    )

    # g) Model + checkpointing
    base = AutoModel.from_pretrained("microsoft/deberta-v3-large")
    base.gradient_checkpointing_enable()   # ← crucial for memory
    model = PPLFusionModel(base).to(device)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds)}

    collator = DataCollatorWithPadding(tokenizer)

    # h) TrainingArguments: B=2, accum=4, fp16, eval/save per epoch
    training_args = TrainingArguments(
        output_dir="./saved_model_deberta_ppl",
        seed=42,
        num_train_epochs=4,

        per_device_train_batch_size=1,       # ↓↓
        gradient_accumulation_steps=4,       # ↕
        per_device_eval_batch_size=8,

        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=200,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",

        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_steps=1000,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,

        fp16=True,
        dataloader_num_workers=4,
    )

    # i) Trainer → train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # j) Save
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("✅ Done — saved to", training_args.output_dir)

if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    main()
