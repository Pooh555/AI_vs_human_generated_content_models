# AI vs. Human Text Detector

A two-model ensemble for detecting AI-generated essays.  
Uses DeBERTa v3 + a GPT-2 perplexity (PPL) feature to achieve ≳ 80 % accuracy.

---

## 📋 Contents

- `data/`  
  - `AI_Human.csv` (downloaded raw 500 K essays)  
  - `AI_Human_with_ppl.csv` (post-PPL annotations)  
- `compute_ppl.py`  
- `train.py`  
- `ensemble_inference.py`  
- `requirements.txt`  
- `README.md`  

---

Download the raw 500 K essay CSV from Kaggle and place it here:

data/AI_Human.csv

Compute GPT-2 Perplexities
Adds a new column ppl_gpt2 to drive our PPL-fusion head:

python compute_ppl.py \
  --input data/AI_Human.csv \
  --output data/AI_Human_with_ppl.csv
