# MedClassify-AI

Fine-tuning DistilBERT to classify sentences in medical research abstracts — with a TF-IDF baseline for comparison.

**Model on HuggingFace Hub →** [SuhailKhan06/medclassify-ai](https://huggingface.co/SuhailKhan06/medclassify-ai)

---

## What it does

Given a sentence from a clinical abstract, the model predicts which structural role it plays:

| Label           | Example sentence                                                           |
| --------------- | -------------------------------------------------------------------------- |
| **BACKGROUND**  | "Cardiovascular disease is a leading cause of death worldwide."            |
| **OBJECTIVE**   | "The aim of this study was to evaluate the safety of drug X."              |
| **METHODS**     | "Patients were randomly assigned to two treatment groups."                 |
| **RESULTS**     | "The treatment significantly improved 30-day survival rates."              |
| **CONCLUSIONS** | "These findings suggest the intervention is effective and well-tolerated." |

This kind of sentence-level structural labeling is useful for automated abstract parsing, evidence extraction, and literature review pipelines.

---

## Dataset

**PubMed 200k RCT** — sentences from randomized controlled trial abstracts, each labeled with its structural role.

| Split      | Size    |
| ---------- | ------- |
| Train      | 176,642 |
| Validation | 29,672  |
| Test       | 29,578  |

Loaded directly from HuggingFace Datasets. Mean sentence length: 151 characters (median 138).

---

## Results

### Baseline — TF-IDF + Logistic Regression

Trained on 50k-feature unigram/bigram TF-IDF vectors with L2 logistic regression.

|             | Test accuracy | Weighted F1 |
| ----------- | ------------- | ----------- |
| TF-IDF + LR | 77.55%        | 77.10%      |

Per-class breakdown on the test set:

| Class       | Precision | Recall | F1   | Support |
| ----------- | --------- | ------ | ---- | ------- |
| background  | 0.57      | 0.54   | 0.56 | 3,077   |
| conclusions | 0.69      | 0.68   | 0.69 | 4,571   |
| methods     | 0.83      | 0.90   | 0.86 | 9,884   |
| objective   | 0.66      | 0.48   | 0.55 | 2,333   |
| results     | 0.84      | 0.84   | 0.84 | 9,713   |

`BACKGROUND` and `OBJECTIVE` are the hardest classes — they're often short and structurally similar to each other.

### DistilBERT fine-tuned

Fine-tuned `distilbert-base-uncased` with the HuggingFace `Trainer` API. Max token length: 128. Training was interrupted before full convergence — checkpoint saved and pushed to the Hub. Full evaluation metrics will be updated once training completes on a GPU.

---

## How to use the model

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="SuhailKhan06/medclassify-ai"
)

result = classifier("Patients were randomly assigned to two treatment groups.")
print(result)
# [{'label': 'METHODS', 'score': 0.94}]
```

Or manually:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("SuhailKhan06/medclassify-ai")
model = AutoModelForSequenceClassification.from_pretrained("SuhailKhan06/medclassify-ai")

text = "The aim of this study was to evaluate the safety of drug X."
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class = logits.argmax(-1).item()
print(model.config.id2label[predicted_class])
# OBJECTIVE
```

---

## Running locally

```bash
git clone https://github.com/Suhail-Khan-06/medclassify-ai
cd medclassify-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook bert_training.ipynb
```

---

## Project structure

```
medclassify-ai/
├── bert_training.ipynb   # full training notebook: EDA, baseline, DistilBERT, eval
├── models/
│   ├── logistic_regression_baseline.pkl
│   └── tfidf_vectorizer.pkl
└── requirements.txt
```

---

## Why DistilBERT over full BERT?

DistilBERT is 40% smaller and 60% faster than BERT-base while retaining about 97% of its performance on GLUE benchmarks. For this task — short, single-sentence inputs averaging 138 characters — the full BERT model is unnecessary overhead. DistilBERT's 6-layer architecture is well-suited to the sentence length distribution here (75th percentile is 190 characters, well within the 128 token limit).

---

Built by [Mohammed Suhail Ahmed Khan](https://github.com/Suhail-Khan-06)
