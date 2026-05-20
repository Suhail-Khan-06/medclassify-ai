# MedClassify-AI

A medical abstract sentence classification project that fine-tunes DistilBERT to identify the structural role of sentences in clinical research abstracts, with a TF-IDF + Logistic Regression baseline for comparison.

**Hugging Face Model:** https://huggingface.co/SuhailKhan06/medclassify-ai

**Live Demo:** https://huggingface.co/spaces/SuhailKhan06/medclassify-ai-demo

---

## Overview

Medical research abstracts usually follow a structured format where each sentence serves a distinct purpose. Some sentences provide background information, others describe the study objective, explain methods, present results, or summarize conclusions.

MedClassify-AI automatically predicts the role of each sentence.

This project demonstrates both a traditional machine learning approach and a transformer-based deep learning approach for medical text classification.

---

## What the Model Predicts

The model classifies each sentence into one of the following labels:

- BACKGROUND
- OBJECTIVE
- METHODS
- RESULTS
- CONCLUSIONS

### Example

Input sentence:

"The aim of this study was to evaluate the safety of drug X."

Predicted label:

OBJECTIVE

---

## Use Cases

- Automated abstract structuring
- Biomedical literature review
- Evidence extraction
- Research summarization
- Clinical NLP pipelines
- Medical search systems

---

## Dataset

This project uses the PubMed 200k RCT dataset.

The dataset contains sentences from randomized controlled trial abstracts, each labeled with its structural role.

Dataset statistics:

- Training set: 176,642 sentences
- Validation set: 29,672 sentences
- Test set: 29,578 sentences
- Average sentence length: 151 characters
- Median sentence length: 138 characters

The dataset is loaded directly using Hugging Face Datasets.

---

## Models Implemented

### TF-IDF + Logistic Regression

A strong classical baseline built using:

- Unigram and bigram TF-IDF features
- 50,000 feature vocabulary
- L2-regularized Logistic Regression

### DistilBERT

A transformer model fine-tuned using:

- distilbert-base-uncased
- Hugging Face Trainer API
- PyTorch backend
- Maximum sequence length of 128 tokens

---

## Results

### TF-IDF + Logistic Regression Baseline

- Test Accuracy: 77.55%
- Weighted F1 Score: 77.10%

The baseline performs especially well on METHODS and RESULTS sentences.

### DistilBERT

DistilBERT was fine-tuned and uploaded to the Hugging Face Hub.

The model leverages contextual embeddings to understand sentence meaning and generally provides stronger performance than keyword-based approaches.

---

## Why DistilBERT?

DistilBERT is a smaller and faster version of BERT.

Advantages:

- Around 40% smaller than BERT-base
- Approximately 60% faster
- Retains about 97% of BERT's performance

Because the task involves short single-sentence inputs, DistilBERT offers an excellent trade-off between speed and accuracy.

---

## Try the Model

### Using the Pipeline API

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="SuhailKhan06/medclassify-ai"
)

result = classifier(
    "Patients were randomly assigned to two treatment groups."
)

print(result)
# [{'label': 'METHODS', 'score': 0.94}]
```

### Using AutoTokenizer and AutoModel

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("SuhailKhan06/medclassify-ai")
model = AutoModelForSequenceClassification.from_pretrained("SuhailKhan06/medclassify-ai")

text = "The aim of this study was to evaluate the safety of drug X."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=128
)

with torch.no_grad():
    logits = model(**inputs).logits

prediction = logits.argmax(-1).item()
print(model.config.id2label[prediction])
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Suhail-Khan-06/medclassify-ai.git
cd medclassify-ai
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

Linux and macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Training Notebook

```bash
jupyter notebook bert_training.ipynb
```

The notebook includes:

- Exploratory Data Analysis
- TF-IDF baseline
- DistilBERT fine-tuning
- Evaluation
- Model export to Hugging Face Hub

---

## Project Structure

```text
medclassify-ai/
├── bert_training.ipynb
├── models/
│   ├── logistic_regression_baseline.pkl
│   └── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

---

## Key Hyperparameters

- Base model: distilbert-base-uncased
- Maximum sequence length: 128
- Optimizer: AdamW
- Framework: Hugging Face Transformers

---

## Applications

- Scientific abstract parsing
- Automated evidence extraction
- Medical search engines
- Literature review assistants
- Research trend analysis

---

## Resume Description

Developed a medical abstract sentence classification system by fine-tuning DistilBERT on the PubMed 200k RCT dataset and compared performance against a TF-IDF + Logistic Regression baseline.

---

## Future Improvements

- Complete full GPU training
- Hyperparameter tuning
- Confusion matrix visualization
- Streamlit or FastAPI deployment
- ONNX export for optimized inference

---

## Author

Mohammed Suhail Ahmed Khan

GitHub: https://github.com/Suhail-Khan-06

Hugging Face: https://huggingface.co/SuhailKhan06

---

## License

This project is released under the MIT License.
