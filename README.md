# Multi-Label Emotion Classifier

A fine-tuned RoBERTa model that detects **multiple co-occurring emotions** in a piece of text, served through an interactive web app. Unlike standard single-label sentiment classification, this model treats emotions as non-exclusive — a sentence like *"I'm exhausted but weirdly hopeful about tomorrow"* correctly surfaces both **fatigue** and **joy** at once.

**[▶ Live app](https://siddharthravindran-emotion-classifier.streamlit.app)** · **[Model on Hugging Face](https://huggingface.co/Siddharthr30/emotion-model)**

---

## What it does

The model scores text across nine emotion families, each predicted independently:

`sadness` · `fatigue` · `anxiety` · `confused` · `anger` · `physical` · `engaged` · `calm` · `joy`

Because emotions overlap in real language, the architecture uses a sigmoid output per label (not a softmax over mutually-exclusive classes) and an adjustable confidence threshold, so the app can return zero, one, or several emotions for a single input.

## Model & approach

- **Base model:** `roberta-base`, fine-tuned with `AutoModelForSequenceClassification` and `problem_type="multi_label_classification"` (BCE-with-logits over the nine labels).
- **Data:** ~3,200 labeled examples across the nine emotion families, split into separate train / validation / test sets.
- **Training:** Hugging Face `Trainer` on Python 3.12 — learning rate 2e-5, effective batch size 16 (8 × 2 gradient-accumulation steps), 5 epochs, weight decay 0.01. Best checkpoint selected by validation micro-F1 (`load_best_model_at_end`).
- **Inference:** sigmoid per label; reported metrics use a 0.5 decision threshold, while the app exposes an adjustable threshold (default 0.40) so users can trade precision against recall.

## Results

Held-out evaluation:

| Metric        | Score |
|---------------|-------|
| F1 (micro)    | 0.846 |
| F1 (macro)    | 0.825 |
| Precision     | 0.882 |
| Recall        | 0.812 |
| Eval loss     | 0.140 |

The small gap between micro- and macro-F1 (~0.02) is the meaningful signal here: on imbalanced multi-label data, a model can post a strong micro score while quietly failing the rarer classes. The near-parity means performance holds up across the underrepresented emotion families, not just the common ones.

## Stack modernization

This project was rebuilt from a deprecated `lightning-transformers` training stack onto the current Hugging Face `Trainer` API running on Python 3.12, which involved resolving the full NumPy 2.x / PyTorch compatibility chain and the associated kernel/environment setup. The notebook reflects the current, maintained tooling rather than the original deprecated path.

## Repository structure

```
app.py                              # Streamlit inference app (loads model from HF Hub)
sentiment_analysis_classifier.ipynb # Training + evaluation notebook
sentiment_analysis_functions.py     # Data prep / training helper functions
requirements.txt                    # Runtime dependencies
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads the fine-tuned weights directly from the [Hugging Face Hub](https://huggingface.co/Siddharthr30/emotion-model) at runtime, so no local model files are required.
