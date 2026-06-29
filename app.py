import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Emotion Classifier", page_icon="🎭")

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained("siddharthravindran/emotion-model")
    mdl = AutoModelForSequenceClassification.from_pretrained("siddharthravindran/emotion-model")
    mdl.eval()
    return tok, mdl

tokenizer, model = load_model()
id2label = model.config.id2label

st.title("🎭 Emotion Classifier")
st.caption("Multi-label emotion detection across 9 families — fine-tuned RoBERTa")

text = st.text_area("Type how you're feeling:", "I'm exhausted but weirdly hopeful about tomorrow.")
threshold = st.slider("Confidence threshold", 0.1, 0.9, 0.4, 0.05)

if st.button("Analyze") and text.strip():
    inputs = tokenizer(text, truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = torch.sigmoid(model(**inputs).logits)[0].numpy()
    scored = sorted(
        [(id2label[i], float(probs[i])) for i in range(len(probs))],
        key=lambda x: x[1], reverse=True
    )
    predicted = [f for f, p in scored if p >= threshold]
    st.subheader("Detected: " + (", ".join(predicted) if predicted else "—"))
    for fam, p in scored:
        st.write(f"**{fam}**")
        st.progress(p)