import streamlit as st
from transformers import pipeline
import evaluate
from PIL import Image
import torch, clip

@st.cache_resource
def load_all_models():
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    bleu = evaluate.load("bleu")
    clip_model, preprocess = clip.load("ViT-B/32", device="cpu")
    return captioner, bleu, clip_model, preprocess

captioner, bleu, clip_model, preprocess = load_all_models()

domains = [
    # Type 1: Human & Lifestyle
    "people_daily_life", "transport_and_vehicles", "food_drink",
    "sports_recreation", "fashion_accessories",
    # Type 2: Environment & Objects
    "animals_wildlife", "outdoor_nature_landscapes",
    "home_furniture", "electronics_devices", "plants_gardens"
]

with torch.no_grad():
    text_feat = clip_model.encode_text(clip.tokenize(domains))
    text_feat /= text_feat.norm(dim=-1, keepdim=True)

st.title("Image Captioning + Domain Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.spinner("### Generating captions... Please wait")

    temperature = [0.1, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 1.9]
    top_k = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    top_p = [0.2, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    typical_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    captions = []

    for t in temperature:
        cap = captioner(image, generate_kwargs={"do_sample": True, "temperature": t})[0]['generated_text']
        captions.append((cap,f"temperature={t}"))

    for k in top_k:
        cap = captioner(image, generate_kwargs={"do_sample": True, "top_k": k})[0]['generated_text']
        captions.append((cap, f"top_k={k}"))

    for p in top_p:
        cap = captioner(image, generate_kwargs={"do_sample": True, "top_p": p})[0]['generated_text']
        captions.append((cap,f"top_p={p}"))

    for tp in typical_p:
        cap = captioner(image, generate_kwargs={"do_sample": True, "typical_p": tp})[0]['generated_text']
        captions.append((cap,f"typical_p={tp}"))

    default_caption = captioner(image)[0]['generated_text']

    st.spinner("### Evaluating captions (BLEU score)...")
    scores = []
    for cap, param in captions:
        result = bleu.compute(predictions=[cap], references=[default_caption])
        scores.append((cap, param, result["bleu"]))

    scores.sort(key=lambda x: x[2], reverse=True)
    top5 = scores[:5]

    st.subheader("Default Caption")
    st.success(default_caption)

    st.subheader("Top 5 Captions (by BLEU score)")
    for cap, param, score in top5:
        st.markdown(f"**{param}** : {cap} _(BLEU: {score:.4f})_")

    st.subheader("Predicted Domain (via CLIP)")
    with torch.no_grad():
        img = preprocess(image).unsqueeze(0)
        img_feat = clip_model.encode_image(img)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        best_idx = (img_feat @ text_feat.T).softmax(-1).argmax().item()
        pred_domain = domains[best_idx]

    st.info(f"This image belongs to the **'{pred_domain}'** domain.")
