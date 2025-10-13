import streamlit as st
from transformers import pipeline
import evaluate
from PIL import Image

@st.cache_resource
def load_model():
    captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    bleu = evaluate.load("bleu")
    llm = pipeline("text2text-generation", model="google/flan-t5-base")
    return captioner, bleu, llm

captioner, bleu, llm = load_model()

st.title("Image Captioning")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("See Captions"):
        col1, col2 = st.columns(2)
        

        with col2:
            with st.spinner("Generating detailed description..."):
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

                capt1 = "\n".join([f"- {param}" for  _,param in captions])
                prompt1 = f"Write about parameters that are used for descriptive caption in very short:\n{capt1}"
                result1 = llm(prompt1)[0]['generated_text']
                st.subheader("LLM Description")
                st.write(result1)

        with col1:
            with st.spinner("Generating default caption..."):
                default_caption = captioner(image)[0]['generated_text']
                st.subheader("Default Caption")
                st.success(default_caption)

            with st.spinner("Calculating BLEU scores..."):
                scores = []
                for cap, param in captions:
                    result = bleu.compute(predictions=[cap], references=[default_caption])
                    scores.append((cap, param, result["bleu"]))

                scores.sort(key=lambda x: x[2], reverse=True)
                top5 = scores[:5]

                st.subheader("Top 5 Captions")
                for cap, param, score in top5:
                    st.markdown(f"**{param}** : {cap} _(BLEU: {score:.4f})_")

            with st.spinner("Merging captions..."):
                top5_texts = "\n".join([f"- {cap}" for cap, _, _ in top5])
                prompt = f"Merge the following captions into one clear, descriptive caption:\n{top5_texts}"
                merged = llm(prompt)[0]['generated_text']
                st.subheader("Merged Caption")
                st.info(merged)
