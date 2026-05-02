import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

labels = ["billing", "technical", "account", "shipping", "refund", "password", "hardware", "software"]

st.title("🎫 Support Ticket Auto Tagger")
st.write("Enter a support ticket and get the top 3 most likely tags.")

ticket = st.text_area("Support Ticket", placeholder="e.g. My internet keeps dropping...")

if st.button("Tag Ticket"):
    if ticket:
        with st.spinner("Analyzing..."):
            result = classifier(ticket, labels, multi_label=False)
            top3 = list(zip(result["labels"][:3], result["scores"][:3]))

        st.subheader("Top 3 Tags:")
        for i, (label, score) in enumerate(top3, 1):
            st.progress(float(score))
            st.write(f"**{i}. {label}** — {round(score*100, 1)}%")
    else:
        st.warning("Please enter a ticket first.")