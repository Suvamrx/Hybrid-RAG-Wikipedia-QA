import streamlit as st
from generate_response_llm import build_context, generate_answer
import time

# =====================================
# Streamlit User Interface for Hybrid RAG
# =====================================
# This app provides an interactive QA interface for the Hybrid RAG system.
# It displays user input, generated answer, top retrieved chunks, RRF scores, and response time.

st.set_page_config(page_title="Hybrid RAG Wikipedia QA", layout="wide")
st.title("Hybrid RAG Wikipedia QA System")

query = st.text_input("Enter your question:")
run = st.button("Get Answer")

if run and query:
    start_time = time.time()
    with st.spinner('Retrieving and generating answer...'):
        context, fused = build_context(query)
        answer = generate_answer(query, context)
    elapsed = time.time() - start_time
    st.info(f"Response time: {elapsed:.2f} seconds")
    st.subheader("Generated Answer")
    st.write(answer)
    st.subheader("Top Retrieved Chunks (RRF)")
    for i, r in enumerate(fused):
        st.markdown(f"**Rank {i+1}** | RRF Score: {r['rrf_score']:.4f}")
        st.markdown(f"**Title:** {r['title']}")
        st.markdown(f"**URL:** [{r['url']}]({r['url']})")
        st.markdown(f"**Chunk:** {r['text']}")
        st.markdown("---")
    
