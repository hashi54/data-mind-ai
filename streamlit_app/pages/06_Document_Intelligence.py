import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app.rag.pipeline import RAGPipeline
from app.rag.vector_store import vector_store
from app.core.config import settings

from streamlit_app.components.cards import render_metric_card

st.set_page_config(page_title="Document Intelligence | DataMind AI", page_icon="📚", layout="wide")

st.markdown("## 📚 Corporate Document Intelligence & Policy RAG")
st.caption("Semantic vector search, policy compliance QA, and knowledge base document ingestion")

# Knowledge Base Overview Cards
docs_dir = Path(settings.DOCUMENTS_DIR)
doc_files = list(docs_dir.glob("*.md")) + list(docs_dir.glob("*.txt"))

c1, c2, c3 = st.columns(3)
with c1:
    render_metric_card("Knowledge Base Docs", f"{len(doc_files)}", "Policy & Strategy Files", icon="📄")
with c2:
    render_metric_card("Vector Store Chunks", f"{vector_store.get_document_count()}", "Indexed in Vector DB", icon="🧩")
with c3:
    render_metric_card("Embedding Model", "TF-IDF + Cosine Vectors", "Sublinear TF & N-Grams", icon="⚡")

st.divider()

col_query, col_docs = st.columns([6, 4])

with col_query:
    st.markdown("### 🔍 Ask Policy Questions")
    sample_queries = [
        "What is our refund policy on consumer electronics?",
        "What is the employee travel allowance for Mumbai and Delhi?",
        "What are the specifications of the AeroBook Pro laptop?",
        "What are our discount approval thresholds for sales managers?",
    ]
    selected_sample = st.selectbox("Quick Sample Query:", ["Custom..."] + sample_queries)
    
    user_query = st.text_input("Enter question about company policies, products, or guidelines:", 
                               value="" if selected_sample == "Custom..." else selected_sample)

    if st.button("Search Knowledge Base", type="primary", use_container_width=True) and user_query:
        with st.spinner("Searching semantic vector index..."):
            rag_res = RAGPipeline.query(user_query, top_k=3)
            chunks = rag_res["retrieved_chunks"]

        if chunks:
            top_chunk = chunks[0]
            st.markdown("#### 🎯 AI Policy Answer")
            st.info(f"{top_chunk['content']}")

            st.markdown("#### 📑 Retrieved Document Citations")
            for i, c in enumerate(chunks):
                with st.expander(f"Citation #{i+1}: {c['document_name']} ({c['department']}) — Score: {c['similarity_score']:.3f}", expanded=(i == 0)):
                    st.markdown(c["content"])
                    st.caption(f"Section Index: {c['page_number']} | Document: `{c['document_name']}`")
        else:
            st.warning("No matching policy documents found.")

with col_docs:
    st.markdown("### 📁 Active Knowledge Base")
    for f in doc_files:
        with st.expander(f"📄 {f.name} ({round(f.stat().st_size/1024, 1)} KB)"):
            with open(f, "r", encoding="utf-8") as file_read:
                st.markdown(file_read.read()[:600] + "...\n\n*(Full document indexed in vector database)*")

    st.markdown("### 📤 Upload New Document")
    uploaded = st.file_uploader("Upload policy document (.md, .txt)", type=["md", "txt"])
    if uploaded:
        target_path = docs_dir / uploaded.name
        with open(target_path, "wb") as f_out:
            f_out.write(uploaded.getbuffer())
        
        with st.spinner("Indexing new document into vector store..."):
            total = RAGPipeline.index_knowledge_base()
        st.success(f"Indexed '{uploaded.name}'! Total vector chunks: {total}")
        st.rerun()
