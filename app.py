import streamlit as st
import subprocess
import os
import json

# Set page config
st.set_page_config(page_title="RAG Metadata Enrichment UI", layout="wide")
st.title("📚 RAG Metadata Enrichment Pipeline")
st.markdown("A unified interface for chunking, enriching, and retrieving internal knowledge.")

# Helper function to run terminal commands and stream output
def run_command(command_list):
    with st.spinner(f"Running: {' '.join(command_list)}"):
        try:
            result = subprocess.run(command_list, capture_output=True, text=True, check=True)
            st.success("Execution Completed Successfully!")
            with st.expander("View Terminal Output"):
                st.code(result.stdout)
        except subprocess.CalledProcessError as e:
            st.error("An error occurred during execution.")
            with st.expander("View Error Log"):
                st.code(e.stderr)

# Create tabs for the pipeline steps
tab1, tab2, tab3, tab4 = st.tabs([
    "✂️ 1. Chunking", 
    "✨ 2. Metadata Enrichment", 
    "🧠 3. Embeddings", 
    "🔍 4. Retrieval & QA"
])

# ==========================================
# TAB 1: CHUNKING
# ==========================================
with tab1:
    st.header("Document Chunking")
    st.write("Break documents into meaningful segments using different strategies.")
    
    uploaded_file = st.file_uploader("Upload a text document (.txt)", type=["txt"])
    
    col1, col2 = st.columns(2)
    with col1:
        chunking_method = st.selectbox("Chunking Method", ["semantic", "recursive", "naive"])
        evaluate_chunks = st.checkbox("Run Evaluation after chunking?", value=False)
        
    with col2:
        if chunking_method == "semantic":
            sentence_model = st.text_input("Sentence Model", value="paraphrase-MiniLM-L3-v2")
            percentile = st.slider("Percentile Threshold", 50, 100, 95)
        elif chunking_method == "recursive":
            split_method = st.selectbox("Split Method", ["length", "delimiter"])
            max_length = st.number_input("Max Chunk Length", value=1000)
        elif chunking_method == "naive":
            chunk_by = st.selectbox("Chunk By", ["paragraph", "sentence"])

    if st.button("Run Chunker", type="primary"):
        if uploaded_file is not None:
            # Save uploaded file temporarily
            os.makedirs("input_files", exist_ok=True)
            file_path = os.path.join("input_files", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Build command
            cmd = ["python", "chunks.py", "--input_file", file_path, "--chunking_method", chunking_method]
            
            if chunking_method == "semantic":
                cmd.extend(["--sentence_model", sentence_model, "--percentile_threshold", str(percentile)])
            elif chunking_method == "recursive":
                cmd.extend(["--split_method", split_method, "--max_chunk_length", str(max_length)])
            elif chunking_method == "naive":
                cmd.extend(["--chunk_by", chunk_by])
                
            if evaluate_chunks:
                cmd.append("--evaluate")
                
            run_command(cmd)
        else:
            st.warning("Please upload a file first.")

# ==========================================
# TAB 2: METADATA ENRICHMENT
# ==========================================

with tab2:
    st.header("LLM Metadata Enrichment")
    st.write("Enhance chunks with rich semantic and structural metadata.")
    
    # We simplified the UI here because metadata_gen.py processes the whole directory at once
    evaluate_meta = st.checkbox("Run Metadata Evaluation?", value=False)
        
    if st.button("Generate Metadata", type="primary"):
        # Only passing the arguments that parse_arguments() in metadata_gen.py actually supports
        cmd = ["python", "metadata_gen.py"]
        
        if evaluate_meta:
            cmd.append("--evaluate")
            
        run_command(cmd)

# ==========================================
# TAB 3: EMBEDDINGS
# ==========================================
with tab3:
    st.header("Embedding Generation")
    st.write("Generate vector representations (Naive, TF-IDF Weighted, Prefix-Fusion).")
    
    emb_types = st.multiselect("Embedding Types", ["naive", "tfidf", "prefix"], default=["naive", "tfidf", "prefix"])
    chunk_types = st.multiselect("Chunking Types to Process", ["semantic", "naive", "recursive"], default=["semantic"])
    model_name = st.text_input("Embedding Model", value="Snowflake/arctic-embed-s")
    
    evaluate_emb = st.checkbox("Run Embedding Evaluation?", value=False)
    
    if st.button("Generate Embeddings", type="primary"):
        if not emb_types or not chunk_types:
            st.warning("Please select at least one embedding type and one chunking type.")
        else:
            cmd = ["python", "embeddings.py", 
                   "--embedding_types"] + emb_types + [
                   "--chunking_types"] + chunk_types + [
                   "--model", model_name]
            if evaluate_emb:
                cmd.append("--evaluate")
                
            run_command(cmd)

# ==========================================
# TAB 4: RETRIEVAL & QA
# ==========================================
with tab4:
    st.header("Retrieval & Answer Generation")
    st.write("Query your processed knowledge base and generate answers.")
    
    user_query = st.text_input("Enter your question:")
    
    col1, col2 = st.columns(2)
    with col1:
        retriever_type = st.selectbox("Retriever Strategy", ["content", "tfidf", "prefix", "reranker"])
        target_chunk_type = st.selectbox("Chunk Database", ["semantic", "naive", "recursive"])
    with col2:
        top_k = st.number_input("Top K Results", min_value=1, value=5)
    
    if st.button("Search & Answer", type="primary"):
        if user_query:
            # 1. Create a temporary queries.json file required by retriever.py
            query_data = [{"id": "q1", "query": user_query}]
            with open("temp_query.json", "w") as f:
                json.dump(query_data, f)
            
            # 2. Run Retriever
            st.info("Running Retrieval...")
            retriever_cmd = [
                "python", "retriever.py", 
                "--queries_file", "temp_query.json",
                "--retrievers", retriever_type,
                "--chunking_types", target_chunk_type,
                "--top_k", str(top_k)
            ]
            run_command(retriever_cmd)
            
            # 3. Run Prompt Answer Generation
            st.info("Generating LLM Answer...")
            # We assume it outputs to the default directory. We pass a generic run_ID pattern or find the latest.
            # For simplicity, we trigger prompt.py without explicit dir if it auto-detects, 
            # otherwise prompt.py requires --retrieval_dir. Let's assume you pass the latest dir.
            
            # Helper to find latest retrieval run
            try:
                base_dir = "retrieval_output"
                runs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))], key=os.path.getmtime, reverse=True)
                latest_run = runs[0] if runs else None
                
                if latest_run:
                    prompt_cmd = ["python", "prompt.py", "--retrieval_dir", latest_run, "--top_k", str(top_k)]
                    run_command(prompt_cmd)
                else:
                    st.error("Could not find retrieval outputs to generate answer.")
            except Exception as e:
                st.warning("Retrieval finished, but could not automatically trigger prompt.py. Check your terminal output.")
        else:
            st.warning("Please enter a query.")