import streamlit as st
import os
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import SpladeEncoder
from pinecone_text.hybrid import hybrid_convex_scale
import hashlib

def chunk_text(text, chunk_size=1000):
    """Chunks text into smaller pieces."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def main():
    st.title("Document to Pinecone Vectorstore")

    # Get Pinecone credentials from secrets
    try:
        pinecone_api_key = st.secrets["pinecone"]["api_key"]
        pinecone_index_name = st.secrets["pinecone"]["index_name"]
    except KeyError:
        st.error("Please add your Pinecone API key and index name to .streamlit/secrets.toml")
        st.stop()

    # Initialize Pinecone
    pc = Pinecone(api_key=pinecone_api_key)

    # Check if the index exists, if not, create it
    if pinecone_index_name not in pc.list_indexes().names():
        st.info(f"Creating index '{pinecone_index_name}'...")
        pc.create_index(
            name=pinecone_index_name,
            dimension=768,  # Standard dimension for many models, can be adjusted
            metric='dotproduct',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )
        st.success(f"Index '{pinecone_index_name}' created successfully.")

    index = pc.Index(pinecone_index_name)
    splade = SpladeEncoder()

    uploaded_file = st.file_uploader("Upload a document (.txt)", type="txt")

    if uploaded_file is not None:
        # Read the file content
        try:
            document_text = uploaded_file.read().decode("utf-8")
            st.text_area("Document Content", document_text, height=200)

            if st.button("Chunk and Upsert to Pinecone"):
                with st.spinner("Processing..."):
                    # Chunk the text
                    chunks = chunk_text(document_text)
                    st.write(f"Document chunked into {len(chunks)} pieces.")

                    if not document_text.strip():
                        st.warning("The uploaded document is empty.")
                        st.stop()

                    # Chunk the text
                    chunks = chunk_text(document_text)
                    st.write(f"Document chunked into {len(chunks)} pieces.")

                    # Create sparse vectors using SPLADE
                    st.write("Creating sparse vectors...")
                    sparse_vecs = splade.encode_documents(chunks)

                    # Prepare vectors for upsert
                    vectors_to_upsert = []
                    for i, (chunk, sparse_vec) in enumerate(zip(chunks, sparse_vecs)):
                        # Create a unique ID for each chunk
                        chunk_id = hashlib.sha256(chunk.encode()).hexdigest()

                        # For this basic app, we will use a placeholder for dense vectors
                        # as pinecone-text focuses on sparse and hybrid.
                        # In a real app, you would generate dense vectors here.
                        dense_vec = [0.1] * 768 # Placeholder dense vector

                        vectors_to_upsert.append({
                            "id": chunk_id,
                            "sparse_values": sparse_vec,
                            "values": dense_vec,
                            "metadata": {"text": chunk}
                        })

                    # Upsert to Pinecone
                    st.write("Upserting vectors to Pinecone...")
                    index.upsert(vectors=vectors_to_upsert)
                    st.success("Successfully upserted document chunks to Pinecone!")

        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
