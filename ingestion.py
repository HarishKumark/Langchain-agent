import os

from dotenv import load_dotenv
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

if __name__ == "__main__":

    print("Starting ingestion")

    # Step 1: Load documents
    loader = UnstructuredLoader(
        file_path="data.txt",
        partition_via_api=False,
        chunking_strategy="basic",
        max_characters=1000000
    )

    documents = loader.load()

    print(f"Loaded {len(documents)} documents")

    # Step 2: Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} chunks")

    print(
        f"Largest chunk: "
        f"{max(len(t.page_content) for t in texts)} characters"
    )

    # Step 3: Initialise embeddings
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Step 4: Determine embedding dimensions
    sample_vector = embeddings.embed_query("Hello world")
    dimension = len(sample_vector)

    print(f"Embedding dimensions: {dimension}")

    # Step 5: Connect to Pinecone
    pc = Pinecone(
        api_key=os.environ["PINECONE_API_KEY"]
    )

    index_name = os.environ["INDEX_NAME"]

    # Step 6: Create index if it does not exist
    if index_name not in pc.list_indexes().names():

        print(f"Creating index: {index_name}")

        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    else:
        print(f"Index already exists: {index_name}")

    # Step 7: Verify index dimensions
    index_info = pc.describe_index(index_name)

    if index_info.dimension != dimension:
        raise ValueError(
            f"Dimension mismatch: "
            f"Index={index_info.dimension}, "
            f"Embeddings={dimension}"
        )

    # Step 8: Store vectors in Pinecone
    print("Uploading to Pinecone")

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=index_name
    )

    print("Ingestion completed successfully")