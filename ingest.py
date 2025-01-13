from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from datetime import datetime

# Load PDF and split into chunks
loader = PyPDFLoader("data.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)

# Load the embedding model
model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding model Loaded")

# Prepare metadata and enrich each chunk
enriched_chunks = []
for i, chunk in enumerate(chunks):
    chunk_id = f"doc123_chunk_{i + 1}"  # Example ID generation
    metadata = {
        "chunk_id": chunk_id,
        "content": chunk.page_content,
        "primary_title": "Introduction to Machine Learning Algorithms",
        "secondary_title": "Training Neural Networks",
        "primary_description": "Overview of key machine learning algorithms and their applications.",
        "secondary_description": "This section discusses various popular algorithms such as Decision Trees, SVMs, and Neural Networks, highlighting their uses and application in real-world scenarios.",
        "summarized_version": "This section covers neural network architectures, training techniques, and key algorithms in machine learning.",
        "tags": ["machine_learning", "neural_networks", "pytorch", "training"],
        "timestamps": {
            "content_created": datetime.utcnow().isoformat(),
            "content_last_updated": datetime.utcnow().isoformat(),
            "document_version": "2.1"
        },
        "source": {
            "document_id": "doc123",
            "document_title": "Deep Learning Best Practices Guide",
            "author": "Jane Smith",
            "publisher": "ML Research Institute"
        },
        "related_chunks": {
            "previous_chunk": f"doc123_chunk_{i}" if i > 0 else None,
            "next_chunk": f"doc123_chunk_{i + 2}" if i < len(chunks) - 1 else None,
            "semantic_neighbors": []  # Populate this based on your application logic
        },
        "confidence_scores": {
            "chunk_quality": 0.95,
            "semantic_coherence": 0.88,
            "information_density": 0.82,
            "embedding_quality": 0.91
        },
        "content_type": {
            "primary_type": "text",
            "secondary_types": ["prose"],
            "language": "English"
        },
        "processing_info": {
            "chunk_size": 512,
            "overlap_previous": 50,
            "overlap_next": 50,
            "embedding_model": model_name,
            "chunk_strategy": "sentence_boundary"
        }
    }
    # Add metadata to chunk
    chunk.metadata = metadata
    enriched_chunks.append(chunk)

# Initialize Qdrant and store enriched chunks with metadata
url = "http://localhost:6333"
qdrant = Qdrant.from_documents(
    enriched_chunks,
    embeddings,
    url=url,
    prefer_grpc=False,
    collection_name="vector_db"
)

print("Vector DB Successfully Created!")
