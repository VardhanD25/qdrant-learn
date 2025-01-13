from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from qdrant_client import QdrantClient

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

url = "http://localhost:6333"
collection_name="vector_db"

client=QdrantClient(
    url=url,
    prefer_grpc=False,
)

print("Client:",client)
print("-----------------------------")

db=Qdrant(
    client=client,
    embeddings=embeddings,
    collection_name=collection_name
)

query="How well did GPT-4 perform on professional as well as academic exams?"

docs=db.similarity_search_with_score(query=query,k=5)

for i in range(len(docs)):
    doc,score=docs[i]
    print("Score for {}:{}".format(i+1,score))
    print("Content for {}:{}".format(i+1,doc.page_content))
    print("Metadata for chunk {}:{}".format(i+1,doc.metadata))
