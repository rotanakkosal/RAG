import json
from pydoc import text
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np

data_path = "/Users/macbookpro/Documents/AI-Research/dynamic-topk-value-rag/dataset/qasper/dummy/qasper/0.1.0/dummy_data/qasper-train-dev-v0.1.tgz/qasper-train-v0.1.json"
passages_path = "/Users/macbookpro/Documents/AI-Research/dynamic-topk-value-rag/dataset.passages.json"

# Extract the text from dataset
def extract_passages(data_path):
    with open(data_path, 'r') as f:
        data = json.load(f)
        passages = []
        for doc_id, doc in data.items():
            for section in doc['full_text']:
                for paragraph in section['paragraphs']:
                    dataObject = {
                        "doc_id": doc_id ,
                        "text": paragraph.strip()
                    }
                    passages.append(dataObject);
        return passages

passages = extract_passages(data_path)

# Chunk and Embed the passages
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [ p['text'] for p in passages] 
embeddings = model.encode(texts, convert_to_numpy=True)


# Create FAISS Index
vector_dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(vector_dimension) 
faiss.normalize_L2(embeddings)
index.add(np.array(embeddings))

# map index id from vector back to text
id_to_passage = {i: passage for i, passage in enumerate(passages)}

# Testig with sample question
question = "Which multilingual approaches do they compare with?"
q_embedding = model.encode([question])

# Retrieve query index from Vector DB
D, I = index.search(np.array(q_embedding), k=8)

print("D is :",D)
print("I is :",I)

top_k_passages = [id_to_passage[i]['text'] for i in I[0]]

# print("Top Retrieved chunks", top_k_passages)
print("Top Retrieved Chunks:\n", "\n---\n".join(top_k_passages))
