from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 1. Load PDF
reader = PdfReader("data/docs/sample_company_policy.pdf")
full_text = ""

for page in reader.pages:
    full_text += page.extract_text()

# 2. Chunking function
def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks

chunks = chunk_text(full_text)

print(f"Total chunks created: {len(chunks)}")

# 3. Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 4. Embed chunks
embeddings = model.encode(chunks)

# 5. Store in FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# 6. Ask a question
question = "Why did the server crash?"
question_embedding = model.encode([question])

# 7. Retrieve top matches
k = 2
distances, indices = index.search(question_embedding, k)

print("\nTop retrieved chunks:\n")
for i in indices[0]:
    print(chunks[i])
    print("-" * 50)
