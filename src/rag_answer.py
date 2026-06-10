from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from ctransformers import AutoModelForCausalLM
import faiss
import numpy as np

# ---------- 1. Load document ----------
reader = PdfReader("data/docs/sample_company_policy.pdf")
full_text = ""
for page in reader.pages:
    full_text += page.extract_text()

# ---------- 2. Chunking ----------
def chunk_text(text, chunk_size=400, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

chunks = chunk_text(full_text)

# ---------- 3. Embeddings ----------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunk_embeddings = embed_model.encode(chunks)

# ---------- 4. Vector index ----------
dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(chunk_embeddings))

# ---------- 5. Load local LLM ----------
llm = AutoModelForCausalLM.from_pretrained(
    "models",
    model_file="mistral.gguf",
    model_type="mistral"
)

# ---------- 6. Ask question ----------
question = "Why did the server crash?"
question_embedding = embed_model.encode([question])

# ---------- 7. Retrieve relevant chunks ----------
k = 2
_, indices = index.search(question_embedding, k)
context = "\n".join([chunks[i] for i in indices[0]])

# ---------- 8. Prompt with context ----------
prompt = f"""
You are an assistant that answers ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}

Answer:
"""

# ---------- 9. Generate answer ----------
answer = llm(prompt, max_new_tokens=150)
print("\nFINAL ANSWER:\n")
print(answer)
