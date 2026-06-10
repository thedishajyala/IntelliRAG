from pypdf import PdfReader
from google import genai
import faiss
import numpy as np
import os
import pickle

# ---------------- FEATURE TOGGLES ----------------
USE_QUERY_REWRITE = True      # Turn OFF if too slow
USE_VALIDATION = False        # Turn ON only if you accept high latency
TOP_K = 2
MAX_TOKENS = 512
# -------------------------------------------------


class RAGSystem:
    def __init__(self):
        print("Initializing RAG system (models load ONCE)")

        # Configure Gemini API Client
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Gemini LLM model name
        self.llm_model = "models/gemini-flash-latest"
        self.embed_model = "models/gemini-embedding-001"

        self.chunks = []
        self.index = None
        self.conversation_memory = []

        self._load_or_build_index()

    # --------------------------------------------------
    # Embed text using Gemini embedding API
    # --------------------------------------------------
    def _embed(self, texts):
        """Embed a list of texts using Gemini embedding model."""
        if isinstance(texts, str):
            texts = [texts]
        
        # New google-genai batch embedding
        response = self.client.models.embed_content(
            model=self.embed_model,
            contents=texts,
            config={'task_type': 'RETRIEVAL_DOCUMENT'}
        )
        
        embeddings = [item.values for item in response.embeddings]
        return np.array(embeddings, dtype=np.float32)

    def _embed_query(self, text):
        """Embed a single query string."""
        response = self.client.models.embed_content(
            model=self.embed_model,
            contents=text,
            config={'task_type': 'RETRIEVAL_QUERY'}
        )
        # For single query, it returns a list of one
        return np.array([response.embeddings[0].values], dtype=np.float32)

    # --------------------------------------------------
    # Load documents and cached vector store
    # --------------------------------------------------
    def _load_or_build_index(self):
        cache_path = "/tmp/cache/vector_store.pkl"

        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    self.chunks, self.index = pickle.load(f)
                print("Loaded cached vector store")
                return
            except Exception as e:
                print(f"Error loading cache: {e}. Rebuilding...")

        print("Building vector store from documents...")
        full_text = ""

        doc_dir = "/tmp/docs"
        os.makedirs(doc_dir, exist_ok=True)

        files = [f for f in os.listdir(doc_dir) if f.endswith(".pdf")]
        if not files:
            print(f"No documents found in {doc_dir}. Upload a PDF to get started.")
            return

        for file in files:
            try:
                reader = PdfReader(os.path.join(doc_dir, file))
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        full_text += text
            except Exception as e:
                print(f"Error reading {file}: {e}")

        self.chunks = self._chunk_text(full_text)

        if not self.chunks:
            print("No text extracted from documents.")
            return

        print("Generating embeddings via Gemini...")
        try:
            embeddings = self._embed(self.chunks)
            dim = embeddings.shape[1]

            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)

            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump((self.chunks, self.index), f)

            print("Vector store built and cached")
        except Exception as e:
            print(f"Error building index: {e}")

    # --------------------------------------------------
    # Rebuild index (called after new PDF upload)
    # --------------------------------------------------
    def rebuild_index(self):
        cache_path = "/tmp/cache/vector_store.pkl"
        if os.path.exists(cache_path):
            os.remove(cache_path)

        self.chunks = []
        self.index = None
        self.conversation_memory = []

        self._load_or_build_index()

    # --------------------------------------------------
    # Chunking logic
    # --------------------------------------------------
    def _chunk_text(self, text, chunk_size=400, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks

    # --------------------------------------------------
    # Query Rewriting Agent
    # --------------------------------------------------
    def _rewrite_query(self, user_query):
        prompt = f"""
Rewrite the question into a short, clear search query
focused only on factual document information.

Question:
{user_query}

Rewritten query:
"""
        response = self.client.models.generate_content(
            model=self.llm_model,
            contents=prompt
        )
        return response.text.strip()

    # --------------------------------------------------
    # Validation Agent (OPTIONAL, SLOW)
    # --------------------------------------------------
    def _validate_answer(self, context, answer):
        prompt = f"""
Determine whether the answer is fully supported by the context.

Context:
{context}

Answer:
{answer}

Reply ONLY:
SUPPORTED
or
NOT_SUPPORTED
"""
        response = self.client.models.generate_content(
            model=self.llm_model,
            contents=prompt
        )
        verdict = response.text.strip().upper()
        return verdict.startswith("SUPPORTED")

    # --------------------------------------------------
    # MAIN RAG PIPELINE
    # --------------------------------------------------
    def ask(self, user_question):

        if not self.index:
            return "No documents indexed yet. Please upload a PDF first."

        # ----- Step 1: Rewrite query (optional) -----
        if USE_QUERY_REWRITE:
            try:
                query = self._rewrite_query(user_question)
            except:
                query = user_question
        else:
            query = user_question

        # ----- Step 2: Embed query -----
        try:
            query_embedding = self._embed_query(query)
        except Exception as e:
            print(f"Embedding error: {e}")
            return "Error generating embeddings for the query."

        # ----- Step 3: Retrieve context -----
        _, indices = self.index.search(query_embedding, TOP_K)
        context = "\n".join([self.chunks[i] for i in indices[0]])

        # ----- Step 4: Conversation memory -----
        memory_context = "\n".join(
            [f"Q: {m['question']}\nA: {m['answer']}"
             for m in self.conversation_memory[-3:]]
        )

        # ----- Step 5: Answer generation -----
        prompt = f"""
You are an assistant that answers ONLY using the provided document context.
If the answer is not present, say "I don't know based on the document".

Previous conversation:
{memory_context}

Document Context:
{context}

Question:
{user_question}

Answer:
"""
        try:
            response = self.client.models.generate_content(
                model=self.llm_model,
                contents=prompt
            )
            answer = response.text.strip()
        except Exception as e:
            print(f"Generation error: {e}")
            answer = "Error generating response from LLM."

        # ----- Step 6: Validation (optional) -----
        if USE_VALIDATION:
            try:
                is_valid = self._validate_answer(context, answer)
                if not is_valid:
                    answer = "I don't know based on the document."
            except:
                pass

        # ----- Step 7: Store memory -----
        self.conversation_memory.append({
            "question": user_question,
            "answer": answer
        })

        return answer
