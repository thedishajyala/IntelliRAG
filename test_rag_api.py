import os
from dotenv import load_dotenv
from src.rag_core import RAGSystem

load_dotenv()

def test_rag():
    print("Testing RAG System...")
    rag = RAGSystem()
    
    if not rag.index:
        print("Index is empty. Please ensure there are PDFs in data/docs.")
        # Create a dummy pdf or something? No, let's just see.
        return

    question = "What is this document about?"
    print(f"Asking: {question}")
    answer = rag.ask(question)
    print(f"Answer: {answer}")
    print(f"Memory length: {len(rag.conversation_memory)}")

if __name__ == "__main__":
    test_rag()
