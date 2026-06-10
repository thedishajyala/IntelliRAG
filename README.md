# 🧠 IntelliRAG

**Your smart, chat-based PDF assistant.**

IntelliRAG lets you upload your PDFs and ask questions about them. It reads the documents and uses AI to give you accurate, grounded answers—without making things up (zero hallucinations). 

---

## ✨ Features

- **Upload & Ask:** Just drag-and-drop a PDF and start chatting!
- **Accurate Answers:** IntelliRAG strictly sticks to what's in your document.
- **Smart Conversations:** It remembers the context of your chat, so you can ask follow-up questions easily.
- **Beautiful UI:** A sleek, modern dark-mode interface built for a great experience.

---

## 🚀 Quick Start

### 1. Set Up Your API Key
Create a `.env` file in the main folder and add your Gemini API Key:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 2. Run Locally

Make sure you have Python 3.10+ installed.

```bash
# Set up a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the app
python app.py
```
Then, open **`http://localhost:5001`** in your browser to start chatting!

### 3. Run with Docker (Optional)
If you prefer Docker, you can start it up in one line:
```bash
docker-compose up --build
```
Open **`http://localhost:5001`**.

---

## 📂 Need More Details?
If you want to dive deep into how IntelliRAG works under the hood (Agentic Loops, FAISS Vector Stores, Vercel Deployment), check out our comprehensive [**Engineering Detail Guide**](detail.md).
