from flask import Flask, render_template, request
import os
from dotenv import load_dotenv

load_dotenv()
from src.rag_core import RAGSystem
import src.rag_core as rc

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "/tmp/docs"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

rag = RAGSystem()  # models load ONCE

@app.route("/", methods=["GET", "POST"])
def index():
    message = None

    if request.method == "POST":

        # Clear chat memory
        if "clear" in request.form:
            rag.conversation_memory = []
            message = "Conversation history cleared."

        # Upload document
        elif "document" in request.files:
            file = request.files["document"]
            if file and file.filename.endswith(".pdf"):
                save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(save_path)
                rag.rebuild_index()
                message = "Document uploaded and indexed."

        # Ask question
        elif "question" in request.form:
            question = request.form["question"].strip()
            if question:
                # VERY IMPORTANT: blocking call
                rag.ask(question)

    # Get list of uploaded documents
    uploaded_files = []
    if os.path.exists(app.config["UPLOAD_FOLDER"]):
        for f in os.listdir(app.config["UPLOAD_FOLDER"]):
            if f.lower().endswith(".pdf"):
                path = os.path.join(app.config["UPLOAD_FOLDER"], f)
                try:
                    size_kb = round(os.path.getsize(path) / 1024, 1)
                except Exception:
                    size_kb = 0.0
                uploaded_files.append({"name": f, "size": size_kb})

    return render_template(
        "index.html",
        history=rag.conversation_memory,
        message=message,
        uploaded_files=uploaded_files,
        query_rewrite=rc.USE_QUERY_REWRITE,
        validation=rc.USE_VALIDATION
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
