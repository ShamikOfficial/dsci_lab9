import os
import shutil

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from App_p1 import (
    BACKEND_OPENAI,
    BACKEND_LLAMA,
    BACKENDS,
    PDF_DIR,
    get_embeddings,
    get_faiss_index_dir,
    get_llm,
    load_pdfs,
    make_chunks,
    build_vector_store,
    load_vector_store,
    create_conversation_chain,
)

app = Flask(
    __name__,
    template_folder=".",
    static_folder=".",
    static_url_path=""
)

ALLOWED_EXTENSIONS = {"pdf"}

# in-memory conversation chains
conversation_chains = {
    BACKEND_OPENAI: None,
    BACKEND_LLAMA: None,
}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_pdf_dir() -> None:
    os.makedirs(PDF_DIR, exist_ok=True)


def clear_pdf_dir() -> None:
    ensure_pdf_dir()
    for name in os.listdir(PDF_DIR):
        path = os.path.join(PDF_DIR, name)
        if os.path.isfile(path):
            os.remove(path)


def reset_index_dir(index_dir: str) -> None:
    if os.path.isdir(index_dir):
        shutil.rmtree(index_dir)


def build_chain_for_backend(backend: str):
    index_dir = get_faiss_index_dir(backend)
    embeddings = get_embeddings(backend)

    if os.path.isdir(index_dir):
        store = load_vector_store(embeddings, index_dir)
    else:
        records = load_pdfs()
        if not records:
            raise ValueError("No readable text found in uploaded PDFs.")
        chunks = make_chunks(records)
        if not chunks:
            raise ValueError("No chunks were created from the uploaded PDFs.")
        store = build_vector_store(chunks, embeddings, index_dir)

    llm = get_llm(backend)
    chain = create_conversation_chain(store, llm)
    return chain


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    backend = request.form.get("backend", BACKEND_OPENAI).strip().lower()

    if backend not in BACKENDS:
        return jsonify({"success": False, "error": "Invalid backend"}), 400

    files = request.files.getlist("pdfs")
    if not files:
        return jsonify({"success": False, "error": "No PDF files uploaded"}), 400

    try:
        clear_pdf_dir()

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(PDF_DIR, filename))

        index_dir = get_faiss_index_dir(backend)
        reset_index_dir(index_dir)

        chain = build_chain_for_backend(backend)
        conversation_chains[backend] = chain

        return jsonify({
            "success": True,
            "message": f"PDFs processed successfully using {backend} backend."
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Missing JSON body"}), 400

    question = data.get("question", "").strip()
    backend = data.get("backend", BACKEND_OPENAI).strip().lower()

    if backend not in BACKENDS:
        return jsonify({"success": False, "error": "Invalid backend"}), 400

    if not question:
        return jsonify({"success": False, "error": "Question is empty"}), 400

    chain = conversation_chains.get(backend)
    if chain is None:
        return jsonify({"success": False, "error": "Please analyze PDFs first"}), 400

    try:
        result = chain.invoke({"question": question})
        answer = result.get("answer") or result.get("result") or ""
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    ensure_pdf_dir()
    app.run(debug=True)