import os

from flask import Flask, render_template, request, jsonify

from App_p1 import (
    BACKEND_OPENAI,
    DEFAULT_BACKEND,
    get_backends,
    get_embeddings,
    get_faiss_index_dir,
    get_llm,
    load_vector_store,
    create_conversation_chain,
)

app = Flask(
    __name__,
    template_folder=".",
    static_folder=".",
    static_url_path="",
)

# Chains loaded on demand; only OpenAI is loaded at startup by default
conversation_chains: dict = {}


def load_chain_for_backend(backend: str):
    """Load the FAISS index and create the conversation chain. No PDF processing."""
    index_dir = get_faiss_index_dir(backend)
    if not os.path.isdir(index_dir):
        return None
    embeddings = get_embeddings(backend)
    store = load_vector_store(embeddings, index_dir)
    llm = get_llm(backend)
    return create_conversation_chain(store, llm)


def init_chains():
    """Load only OpenAI at startup. Other models load on selection."""
    try:
        chain = load_chain_for_backend(BACKEND_OPENAI)
        conversation_chains[BACKEND_OPENAI] = chain
        if chain is not None:
            print("Loaded OpenAI backend (default).")
    except Exception as e:
        print(f"Could not load OpenAI backend: {e}")
        conversation_chains[BACKEND_OPENAI] = None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/backends")
def backends():
    """Return list of models: OpenAI + each file in models/ by filename. Default is openai."""
    backends_list = get_backends()
    options = [
        {"id": b, "label": "OpenAI" if b == BACKEND_OPENAI else b}
        for b in backends_list
    ]
    return jsonify({"backends": options, "default": DEFAULT_BACKEND})


@app.route("/load_backend", methods=["POST"])
def load_backend():
    """Load a backend on demand. Returns immediately if already loaded."""
    data = request.get_json() or {}
    backend = (data.get("backend") or "").strip()

    if backend not in get_backends():
        return jsonify({"success": False, "error": "Invalid backend"}), 400

    if conversation_chains.get(backend) is not None:
        return jsonify({"success": True, "message": "Already loaded"})

    try:
        chain = load_chain_for_backend(backend)
        conversation_chains[backend] = chain
        if chain is None:
            return jsonify({
                "success": False,
                "error": f"No pre-built index for '{backend}'. Build it first with: python App_p1.py",
            }), 400
        return jsonify({"success": True, "message": f"{backend} loaded"})
    except Exception as e:
        conversation_chains[backend] = None
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "error": "Missing JSON body"}), 400

    question = data.get("question", "").strip()
    backend = (data.get("backend") or DEFAULT_BACKEND).strip()

    if backend not in get_backends():
        return jsonify({"success": False, "error": "Invalid backend"}), 400

    if not question:
        return jsonify({"success": False, "error": "Question is empty"}), 400

    chain = conversation_chains.get(backend)
    if chain is None:
        return jsonify({
            "success": False,
            "error": f"Model '{backend}' not loaded. Select it from the dropdown to load.",
        }), 400

    try:
        result = chain.invoke({"question": question})
        answer = result.get("answer") or result.get("result") or ""
        return jsonify({"success": True, "answer": answer})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    init_chains()
    app.run(debug=True)
