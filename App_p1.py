import os

from dotenv import load_dotenv

load_dotenv()

from dataclasses import dataclass
from typing import List, Any

from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

# Backends: add new keys here and in get_embeddings / get_llm / get_faiss_dir
BACKEND_OPENAI = "openai"
BACKEND_LLAMA = "llama"
BACKENDS = (BACKEND_OPENAI, BACKEND_LLAMA)

PDF_DIR = os.environ.get("ADS_PDF_DIR", os.path.join("data", "pdfs"))
FAISS_INDEX_DIR_OPENAI = os.environ.get("ADS_FAISS_INDEX_DIR", "faiss_index")
FAISS_INDEX_DIR_LLAMA = os.environ.get("ADS_FAISS_INDEX_DIR_LLAMA", "faiss_index_llama")
LLAMA_MODEL_PATH = os.environ.get("ADS_LLAMA_MODEL_PATH", os.path.join("models", "llama-7b.ggmlv3.q4_1.bin"))
EMBEDDING_MODEL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class PageRecord:
    doc_id: str
    page: int
    source: str
    text: str


def load_pdfs(pdf_dir: str = PDF_DIR) -> List[PageRecord]:
    """Read all PDFs in pdf_dir and return a list of page records."""
    records: List[PageRecord] = []
    if not os.path.isdir(pdf_dir):
        raise FileNotFoundError(f"PDF directory '{pdf_dir}' does not exist")

    for name in os.listdir(pdf_dir):
        if not name.lower().endswith(".pdf"):
            continue
        path = os.path.join(pdf_dir, name)
        reader = PdfReader(path)
        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            records.append(
                PageRecord(
                    doc_id=name,
                    page=page_index + 1,
                    source=path,
                    text=text,
                )
            )
    return records


def make_chunks(
    pages: List[PageRecord],
    chunk_size: int = 500,
    chunk_overlap: int = 80,
) -> List[Document]:
    """Split page text into chunks of ~chunk_size chars with overlap."""
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks: List[Document] = []
    for record in pages:
        for piece in splitter.split_text(record.text):
            if not piece.strip():
                continue
            chunks.append(
                Document(
                    page_content=piece,
                    metadata={
                        "doc_id": record.doc_id,
                        "page": record.page,
                        "source": record.source,
                    },
                )
            )
    return chunks


def get_embeddings(backend: str) -> Any:
    """Return the embedding model for the given backend."""
    if backend == BACKEND_OPENAI:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    if backend == BACKEND_LLAMA:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_MINILM)
    raise ValueError(f"Unknown backend: {backend}")


def get_faiss_index_dir(backend: str) -> str:
    """Return the FAISS index directory for the given backend."""
    if backend == BACKEND_OPENAI:
        return FAISS_INDEX_DIR_OPENAI
    if backend == BACKEND_LLAMA:
        return FAISS_INDEX_DIR_LLAMA
    raise ValueError(f"Unknown backend: {backend}")


def get_llm(backend: str) -> Any:
    """Return the LLM for the given backend."""
    if backend == BACKEND_OPENAI:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI()
    if backend == BACKEND_LLAMA:
        from langchain_community.llms import LlamaCpp
        path = os.path.abspath(LLAMA_MODEL_PATH)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Llama model not found at {path}. "
                "Download a GGML model and set ADS_LLAMA_MODEL_PATH or place it in models/."
            )
        return LlamaCpp(
            model_path=path,
            n_ctx=4096,
            n_batch=256,
            max_tokens=512,
            verbose=False,
        )
    raise ValueError(f"Unknown backend: {backend}")


def build_vector_store(chunks: List[Document], embeddings: Any, index_dir: str) -> FAISS:
    """Build FAISS index from chunks and save to index_dir."""
    store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    store.save_local(index_dir)
    return store


def load_vector_store(embeddings: Any, index_dir: str) -> FAISS:
    """Load existing FAISS index from index_dir."""
    store = FAISS.load_local(
        index_dir,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return store


QA_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the following excerpts from the textbook and course materials (Advanced Digital Systems / ADS cookbook). The context below is the only source you should use.

Rules:
- Answer the question using only the provided context.
- If the answer is not in the context, say "I couldn't find this in the provided materials"
- Do not use outside knowledge or refer to "the book" vaguely; the context below IS the book content.

Context from the materials:
{context}

Question: {question}

Answer based only on the context above:"""


def create_conversation_chain(store: FAISS, llm: Any) -> ConversationalRetrievalChain:
    """Build retrieval chain with the given store and LLM."""
    qa_prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    # Use a small k to keep the combined prompt within the model context window.
    retriever = store.as_retriever(search_kwargs={"k": 2})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return chain


def get_backend() -> str:
    """Resolve backend from env or prompt (openai | llama)."""
    env_backend = os.environ.get("ADS_LLM_BACKEND", "").strip().lower()
    if env_backend in BACKENDS:
        return env_backend
    prompt = f"Choose backend: 1) OpenAI  2) Llama (local) [1/2, default 1]: "
    choice = input(prompt).strip() or "1"
    return BACKEND_LLAMA if choice == "2" else BACKEND_OPENAI


def main() -> None:
    backend = get_backend()
    index_dir = get_faiss_index_dir(backend)
    embeddings = get_embeddings(backend)

    if os.path.isdir(index_dir):
        store = load_vector_store(embeddings, index_dir)
        print(f"Loaded existing FAISS index for {backend}.")
    else:
        records = load_pdfs()
        print(f"Loaded {len(records)} pages from PDFs in '{PDF_DIR}'.")
        chunks = make_chunks(records)
        print(f"Created {len(chunks)} text chunks.")
        store = build_vector_store(chunks, embeddings, index_dir)
        print(f"Built and saved FAISS index for {backend}.")

    llm = get_llm(backend)
    chain = create_conversation_chain(store, llm)
    print(f"Using backend: {backend}. Type 'exit' to quit.")
    while True:
        question = input("Ask a question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        result = chain.invoke({"question": question})
        answer = result.get("answer") or result.get("result") or ""
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()
