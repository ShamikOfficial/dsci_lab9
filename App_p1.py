import os

from dotenv import load_dotenv

load_dotenv()

from dataclasses import dataclass
from typing import List

from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain


PDF_DIR = os.environ.get("ADS_PDF_DIR", os.path.join("data", "pdfs"))
FAISS_INDEX_DIR = os.environ.get("ADS_FAISS_INDEX_DIR", "faiss_index")


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


def build_openai_vector_store(chunks: List[Document]) -> FAISS:
    """Build FAISS index from chunks and save to FAISS_INDEX_DIR."""
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(documents=chunks, embedding=embeddings)
    store.save_local(FAISS_INDEX_DIR)
    return store


def load_openai_vector_store() -> FAISS:
    """Load existing FAISS index from FAISS_INDEX_DIR."""
    embeddings = OpenAIEmbeddings()
    store = FAISS.load_local(
        FAISS_INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return store


QA_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based ONLY on the following excerpts from the textbook and course materials (Advanced Digital Systems / ADS cookbook). The context below is the only source you should use.

Rules:
- Answer the question using only the provided context.
- If the answer is not in the context, say "I couldn't find this in the provided materials"
- Do not use outside knowledge or refer to "the book" vaguel, the context below IS the book content.

Context from the materials:
{context}

Question: {question}

Answer based only on the context above:"""


def create_openai_conversation_chain(store: FAISS) -> ConversationalRetrievalChain:
    """Build retrieval chain with OpenAI LLM and chat memory."""
    qa_prompt = PromptTemplate(
        template=QA_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )
    retriever = store.as_retriever(search_kwargs={"k": 4})
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )
    return chain


def main() -> None:
    if os.path.isdir(FAISS_INDEX_DIR):
        store = load_openai_vector_store()
        print("Loaded existing FAISS index.")
    else:
        records = load_pdfs()
        print(f"Loaded {len(records)} pages from PDFs in '{PDF_DIR}'.")
        chunks = make_chunks(records)
        print(f"Created {len(chunks)} text chunks.")
        store = build_openai_vector_store(chunks)
        print("Built and saved FAISS index.")

    chain = create_openai_conversation_chain(store)
    print("Type 'exit' to quit.")
    while True:
        question = input("Ask a question (or 'exit'): ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        result = chain({"question": question})
        answer = result.get("answer") or result.get("result") or ""
        print(f"\nAnswer:\n{answer}\n")


if __name__ == "__main__":
    main()

