
import logging
logging.basicConfig(level=logging.INFO)
import os
import torch
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
print("⚡ Forcing PyTorch backend only...")

import gradio as gr
from typing import List
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import fitz  # PyMuPDF
import pdfplumber
import re
print("🔍 Reached start of docsum.py", flush=True)
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/pegasus-xsum"  # summarization-optimized
CHUNK_SIZE = 800
def extract_text(file_path: str) -> str:
    """Extract clean text from PDF or TXT."""
    text = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    elif file_path.endswith(".pdf"):
        try:
            # Preferred: pdfplumber
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    txt = page.extract_text()
                    if txt:
                        text += txt + "\n"
        except Exception as e:
            print("⚠️ pdfplumber failed, falling back to PyMuPDF:", e)
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text("plain")
    else:
        return ""

    text = re.sub(r"http\S+", "", text)   # remove URLs
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def sentence_split(text: str) -> List[str]:
    """Lightweight regex-based sentence splitter (no nltk)."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(text: str, chunk_size=CHUNK_SIZE) -> List[str]:
    """Split into chunks at sentence boundaries, remove duplicates & garbage."""
    sentences = sentence_split(text)
    chunks, current, current_len = [], [], 0
    for sent in sentences:
        if current_len + len(sent) > chunk_size:
            chunks.append(" ".join(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += len(sent)
    if current:
        chunks.append(" ".join(current))
      
    seen, unique_chunks = set(), []
    for c in chunks:
        c = c.strip()
        if c and c not in seen and len(c.split()) > 5:
            unique_chunks.append(c)
            seen.add(c)

    return unique_chunks
  
class Retriever:
    def __init__(self, embed_model=EMBED_MODEL):
        self.embedder = SentenceTransformer(embed_model)
        self.index = None
        self.chunks = []

    def build(self, chunks: List[str]):
        if not chunks:
            return
        embeddings = self.embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
        faiss.normalize_L2(embeddings)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.chunks = chunks

class LocalLLM:
    def __init__(self, model_name=GEN_MODEL):
        print("Loading model:", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.pipe = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=-1)

    def generate(self, text: str, max_new_tokens=256):
        out = self.pipe(
            text, 
            max_length=max_new_tokens, 
            min_length=30, 
            do_sample=False
        )[0]["summary_text"]
        return out

class DocSummarizer:
    def __init__(self):
        self.retriever = Retriever()
        self.llm = LocalLLM()
        self.partial_summaries = []

    def prepare_doc(self, file_path: str):
        text = extract_text(file_path)
        chunks = chunk_text(text)
        self.retriever.build(chunks)
        return len(chunks)

    def summarize(self):
        chunks = self.retriever.chunks
        if not chunks:
            return "No document content available to summarize.", []

        self.partial_summaries = []
        for i, chunk in enumerate(chunks, start=1):
            print(f"📝 Summarizing chunk {i}/{len(chunks)}...", flush=True)
            summary = self.llm.generate(chunk, max_new_tokens=120)
            self.partial_summaries.append(f"Chunk {i}: {summary}")

        if not self.partial_summaries:
            return "Document contained no text to summarize.", []

        combined = " ".join(self.partial_summaries)
        print("🔁 Creating final concise summary...", flush=True)
        final_summary = self.llm.generate(combined, max_new_tokens=250)

        return final_summary, self.partial_summaries


SUMMARIZER = DocSummarizer()

def handle_upload(file):
    if file is None:
        return "No file uploaded yet."
    n_chunks = SUMMARIZER.prepare_doc(file.name)
    return f"Document processed with {n_chunks} clean chunks. Click 'Summarize'!"

def handle_summary():
    final_summary, partials = SUMMARIZER.summarize()
    return final_summary, "\n\n".join(partials)

with gr.Blocks() as demo:
    gr.Markdown("# 📄 Document Summarizer (Local, Pegasus-XSum, Cleaned)")

    with gr.Row():
        file_in = gr.File(label="Upload TXT or PDF", file_types=[".txt", ".pdf"])

    status = gr.Textbox(label="Status", interactive=False)
    summarize_btn = gr.Button("Summarize")

    with gr.Row():
        output_final = gr.Textbox(label="Final Summary", lines=10)
        output_partials = gr.Textbox(label="Intermediate Chunk Summaries", lines=20)

    file_in.change(fn=handle_upload, inputs=[file_in], outputs=[status])
    summarize_btn.click(fn=handle_summary, outputs=[output_final, output_partials])

if __name__ == "__main__":
    print("🚀 Starting Gradio app... open http://127.0.0.1:7860 in your browser", flush=True)
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, debug=True)
