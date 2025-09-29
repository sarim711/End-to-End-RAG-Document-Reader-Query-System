
# End-to-End RAG Document Reader and Query System

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Framework](https://img.shields.io/badge/Framework-Gradio-orange)
![Vector Search](https://img.shields.io/badge/FAISS-Enabled-purple)


## 📖 Overview

This Python-based application extracts text from PDFs and Word files, processes it using **Gemini embeddings** and **FAISS indexing** for efficient retrieval, and generates accurate answers with the **Gemini-2.5-flash model**.

The system is evaluated with **ROUGE** and **cosine similarity**, achieving:

* Mean **Cosine Similarity**: **0.95**
* Mean **ROUGE-1 (Unigram F1)**: **0.75**

Deployed with **Gradio** for an interactive document querying interface.


## ✨ Features

* **Text Extraction**: Supports PDFs (`PyPDF2`) and Word files (`.doc`, `.docx`).
* **RAG Pipeline**: Splits text into chunks, generates Gemini embeddings, and retrieves top-k chunks via FAISS.
* **Question Answering**: Uses **Gemini-2.5-flash** for context-aware responses.
* **Evaluation**: Performance measured with **ROUGE** and **cosine similarity** metrics.
* **Interface**: Simple **Gradio UI** for uploading files and querying.


## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/sarim711/End-to-End-RAG-Document-Reader-Query-System.git
cd End-to-End-RAG-Document-Reader-Query-System
````

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up environment variables:

1. Create a `.env` file in the root directory.
2. Add your Gemini API key:

```text
GEMINI_API_KEY=your-api-key-here
```

## ▶️ Usage

Run the Gradio interface:

```bash
python pdf_reader.py
```

Access the app at: **[http://127.0.0.1:7862](http://127.0.0.1:7862)**

Steps:

1. Upload a PDF or Word file.
2. Enter a question (and optional reference answer if evaluation is being done).
3. View the **Gemini-generated response** along with **similarity scores**.


## 📊 Evaluation

The system was evaluated on **3 PDFs × 3 questions each (9 queries total)**.

| Metric               | Mean Score |
| -------------------- | ---------- |
| Cosine Similarity    | **0.95**   |
| ROUGE-1 (Unigram F1) | **0.75**   |
| ROUGE-2 (Bigram F1)  | 0.65       |
| ROUGE-L (Seq F1)     | 0.70       |

**Example Query**

* **Question**: What is the standard size and shape of a standard soccer ball?

* **Reference Answer from the document**: A soccer ball must be spherical in shape and made of leather or another comparable medium. Its circumference must be in the range of 27 to 28 inches.

* **Gemini Response**: According to Law 2: The Ball, a standard soccer ball must be:

  * Shape: Spherical
  * Size: Have a circumference in the range of 27 to 28 inches.

* **Scores**: Cosine = 0.97, ROUGE-1 = 0.59, ROUGE-2 = 0.38, ROUGE-L = 0.55


## 📂 Repository Structure

```text
End-to-End-RAG-Document-Reader-Query-System/
├── pdf_reader.py          # Core RAG pipeline + Gradio app
├── requirements.txt       # Python dependencies
├── .gitignore             # Ignore venv, .env, evaluation artifacts
├── LICENSE                # MIT License
└── README.md              # Project documentation
```

## 📦 Dependencies

* Python 3.8+
* [Gradio](https://www.gradio.app/)
* [PyPDF2](https://pypi.org/project/PyPDF2/)
* [Google Generative AI](https://ai.google.dev/)
* [FAISS](https://github.com/facebookresearch/faiss)
* [python-docx](https://pypi.org/project/python-docx/)
* [ROUGE-score](https://pypi.org/project/rouge-score/)
* [NumPy](https://numpy.org/)

See `requirements.txt` for pinned versions.

## ⚠️ Notes & Limitations

* Requires a valid **Gemini API key** in `.env`.
* Currently supports **PDFs and Word files** only.
* FAISS index is built **in-memory** (not persisted). For large-scale or persistent storage, consider integrating a vector database such as **Chroma**, **Milvus**, or **Weaviate**.
* Evaluation was conducted on a **small dataset**. Future work: expand to larger, domain-specific datasets.

## 📜 License

This project is licensed under the [MIT License](LICENSE).
