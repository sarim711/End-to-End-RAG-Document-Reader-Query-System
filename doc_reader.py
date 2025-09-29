import gradio as gr
import PyPDF2
import google.generativeai as genai
import os
from dotenv import load_dotenv
from rouge_score import rouge_scorer
import numpy as np
import faiss
import docx

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    model = None

def chunk_text(text, chunk_size=1000):
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_text(file):
    """Extract text from PDF or Word file."""
    file_extension = os.path.splitext(file.name)[1].lower()
    text = ""
    try:
        if file_extension == '.pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if not text.strip():
                text = "No text could be extracted using PyPDF2. OCR fallback not available without pdf2image."
        elif file_extension in ['.doc', '.docx']:
            if file_extension == '.docx':
                doc = docx.Document(file)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            if not text.strip():
                text = "No text could be extracted from Word file."
        else:
            text = f"Unsupported file format: {file_extension}"
        print(f"Debug: Extracted text length (words): {len(text.split())}, Sample: {text[:200]}...")  # Debug extraction
    except Exception as e:
        text = f"Error extracting text: {str(e)}"
    return text

def chunk_text(text, chunk_size=1000):
    """Split text into chunks of approximately chunk_size words."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def get_gemini_embedding(text):
    if not GEMINI_API_KEY:
        return np.zeros(768)
    try:
        result = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text,
            task_type="semantic_similarity"
        )
        embed = np.array(result['embedding'])
        if np.all(embed == 0):
            print("Warning: Embedding is all zeros.")
        return embed
    except Exception as e:
        print(f"Embedding error: {str(e)}")
        return np.zeros(768)

def calculate_cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings."""
    dot_product = np.dot(emb1, emb2)
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    return dot_product / (norm_emb1 * norm_emb2) if norm_emb1 * norm_emb2 != 0 else 0.0

def build_faiss_index(chunks):
    """Build FAISS index from document chunks."""
    embeddings = [get_gemini_embedding(chunk) for chunk in chunks]
    embeddings = np.array(embeddings).astype('float32')
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return index, chunks

def retrieve_top_chunks(question, index, chunks, k=3):
    """Retrieve top-k most relevant chunks from FAISS index."""
    q_emb = get_gemini_embedding(question).astype('float32').reshape(1, -1)
    faiss.normalize_L2(q_emb)
    
    scores, indices = index.search(q_emb, k)
    top_chunks = [chunks[i] for i in indices[0]]
    
    print(f"Debug: Top chunk indices: {indices[0]}, Scores: {scores[0]}")
    return top_chunks


def call_llm(question, index, chunks, k=3):
    if not model:
        return "Error: GEMINI_API_KEY environment variable not set."
    try:
        top_chunks = retrieve_top_chunks(question, index, chunks, k)
        context = ' '.join(top_chunks)
        
        prompt = f"Context: {context}\n\nUsing *only* the provided context, answer: {question} Answer:"
        generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=1024
        )
        response = model.generate_content(prompt, generation_config=generation_config)
        
        if response and response.candidates and response.candidates[0].content.parts:
            return response.text.strip() or "No meaningful response generated."
        return "No response generated."
    except Exception as e:
        return f"Error calling Gemini API: {str(e)}"


def calculate_scores(reference_answer, gemini_response):
    reference_embed = get_gemini_embedding(reference_answer)
    gemini_embed = get_gemini_embedding(gemini_response)
    
    if np.linalg.norm(reference_embed) == 0 or np.linalg.norm(gemini_embed) == 0:
        print("Warning: Zero vector detected in embeddings.")
        return {"cosine_similarity": 0.0, "rouge1_fmeasure": 0.0, 
                "rouge2_fmeasure": 0.0, "rougeL_fmeasure": 0.0}
    
    reference_embed = reference_embed / np.linalg.norm(reference_embed)
    gemini_embed = gemini_embed / np.linalg.norm(gemini_embed)
    
    cosine_sim = calculate_cosine_similarity(reference_embed, gemini_embed)
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(reference_answer, gemini_response)
    
    return {
        "cosine_similarity": float(cosine_sim),
        "rouge1_fmeasure": rouge_scores['rouge1'].fmeasure,
        "rouge2_fmeasure": rouge_scores['rouge2'].fmeasure,
        "rougeL_fmeasure": rouge_scores['rougeL'].fmeasure
    }


def process_and_answer(file, eval_question, reference_answer):
    if not file:
        return "Please upload a file.", "", "", "", {"cosine_similarity": 0.0, "rouge1_fmeasure": 0.0,
                                                      "rouge2_fmeasure": 0.0, "rougeL_fmeasure": 0.0}
    
    text = extract_text(file)
    chunks = chunk_text(text, chunk_size=1000)
    
    # Build FAISS index once
    index, chunks = build_faiss_index(chunks)
    
    if not chunks or not eval_question:
        return text, eval_question, reference_answer, "No question provided.", {"cosine_similarity": 0.0,
                                                                                "rouge1_fmeasure": 0.0,
                                                                                "rouge2_fmeasure": 0.0,
                                                                                "rougeL_fmeasure": 0.0}
    
    gemini_response = call_llm(eval_question, index, chunks)
    
    scores = {"cosine_similarity": 0.0, "rouge1_fmeasure": 0.0, 
              "rouge2_fmeasure": 0.0, "rougeL_fmeasure": 0.0}
    if reference_answer and reference_answer.strip():
        scores = calculate_scores(reference_answer, gemini_response)
    
    return text, eval_question, reference_answer, gemini_response, scores

demo = gr.Interface(
    fn=process_and_answer,
    inputs=[
        gr.File(label="Upload File", file_types=[".pdf", ".doc", ".docx"]),
        gr.Textbox(label="Evaluation Question", placeholder="Enter the evaluation question"),
        gr.Textbox(label="Reference Answer", placeholder="Enter the reference answer")
    ],
    outputs=[
        gr.Textbox(label="Extracted Text", lines=20, max_lines=None),
        gr.Textbox(label="Evaluation Question", lines=5, max_lines=None),
        gr.Textbox(label="Reference Answer", lines=10, max_lines=None),
        gr.Textbox(label="Gemini Response", lines=10, max_lines=None),
        gr.JSON(label="Similarity Scores", value={"cosine_similarity": 0.0, 
                                                  "rouge1_fmeasure": 0.0,
                                                  "rouge2_fmeasure": 0.0,
                                                  "rougeL_fmeasure": 0.0})
    ],
    title="Document Reader with Evaluation (FAISS-enabled)",
    description="Upload a PDF or Word file, input evaluation question and reference answer (optional), "
                "generate Gemini response using top FAISS-retrieved chunks, and see similarity scores."
)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862, share=False)
