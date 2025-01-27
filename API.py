from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from transformers import pipeline, AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time

# Initialize FastAPI app
app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)

# CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (for production, restrict it to your frontend domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download NLTK resources (download once, not repeatedly)
nltk.download("punkt_tab")
nltk.download("stopwords")

# Preprocessing function to clean the text
def preprocess_text(text: str) -> str:
    """Preprocess the text by tokenizing and removing stopwords."""
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return " ".join([word for word in tokens if word.isalnum() and word not in stop_words])

# Function to load transformer model (using distilbert for performance improvement)
def load_transformer_model():
    model_name = "distilbert-base-uncased"  # A smaller model than bert-base-uncased for faster responses
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer)

# Preprocess all FAQ questions once during startup for better performance
def load_faq_data():
    try:
        faqs = pd.read_csv("conversation.csv")

        # Define possible column names in your CSV
        possible_question_names = ["question", "questions"]
        possible_answer_names = ["answer", "answers"]

        # Function to find valid columns for questions and answers
        def find_columns(df: pd.DataFrame, possible_question_names: list, possible_answer_names: list) -> tuple:
            question_col, answer_col = None, None
            for col in df.columns:
                if col.lower() in [q.lower() for q in possible_question_names]:
                    question_col = col
                if col.lower() in [a.lower() for a in possible_answer_names]:
                    answer_col = col
            return question_col, answer_col

        # Detect question and answer columns
        question_col, answer_col = find_columns(faqs, possible_question_names, possible_answer_names)

        if question_col and answer_col:
            faqs = faqs.rename(columns={question_col: "Question", answer_col: "Answer"})
            return faqs
        else:
            raise ValueError("The CSV file must contain columns for questions and answers.")
    except Exception as e:
        logging.error(f"Error loading FAQ data: {e}")
        raise RuntimeError(f"Error loading the CSV file: {e}")

# Load FAQ data and preprocess questions
faqs = load_faq_data()
preprocessed_questions = [preprocess_text(q) for q in faqs["Question"].tolist()]
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(preprocessed_questions)

# Function to calculate similarity using TF-IDF
def get_tfidf_similarity(user_question: str) -> tuple:
    """Compute cosine similarity between user question and FAQ questions."""
    preprocessed_user_question = preprocess_text(user_question)
    vectorized_user_question = vectorizer.transform([preprocessed_user_question])
    similarity_scores = cosine_similarity(vectorized_user_question, vectors).flatten()
    best_match_idx = np.argmax(similarity_scores)
    return faqs.iloc[best_match_idx]["Answer"], similarity_scores[best_match_idx]

# Request/Response Model for user input
class UserQuestion(BaseModel):
    question: str
    similarity_threshold: float = 0.2  # Default threshold for similarity score

# Endpoint for getting the answer from the FAQ
@app.post("/answer")
async def get_answer(question: UserQuestion):
    """Handle user question and provide an answer from the FAQ."""
    start_time = time.time()
    logging.info(f"Received question: {question.question}")
    
    try:
        # Get similarity and answer
        answer, score = get_tfidf_similarity(question.question)
        
        elapsed_time = time.time() - start_time
        logging.info(f"Processed in {elapsed_time:.2f} seconds")

        # Return answer based on similarity threshold
        if score > question.similarity_threshold:
            return {"status": "success", "answer": answer, "score": score}
        else:
            return {"status": "warning", "message": "No suitable answer found. Try rephrasing your question.", "score": score}
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Error during request processing: {e} (Processed in {elapsed_time:.2f} seconds)")
        raise HTTPException(status_code=500, detail=f"Error processing the request: {e}")

# Health check route
@app.get("/health")
async def health_check():
    """Health check endpoint to verify the service is running."""
    return {"status": "ok", "message": "Service is healthy!"}

# Catch-all route for 404 errors
@app.get("/{full_path:path}")
async def catch_all(full_path: str):
    """Catch all invalid routes."""
    return JSONResponse(status_code=404, content={"message": "This route doesn't exist."})

# Root route
@app.get("/")
async def root():
    return {"message": "Welcome to the AI-powered FAQ service!"}
