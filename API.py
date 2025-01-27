from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from typing import List
import pandas as pd
from transformers import pipeline, AutoModel, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (or specify your frontend URL like ["http://127.0.0.1:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Preprocessing for TF-IDF
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text.lower())
    return " ".join([word for word in tokens if word.isalnum() and word not in stop_words])

# Load transformer pipeline with a more advanced model
def load_transformer_model():
    model_name = "bert-base-uncased"  # Improved model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return pipeline("feature-extraction", model=model, tokenizer=tokenizer)

# Compute similarity with TF-IDF
def get_tfidf_similarity(user_question, faqs):
    questions = faqs["Question"].tolist()
    preprocessed_questions = [preprocess_text(q) for q in questions]
    preprocessed_user_question = preprocess_text(user_question)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(preprocessed_questions + [preprocessed_user_question])
    similarity_scores = cosine_similarity(vectors[-1], vectors[:-1]).flatten()

    best_match_idx = np.argmax(similarity_scores)
    return faqs.iloc[best_match_idx]["Answer"], similarity_scores[best_match_idx]

# Helper function to find valid columns
def find_columns(df, possible_question_names, possible_answer_names):
    question_col = None
    answer_col = None

    for col in df.columns:
        if col.lower() in [q.lower() for q in possible_question_names]:
            question_col = col
        if col.lower() in [a.lower() for a in possible_answer_names]:
            answer_col = col

    return question_col, answer_col

# Load FAQs CSV from local file
try:
    faqs = pd.read_csv("conversation.csv")

    # Define possible column names
    possible_question_names = ["question", "questions"]
    possible_answer_names = ["answer", "answers"]

    # Detect columns
    question_col, answer_col = find_columns(faqs, possible_question_names, possible_answer_names)

    if question_col and answer_col:
        faqs = faqs.rename(columns={question_col: "Question", answer_col: "Answer"})
    else:
        raise Exception("The CSV file must contain columns for questions and answers. Recognized names: 'Question', 'Questions', 'Answer', 'Answers'.")
except Exception as e:
    raise RuntimeError(f"Error loading the CSV file: {e}")

# Request/Response Models
class UserQuestion(BaseModel):
    question: str
    similarity_threshold: float = 0.2

@app.post("/answer")
async def get_answer(question: UserQuestion):
    try:
        # Calculate similarity
        answer, score = get_tfidf_similarity(question.question, faqs)

        if score > question.similarity_threshold:
            return {"status": "success", "answer": answer, "score": score}
        else:
            return {
                "status": "warning",
                "message": "No suitable answer found. Try rephrasing your question.",
                "score": score,
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the request: {e}")