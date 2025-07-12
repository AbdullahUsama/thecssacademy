import os
import json
import numpy as np
from fastapi import FastAPI, Query
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

EMBEDDED_ARTICLES_DIRECTORY = r"C:\Users\abdul\Desktop\csspreparation\article-embedder\scraped_articles_tribune\embedded_articles"
START_PAGE = 1
END_PAGE = 50

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  #testing k liye 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2') # musfir u can try diff embedding models for speed, but this is a good balance betwen speed and good quality embeddings
print("Model loaded.")

def load_all_embedded_articles(directory, start_page, end_page):
    all_articles = []
    for i in range(start_page, end_page + 1):
        file_name = f"articles_page_{i}_embedded.json"
        file_path = os.path.join(directory, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                articles_in_file = json.load(f)
            for article in articles_in_file:
                if "embedding" in article and isinstance(article["embedding"], list):
                    article["embedding"] = np.array(article["embedding"])
                    all_articles.append(article)
    return all_articles

articles_with_embeddings = load_all_embedded_articles(EMBEDDED_ARTICLES_DIRECTORY, START_PAGE, END_PAGE)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10 #ye change krlin depending on the app scenario

@app.post("/semantic_search")
def semantic_search_api(request: SearchRequest):
    query = request.query
    top_k = request.top_k
    if not articles_with_embeddings:
        return []
    query_embedding = model.encode(query, convert_to_numpy=True)
    valid_articles = [art for art in articles_with_embeddings if "embedding" in art and isinstance(art["embedding"], np.ndarray)]
    article_embeddings_array = np.array([article["embedding"] for article in valid_articles])
    similarities = cosine_similarity(query_embedding.reshape(1, -1), article_embeddings_array)[0]
    article_scores = []
    for i, article in enumerate(valid_articles):
        article_scores.append({
            "article": article,
            "similarity_score": float(similarities[i])
        })
    article_scores.sort(key=lambda x: x["similarity_score"], reverse=True)
    results = []
    for item in article_scores[:top_k]:
        article_copy = item["article"].copy()
        if "embedding" in article_copy:
            del article_copy["embedding"]
        article_copy["similarity_score"] = item["similarity_score"]
        results.append(article_copy)
    return results