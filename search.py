import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --- Configuration ---
# Define the directory where your embedded articles JSON files are located
# This should be the 'embedded_articles' folder created by the previous script
# EMBEDDED_ARTICLES_DIRECTORY = r"C:\Users\abdul\Desktop\csspreparation\article-embedder\scraped_articles_tribune\embedded_articles"
# EMBEDDED_ARTICLES_DIRECTORY = r"article-embedder\articles\embedded_articles"
EMBEDDED_ARTICLES_DIRECTORY = r"article-embedder\opinion\embedded_articles"

# The START_PAGE and END_PAGE are no longer strictly needed for file iteration,
# but can be kept if you still want to define a "logical" range or for other uses.
# For file loading, we will now scan the directory directly.
# START_PAGE = 1
# END_PAGE = 50 # Adjust this if you have more or fewer pages

# --- Load Sentence-BERT Model ---
print("Loading Sentence-BERT model (this may take a moment)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence-BERT model loaded successfully.")

# --- Function to load all embedded articles ---
def load_all_embedded_articles(directory): # Removed start_page, end_page parameters
    """
    Loads articles with embeddings from all JSON files in the specified directory.

    Args:
        directory (str): The path to the directory containing embedded JSON files.

    Returns:
        list: A list of all articles loaded from the files, with embeddings as NumPy arrays.
    """
    all_articles = []
    # Iterate over all files in the specified directory
    for file_name in os.listdir(directory):
        if file_name.endswith(".json"): # Process only JSON files
            file_path = os.path.join(directory, file_name)

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    articles_in_file = json.load(f)
                
                # Ensure articles_in_file is a list, as expected
                if not isinstance(articles_in_file, list):
                    print(f"Warning: {file_name} does not contain a list of articles. Skipping.")
                    continue

                # Convert embeddings back to NumPy arrays for efficient calculation
                for article in articles_in_file:
                    if "embedding" in article and isinstance(article["embedding"], list):
                        article["embedding"] = np.array(article["embedding"])
                        all_articles.append(article)
                    else:
                        # print(f"Warning: Article in {file_name} missing or invalid embedding. Skipping article.")
                        continue # Skip this article if embedding is bad
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {file_path}. Skipping.")
            except Exception as e:
                print(f"An unexpected error occurred while reading {file_path}: {e}. Skipping.")
    print(f"Loaded {len(all_articles)} embedded articles in total.")
    return all_articles

# --- Load all embedded articles initially ---
# Now call load_all_embedded_articles only with the directory
articles_with_embeddings = load_all_embedded_articles(EMBEDDED_ARTICLES_DIRECTORY)


# --- Semantic Search Function (unchanged as it depends on the loaded data) ---
def semantic_search(query, articles, model, top_k=10):
    """
    Performs semantic search on articles based on a query.

    Args:
        query (str): The user's search query.
        articles (list): A list of article dictionaries, each with an 'embedding' key (NumPy array).
        model (SentenceTransformer): The SBERT model used for embedding.
        top_k (int): The number of top relevant articles to return.

    Returns:
        list: A list of top_k relevant articles, sorted by similarity,
              with a 'similarity_score' added.
    """
    if not articles:
        return []

    # 1. Embed the query
    query_embedding = model.encode(query, convert_to_numpy=True)

    # Extract all article embeddings into a single NumPy array for batch processing
    # Filter out articles that might not have a valid embedding
    valid_articles = [art for art in articles if "embedding" in art and isinstance(art["embedding"], np.ndarray)]
    if not valid_articles:
        print("No valid article embeddings found to search against.")
        return []

    article_embeddings_array = np.array([article["embedding"] for article in valid_articles])

    # 2. Calculate Cosine Similarity
    # Reshape query_embedding to be 2D for cosine_similarity
    similarities = cosine_similarity(query_embedding.reshape(1, -1), article_embeddings_array)[0]

    # Store similarity with original article and sort
    article_scores = []
    for i, article in enumerate(valid_articles): # Use valid_articles for indexing
        article_scores.append({
            "article": article,
            "similarity_score": similarities[i]
        })

    # 3. Rank Results
    article_scores.sort(key=lambda x: x["similarity_score"], reverse=True)

    # 4. Return Top Results
    results = []
    for item in article_scores[:top_k]:
        article_copy = item["article"].copy()
        if "embedding" in article_copy:
            del article_copy["embedding"] # Remove embedding for cleaner display
        article_copy["similarity_score"] = item["similarity_score"]
        results.append(article_copy)

    return results

# --- Example Usage ---
# You can now call semantic_search with your query, and it will search across all loaded articles.

search_query_1 = "Economic State of Pakistan"
results_1 = semantic_search(search_query_1, articles_with_embeddings, model, top_k=50)

print(f"\n--- Search results for: '{search_query_1}' (Top 100) ---")
if results_1:
    for i, result in enumerate(results_1[:50]): # Print top 10 for example
        print(f"Rank {i+1}")
        print(f"Title: {result.get('title', 'N/A')}")
        print(f"Similarity: {result['similarity_score']:.4f}")
        print(f"URL: {result.get('url', 'N/A')}")
        print("-" * 30)
else:
    print("No relevant articles found.")
