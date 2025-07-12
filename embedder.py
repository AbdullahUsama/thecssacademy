import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# --- Configuration ---
# Define the directory where your articles JSON files are located
ARTICLES_DIRECTORY = r"article-embedder\opinion" 

# Define the output directory for embedded articles
# This will create a new folder to store the processed files
OUTPUT_DIRECTORY = os.path.join(ARTICLES_DIRECTORY, "embedded_articles")

# --- Load Sentence-BERT Model ---
print("Loading Sentence-BERT model (this may take a moment)...")
# 'all-MiniLM-L6-v2' is a good balance of speed and performance.
# For higher quality, consider 'all-mpnet-base-v2' (larger model, slower but often better).
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Sentence-BERT model loaded successfully.")

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

# --- Process Each JSON File in the Directory ---
all_embedded_articles = [] # Optional: to collect all articles into one list if needed later

# Get a list of all JSON files in the specified directory
json_files = [f for f in os.listdir(ARTICLES_DIRECTORY) if f.endswith('.json')]
json_files.sort() # Sort them to process in a consistent order (e.g., by date range)

total_files_processed = 0
total_articles_embedded = 0

for file_name in json_files:
    input_file_path = os.path.join(ARTICLES_DIRECTORY, file_name)
    
    # Create a new output file name to avoid overwriting original files
    # Appending '_embedded' before the .json extension
    base_name, ext = os.path.splitext(file_name)
    output_file_name = f"{base_name}_embedded{ext}"
    output_file_path = os.path.join(OUTPUT_DIRECTORY, output_file_name)

    print(f"\n--- Processing: {file_name} ---")

    articles_in_current_file = []
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            articles_in_current_file = json.load(f)
        print(f"Successfully loaded {len(articles_in_current_file)} articles from {file_name}")
    except FileNotFoundError:
        print(f"Error: File not found at {input_file_path}. This should not happen if os.listdir worked.")
        continue # This case is less likely with os.listdir, but good to keep.
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file_path}. Skipping.")
        continue
    except Exception as e:
        print(f"An unexpected error occurred while reading {input_file_path}: {e}. Skipping.")
        continue

    if not articles_in_current_file:
        print(f"No articles found in {file_name}. Skipping embedding.")
        continue

    # Prepare texts for embedding in a batch for the current file
    texts_to_embed = []
    valid_articles_for_embedding = [] # To store articles that have valid title/content for embedding
    for article in articles_in_current_file:
        # Ensure 'title' and 'content' exist and are strings
        title = article.get("title", "") or ""
        content = article.get("content", "") or ""
        
        # Only embed if there's substantial text
        if title.strip() or content.strip():
            texts_to_embed.append(title + ". " + content)
            valid_articles_for_embedding.append(article)
        else:
            # Optionally, you can add a placeholder or skip articles with no content
            print(f"Warning: Article with no title or content in {file_name}. Skipping embedding for this article.")
            # If you still want to keep the article but without an embedding for now:
            # article["embedding"] = None 
            # articles_in_current_file.append(article) # Make sure it's added back if not added above
    
    if not texts_to_embed:
        print(f"No valid texts found for embedding in {file_name}. Skipping embedding for this file.")
        continue

    # Generate embeddings for the batch of texts from the current file
    print(f"Generating embeddings for {len(texts_to_embed)} articles in {file_name}...")
    try:
        current_file_embeddings = model.encode(texts_to_embed, convert_to_numpy=True, show_progress_bar=True)
        print(f"Embeddings generated for {file_name}.")
    except Exception as e:
        print(f"Error generating embeddings for {file_name}: {e}. Skipping saving this file.")
        continue

    # Add embeddings to the articles that were successfully embedded
    for j, article_to_update in enumerate(valid_articles_for_embedding):
        article_to_update["embedding"] = current_file_embeddings[j].tolist() # Convert numpy array to list for JSON serialization
        total_articles_embedded += 1

    # Save the updated articles (including those that might not have been embedded if you decided to keep them)
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            json.dump(articles_in_current_file, f, indent=4) # Save the original list, now with embeddings where applicable
        print(f"Updated articles with embeddings saved to: {output_file_path}")
        total_files_processed += 1
        all_embedded_articles.extend(articles_in_current_file) # Add to the collective list (optional)
    except Exception as e:
        print(f"Error saving embedded articles to {output_file_path}: {e}")

print(f"\n--- Embedding process completed for all files in '{ARTICLES_DIRECTORY}'. ---")
print(f"Total JSON files processed: {total_files_processed}")
print(f"Total articles successfully embedded: {total_articles_embedded}")
print(f"All embedded articles are saved in the '{OUTPUT_DIRECTORY}' directory.")