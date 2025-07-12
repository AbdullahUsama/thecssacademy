### Script Descriptions and Usage

#### `json_key_printer.py`
**Purpose**
Reads a JSON file containing multiple-choice questions (MCQs) organized by section, and prints the total number of MCQs and a count of MCQs in each section.

**Usage**
* Update the `file_to_process` variable in the script with the full path to your `.json` file.
* Run the script: `python json_key_printer.py`
* The output will display the total count and section-wise breakdown of MCQs.

#### `embedder.py`
**Purpose**
Embeds the `title` and `content` of JSON-formatted articles using a Sentence-BERT model (`all-MiniLM-L6-v2`). Embeddings are added as a new `embedding` field in each article and saved to new JSON files.

**Usage**
* Place your raw JSON article files in the folder defined by `ARTICLES_DIRECTORY` (default is `article-embedder/opinion`).
* Run the script: `python embedder.py`
* The script processes all `.json` files in the directory and saves embedded versions to `article-embedder/opinion/embedded_articles/`.

#### `search.py`
**Purpose**
Performs a local semantic search on pre-embedded articles using cosine similarity.

**Usage**
* Set the `EMBEDDED_ARTICLES_DIRECTORY` variable to the path where your embedded article JSON files are located.
* Run the script: `python search.py`
* Update the query string inside the script.
* The script will output the top semantically similar articles with their similarity scores.

#### `semantic_search_api.py`
**Purpose**
Launches a FastAPI-based REST API for semantic search. Takes a query and returns the most similar articles using Sentence-BERT embeddings.

**Usage**
* Set the path to your embedded JSON files in the script.
* Run the API server with Uvicorn: `uvicorn semantic_search_api:app --reload`
* Send a POST request to `/semantic_search` with a JSON body containing your query.
* The API responds with the top `k` most relevant articles.
