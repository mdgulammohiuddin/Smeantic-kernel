import pandas as pd
import spacy
import networkx as nx
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from pathlib import Path
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])  # Medium model for word embeddings

# Configuration
# Update these paths to point to your Excel files and desired JSON output location
AUTOMATABLE_FILE = "automatable_use_cases.xlsx"  # Example: r"C:\Data\automatable_use_cases.xlsx" or "/home/user/data/automatable_use_cases.xlsx"
NON_AUTOMATABLE_FILE = "non_automatable_use_cases.xlsx"  # Example: r"C:\Data\non_automatable_use_cases.xlsx" or "/home/user/data/non_automatable_use_cases.xlsx"
OUTPUT_FILE = "keyword_network.json"  # Example: r"C:\Data\keyword_network.json" or "/home/user/data/keyword_network.json"
BATCH_SIZE = 10000  # Number of rows to process in each batch
MIN_SUBGRAPH_SIZE = 3  # Minimum keywords for sufficiency
MIN_EDGE_WEIGHT = 7  # Minimum edge weight for sufficiency
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for edges

# Step 1: Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Step 2: Process Excel file with batch processing
def process_excel_file(file_path, temp_file, batch_size=BATCH_SIZE):
    logging.info(f"Processing {file_path}")
    if not Path(file_path).exists():
        logging.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read the entire Excel file
    df = pd.read_excel(file_path, sheet_name=0)
    logging.info(f"Loaded {len(df)} rows from {file_path}")
    
    # Process in batches
    first_batch = True
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch = df.iloc[start:end]
        batch['Cleaned_Description'] = batch['Description'].apply(preprocess_text)
        mode = 'w' if first_batch else 'a'
        header = first_batch
        batch[['Cleaned_Description']].to_csv(temp_file, mode=mode, header=header, index=False)
        first_batch = False
        logging.info(f"Processed batch of {len(batch)} rows from {file_path}")
    
    return temp_file

# Step 3: Extract unique keywords using TF-IDF
def extract_unique_keywords(cleaned_texts):
    logging.info("Extracting unique keywords with TF-IDF")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    feature_names = vectorizer.get_feature_names_out()
    # Select all keywords based on sum of TF-IDF scores
    keyword_scores = tfidf_matrix.sum(axis=0).A1
    keyword_ranking = [(feature_names[i], keyword_scores[i]) for i in range(len(feature_names))]
    keyword_ranking.sort(key=lambda x: x[1], reverse=True)
    logging.info(f"Extracted {len(keyword_ranking)} unique keywords")
    return [kw for kw, _ in keyword_ranking]

# Step 4: Build keyword connection graph based on semantic similarity
def build_similarity_graph(keywords):
    G = nx.Graph()
    G.add_nodes_from(keywords)
    logging.info("Building similarity graph")
    # Batch process with spacy.pipe for efficiency
    keyword_tokens = list(nlp.pipe(keywords))
    for i, token1 in enumerate(keyword_tokens):
        for j, token2 in enumerate(keyword_tokens[i+1:], start=i+1):
            similarity = token1.similarity(token2)
            if similarity > SIMILARITY_THRESHOLD:
                G.add_edge(keywords[i], keywords[j], weight=similarity * 10)  # Scale weight
    return G

# Step 5: Refine connections iteratively
def refine_connections(graph, max_iterations=3):
    logging.info("Refining connections")
    G = graph.copy()
    for iteration in range(max_iterations):
        new_edges = []
        for node in G.nodes():
            neighbors = list(G.neighbors(node))
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if not G.has_edge(n1, n2):
                        # Estimate transitive weight
                        weight1 = G[node][n1]['weight']
                        weight2 = G[node][n2]['weight']
                        transitive_weight = (weight1 + weight2) / 2
                        if transitive_weight > MIN_EDGE_WEIGHT:
                            new_edges.append((n1, n2, {'weight': transitive_weight}))
        G.add_edges_from(new_edges)
        logging.info(f"Iteration {iteration + 1}: Added {len(new_edges)} new edges")
        if not new_edges:  # Stop if no new edges
            break
    return G

# Step 6: Determine sufficiency for each keyword
def is_keyword_sufficient(keyword, graph):
    neighbors = list(graph.neighbors(keyword))
    subgraph_nodes = [keyword] + neighbors
    subgraph = graph.subgraph(subgraph_nodes)
    strong_edges = sum(1 for _, _, data in subgraph.edges(data=True) if data['weight'] >= MIN_EDGE_WEIGHT)
    return len(subgraph_nodes) >= MIN_SUBGRAPH_SIZE and strong_edges >= MIN_SUBGRAPH_SIZE - 1

# Step 7: Get connected keywords
def get_connected_keywords(keyword, graph):
    connections = []
    for neighbor in graph.neighbors(keyword):
        weight = graph[keyword][neighbor]['weight']
        connections.append({"keyword": neighbor, "weight": weight})
    connections.sort(key=lambda x: x['weight'], reverse=True)
    return connections

# Step 8: Main processing
def main():
    # Create temporary file for cleaned texts
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
        temp_file_path = temp_file.name
    
    try:
        # Process both Excel files
        logging.info("Reading and processing Excel files")
        process_excel_file(AUTOMATABLE_FILE, temp_file_path)
        process_excel_file(NON_AUTOMATABLE_FILE, temp_file_path)
        
        # Read cleaned texts from temporary file
        logging.info("Reading cleaned texts from temporary file")
        cleaned_texts = pd.read_csv(temp_file_path)['Cleaned_Description'].tolist()
        
        # Filter out empty texts
        cleaned_texts = [text for text in cleaned_texts if text.strip()]
        if not cleaned_texts:
            logging.error("No valid descriptions found after preprocessing")
            return
        
        # Extract unique keywords
        unique_keywords = extract_unique_keywords(cleaned_texts)
        
        # Build and refine similarity graph
        similarity_graph = build_similarity_graph(unique_keywords)
        refined_graph = refine_connections(similarity_graph)
        
        # Prepare output
        logging.info("Preparing output")
        output = {
            "keywords": []
        }
        for keyword in unique_keywords:
            is_sufficient = is_keyword_sufficient(keyword, refined_graph)
            connected_keywords = get_connected_keywords(keyword, refined_graph)
            output["keywords"].append({
                "keyword": keyword,
                "connected_keywords": connected_keywords,
                "is_sufficient": is_sufficient
            })
        
        # Write to JSON
        logging.info("Writing results to JSON")
        output_dir = Path(OUTPUT_FILE).parent
        if output_dir != Path("."):
            output_dir.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output, f, indent=4)
    
    finally:
        # Clean up temporary file
        Path(temp_file_path).unlink()

if __name__ == '__main__':
    main()
    logging.info("Processing complete. Results saved to 'keyword_network.json'.")
