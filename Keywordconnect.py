import dask.dataframe as dd
import pandas as pd
import spacy
import networkx as nx
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from concurrent.futures import ProcessPoolExecutor
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])  # Medium model for word embeddings

# Configuration
AUTOMATABLE_FILE = "automatable_use_cases.xlsx"
NON_AUTOMATABLE_FILE = "non_automatable_use_cases.xlsx"
OUTPUT_FILE = "keyword_network.json"
CHUNK_SIZE = 10000  # Adjust based on memory
MAX_KEYWORDS = 1000  # Maximum number of unique keywords
MIN_SUBGRAPH_SIZE = 3  # Minimum keywords for sufficiency
MIN_EDGE_WEIGHT = 7                # Minimum edge weight for sufficiency
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for edges

# Step 1: Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)

# Step 2: Process chunk of descriptions
def process_chunk(chunk, chunk_id):
    logging.info(f"Processing chunk {chunk_id}")
    chunk['Cleaned_Description'] = chunk['Description'].apply(preprocess_text)
    return chunk[['Cleaned_Description']]

# Step 3: Extract unique keywords using TF-IDF
def extract_unique_keywords(cleaned_texts, max_keywords=MAX_KEYWORDS):
    logging.info("Extracting unique keywords with TF-IDF")
    vectorizer = TfidfVectorizer(max_features=max_keywords)
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    feature_names = vectorizer.get_feature_names_out()
    # Select top keywords based on sum of TF-IDF scores across documents
    keyword_scores = tfidf_matrix.sum(axis=0).A1
    keyword_ranking = [(feature_names[i], keyword_scores[i]) for i in range(len(feature_names))]
    keyword_ranking.sort(key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in keyword_ranking[:max_keywords]]

# Step 4: Build keyword connection graph based on semantic similarity
def build_similarity_graph(keywords):
    G = nx.Graph()
    G.add_nodes_from(keywords)
    logging.info("Building similarity graph")
    for i, kw1 in enumerate(keywords):
        token1 = nlp(kw1)
        for kw2 in keywords[i+1:]:
            token2 = nlp(kw2)
            similarity = token1.similarity(token2)
            if similarity > SIMILARITY_THRESHOLD:
                G.add_edge(kw1, kw2, weight=similarity * 10)  # Scale weight
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
                        # Estimate transitive weight (average of paths through node)
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
    # Get subgraph of keyword and its neighbors
    neighbors = list(graph.neighbors(keyword))
    subgraph_nodes = [keyword] + neighbors
    subgraph = graph.subgraph(subgraph_nodes)
    # Check if subgraph is sufficient
    strong_edges = sum(1 for _, _, data in subgraph.edges(data=True) if data['weight'] >= MIN_EDGE_WEIGHT)
    return len(subgraph_nodes) >= MIN_SUBGRAPH_SIZE and strong_edges >= MIN_SUBGRAPH_SIZE - 1

# Step 7: Get connected keywords
def get_connected_keywords(keyword, graph):
    connections = []
    for neighbor in graph.neighbors(keyword):
        weight = graph[keyword][neighbor]['weight']
        connections.append({"keyword": neighbor, "weight": weight})
    # Sort by weight descending
    connections.sort(key=lambda x: x['weight'], reverse=True)
    return connections

# Step 8: Main processing
def main():
    # Read Excel files with dask
    logging.info("Reading Excel files")
    automatable_df = dd.read_excel(AUTOMATABLE_FILE, sheet_name=0)
    non_automatable_df = dd.read_excel(NON_AUTOMATABLE_FILE, sheet_name=0)
    
    # Combine datasets
    df = dd.concat([automatable_df[['Description']], non_automatable_df[['Description']]])
    
    # Process in chunks
    logging.info("Processing descriptions in chunks")
    chunks = [chunk.compute() for chunk in df.to_delayed()]
    with ProcessPoolExecutor() as executor:
        processed_chunks = list(executor.map(process_chunk, chunks, range(len(chunks))))
    
    # Combine processed chunks
    processed_df = pd.concat(processed_chunks, ignore_index=True)
    cleaned_texts = processed_df['Cleaned_Description'].tolist()
    
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
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == '__main__':
    main()
    logging.info("Processing complete. Results saved to 'keyword_network.json'.")
