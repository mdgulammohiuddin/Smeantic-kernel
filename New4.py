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

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_md", disable=["parser", "ner"])

# Configuration
# Update these paths to your Excel files and JSON output location
AUTOMATABLE_FILE = "automatable_use_cases.xlsx"  # Example: r"C:\Data\automatable_use_cases.xlsx" or "/home/user/data/automatable_use_cases.xlsx"
NON_AUTOMATABLE_FILE = "non_automatable_use_cases.xlsx"  # Example: r"C:\Data\non_automatable_use_cases.xlsx" or "/home/user/data/non_automatable_use_cases.xlsx"
OUTPUT_FILE = "keyword_network.json"  # Example: r"C:\Data\keyword_network.json" or "/home/user/data/keyword_network.json"
BATCH_SIZE = 10000  # Rows per batch
MIN_SUBGRAPH_SIZE = 3  # Minimum keywords for sufficiency
MIN_EDGE_WEIGHT = 7  # Minimum edge weight for sufficiency
SIMILARITY_THRESHOLD = 0.8  # Increased to reduce graph density
MAX_SEQUENCE_LENGTH = 3  # Maximum length of sequences
TFIDF_SCORE_THRESHOLD = 0.1  # Minimum TF-IDF score for keywords

# Step 1: Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = text.lower()
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
    
    df = pd.read_excel(file_path, sheet_name=0)
    logging.info(f"Loaded {len(df)} rows from {file_path}")
    
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
    keyword_scores = tfidf_matrix.sum(axis=0).A1
    keyword_ranking = [(feature_names[i], keyword_scores[i]) for i in range(len(feature_names))]
    keyword_ranking.sort(key=lambda x: x[1], reverse=True)
    # Filter keywords by TF-IDF score
    keyword_ranking = [(kw, score) for kw, score in keyword_ranking if score >= TFIDF_SCORE_THRESHOLD]
    logging.info(f"Extracted {len(keyword_ranking)} unique keywords after filtering")
    return [kw for kw, _ in keyword_ranking]

# Step 4: Build keyword connection graph
def build_similarity_graph(keywords):
    G = nx.Graph()
    G.add_nodes_from(keywords)
    logging.info("Building similarity graph")
    keyword_tokens = list(nlp.pipe(keywords))
    for i, token1 in enumerate(keyword_tokens):
        for j, token2 in enumerate(keyword_tokens[i+1:], start=i+1):
            similarity = token1.similarity(token2)
            if similarity > SIMILARITY_THRESHOLD:
                G.add_edge(keywords[i], keywords[j], weight=similarity * 10)
    return G

# Step 5: Refine connections iteratively
def refine_connections(graph, max_iterations=2):  # Reduced to 2 iterations
    logging.info("Refining connections")
    G = graph.copy()
    for iteration in range(max_iterations):
        new_edges = []
        total_nodes = len(G.nodes())
        for idx, node in enumerate(G.nodes(), 1):
            neighbors = list(G.neighbors(node))
            for i, n1 in enumerate(neighbors):
                for n2 in neighbors[i+1:]:
                    if not G.has_edge(n1, n2):
                        weight1 = G[node][n1]['weight']
                        weight2 = G[node][n2]['weight']
                        transitive_weight = (weight1 + weight2) / 2
                        if transitive_weight > MIN_EDGE_WEIGHT:
                            new_edges.append((n1, n2, {'weight': transitive_weight}))
            # Log progress every 100 nodes
            if idx % 100 == 0:
                logging.info(f"Iteration {iteration + 1}: Processed {idx}/{total_nodes} nodes")
        G.add_edges_from(new_edges)
        logging.info(f"Iteration {iteration + 1}: Added {len(new_edges)} new edges")
        if not new_edges:
            break
    return G

# Step 6: Determine sufficiency
def is_sufficient(nodes, graph):
    subgraph = graph.subgraph(nodes)
    strong_edges = sum(1 for _, _, data in subgraph.edges(data=True) if data['weight'] >= MIN_EDGE_WEIGHT)
    return len(subgraph.nodes) >= MIN_SUBGRAPH_SIZE and strong_edges >= MIN_SUBGRAPH_SIZE - 1

# Step 7: Get connected keywords
def get_connected_keywords(keyword, graph):
    connections = []
    for neighbor in graph.neighbors(keyword):
        weight = graph[keyword][neighbor]['weight']
        connections.append({"keyword": neighbor, "weight": weight})
    connections.sort(key=lambda x: x['weight'], reverse=True)
    return connections

# Step 8: Find sequences (paths in the graph)
def find_sequences(graph, max_length=MAX_SEQUENCE_LENGTH):
    sequences = []
    for node in graph.nodes():
        for length in range(2, max_length + 1):
            for target in graph.nodes():
                if node != target:
                    paths = list(nx.all_simple_paths(graph, node, target, cutoff=length))
                    for path in paths:
                        if len(path) >= 2:
                            sequences.append(path)
    return sequences

# Step 9: Get nested sequences
def get_nested_sequences(sequence):
    nested = []
    for i in range(2, len(sequence) + 1):
        for j in range(len(sequence) - i + 1):
            sub_sequence = sequence[j:j+i]
            if len(sub_sequence) >= 2:
                nested.append(sub_sequence)
    return nested

# Step 10: Main processing
def main():
    # Create temporary files for each category
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_auto, \
         tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_nonauto:
        temp_auto_path = temp_auto.name
        temp_nonauto_path = temp_nonauto.name
    
    try:
        # Process Excel files
        logging.info("Reading and processing Excel files")
        process_excel_file(AUTOMATABLE_FILE, temp_auto_path)
        process_excel_file(NON_AUTOMATABLE_FILE, temp_nonauto_path)
        
        # Read cleaned texts
        logging.info("Reading cleaned texts")
        auto_texts = pd.read_csv(temp_auto_path)['Cleaned_Description'].tolist()
        nonauto_texts = pd.read_csv(temp_nonauto_path)['Cleaned_Description'].tolist()
        
        # Filter empty texts, handling non-string values
        auto_texts = [text for text in auto_texts if isinstance(text, str) and text.strip()]
        nonauto_texts = [text for text in nonauto_texts if isinstance(text, str) and text.strip()]
        
        if not auto_texts and not nonauto_texts:
            logging.error("No valid descriptions found")
            return
        
        # Extract keywords
        logging.info("Extracting keywords")
        auto_keywords = extract_unique_keywords(auto_texts) if auto_texts else []
        nonauto_keywords = extract_unique_keywords(nonauto_texts) if nonauto_texts else []
        
        # Build and refine graphs
        auto_graph = build_similarity_graph(auto_keywords) if auto_keywords else nx.Graph()
        nonauto_graph = build_similarity_graph(nonauto_keywords) if nonauto_keywords else nx.Graph()
        auto_graph = refine_connections(auto_graph)
        nonauto_graph = refine_connections(nonauto_graph)
        
        # Prepare output
        output = []
        
        # Automatable category
        if auto_keywords:
            auto_output = {
                "Category": "automatable",
                "Keywords": [],
                "Sequences": []
            }
            # Keywords
            for keyword in auto_keywords:
                nested_keywords = get_connected_keywords(keyword, auto_graph)
                is_suff = is_sufficient([keyword] + [nk["keyword"] for nk in nested_keywords], auto_graph)
                auto_output["Keywords"].append({
                    "keyword": keyword,
                    "nested_keywords": nested_keywords,
                    "is_sufficient": is_suff
                })
            # Sequences
            sequences = find_sequences(auto_graph)
            for seq in sequences:
                nested_seqs = get_nested_sequences(seq)
                is_suff = is_sufficient(seq, auto_graph)
                auto_output["Sequences"].append({
                    "sequence": seq,
                    "nested_sequences": nested_seqs,
                    "is_sufficient": is_suff
                })
            output.append(auto_output)
        
        # Non-automatable category
        if nonauto_keywords:
            nonauto_output = {
                "Category": "non_automatable",
                "Keywords": [],
                "Sequences": []
            }
            # Keywords
            for keyword in nonauto_keywords:
                nested_keywords = get_connected_keywords(keyword, nonauto_graph)
                is_suff = is_sufficient([keyword] + [nk["keyword"] for nk in nested_keywords], nonauto_graph)
                nonauto_output["Keywords"].append({
                    "keyword": keyword,
                    "nested_keywords": nested_keywords,
                    "is_sufficient": is_suff
                })
            # Sequences
            sequences = find_sequences(nonauto_graph)
            for seq in sequences:
                nested_seqs = get_nested_sequences(seq)
                is_suff = is_sufficient(seq, nonauto_graph)
                nonauto_output["Sequences"].append({
                    "sequence": seq,
                    "nested_sequences": nested_seqs,
                    "is_sufficient": is_suff
                })
            output.append(nonauto_output)
        
        # Write to JSON
        logging.info("Writing results to JSON")
        output_dir = Path(OUTPUT_FILE).parent
        if output_dir != Path("."):
            output_dir.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(output, f, indent=4)
    
    finally:
        # Clean up temporary files
        Path(temp_auto_path).unlink()
        Path(temp_nonauto_path).unlink()

if __name__ == '__main__':
    main()
    logging.info("Processing complete. Results saved to 'keyword_network.json'.")
