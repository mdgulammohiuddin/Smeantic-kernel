import pandas as pd
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load JSON data
with open('keyword_network2.json', 'r') as f:
    data = json.load(f)

# Extract keywords and sequences
keywords = data[0]["Keywords"]
sequences = data[0]["Sequences"]

# Separate sufficient and insufficient keywords and sequences
sufficient_keywords = [kw["keyword"] for kw in keywords if kw["is_sufficient"]]
insufficient_keywords = [kw["keyword"] for kw in keywords if not kw["is_sufficient"]]
sufficient_sequences = [seq["sequence"] for seq in sequences if seq["is_sufficient"]]
insufficient_sequences = [seq["sequence"] for seq in sequences if not seq["is_sufficient"]]

# Convert sequences to regex patterns (words in order with any characters between)
sufficient_patterns = [r'\b' + r'\b.*\b'.join(seq) + r'\b' for seq in sufficient_sequences]
insufficient_patterns = [r'\b' + r'\b.*\b'.join(seq) + r'\b' for seq in insufficient_sequences]

# Function to analyze a description
def analyze_description(description):
    # Handle NaN or non-string input
    desc_str = str(description) if pd.notna(description) else ""
    desc_lower = desc_str.lower()

    # Regex matching for sequences
    suff_seq_matches = sum(1 for pattern in sufficient_patterns if re.search(pattern, desc_lower, re.IGNORECASE))
    insuff_seq_matches = sum(1 for pattern in insufficient_patterns if re.search(pattern, desc_lower, re.IGNORECASE))

    # Keyword matching (ensure keywords are strings)
    suff_kw_matches = sum(1 for kw in sufficient_keywords if isinstance(kw, str) and kw.lower() in desc_lower)
    insuff_kw_matches = sum(1 for kw in insufficient_keywords if isinstance(kw, str) and kw.lower() in desc_lower)

    # Total matches
    total_suff_matches = suff_seq_matches + suff_kw_matches
    total_insuff_matches = insuff_seq_matches + insuff_kw_matches

    # Prepare documents for NLP (convert sequences to strings)
    sufficient_docs = [kw for kw in sufficient_keywords if isinstance(kw, str)] + [' '.join(seq) for seq in sufficient_sequences]
    insufficient_docs = [kw for kw in insufficient_keywords if isinstance(kw, str)] + [' '.join(seq) for seq in insufficient_sequences]
    
    # Prepare documents for TF-IDF
    all_docs = sufficient_docs + insufficient_docs + [desc_str]
    if not desc_str.strip() or (not sufficient_docs and not insufficient_docs):
        return "not sufficient", 0.0, desc_str

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    
    # Split vectors
    desc_vector = tfidf_matrix[-1]  # Last one is the description
    suff_vectors = tfidf_matrix[:len(sufficient_docs)] if sufficient_docs else None
    insuff_vectors = tfidf_matrix[len(sufficient_docs):-1] if insufficient_docs else None

    # Calculate similarities
    suff_similarity = np.mean(cosine_similarity(desc_vector, suff_vectors)) if suff_vectors is not None else 0
    insuff_similarity = np.mean(cosine_similarity(desc_vector, insuff_vectors)) if insuff_vectors is not None else 0

    # Determine sufficiency
    base_score = suff_similarity - insuff_similarity if insuff_similarity > 0 else suff_similarity
    boost = min(total_suff_matches * 0.1, 0.5)  # Boost up to 50% based on matches
    final_score = min(base_score + boost, 1.0)
    
    is_sufficient = (final_score > 0.5) or (total_suff_matches > total_insuff_matches and final_score > 0)
    percentage = round(final_score * 100, 2)
    sufficiency_label = "sufficient" if is_sufficient else "not sufficient"
    
    return sufficiency_label, percentage, desc_str

# Load Excel file
df = pd.read_excel('data.xlsx')

# Open a log file to write results
with open('analysis_log.txt', 'w', encoding='utf-8') as log_file:
    # Write header
    log_file.write("Analysis Results\n")
    log_file.write("=" * 50 + "\n\n")
    
    # Process each row
    for index, row in df.iterrows():
        short_desc = row["Short description"]
        desc = row["Description"]
        
        # Analyze both columns
        short_suff, short_pct, short_text = analyze_description(short_desc)
        desc_suff, desc_pct, desc_text = analyze_description(desc)
        
        # Write results to file
        log_file.write(f"Row {index}:\n")
        log_file.write(f"  Short description: {short_text}\n")
        log_file.write(f"    Sufficiency: {short_suff}, Percentage: {short_pct}%\n")
        log_file.write(f"  Description: {desc_text}\n")
        log_file.write(f"    Sufficiency: {desc_suff}, Percentage: {desc_pct}%\n")
        log_file.write("-" * 50 + "\n\n")
