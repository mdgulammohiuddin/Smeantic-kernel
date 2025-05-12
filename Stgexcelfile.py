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

# Convert sequences to regex patterns
sufficient_patterns = [r'\b' + r'\b.*\b'.join(seq) + r'\b' for seq in sufficient_sequences]
insufficient_patterns = [r'\b' + r'\b.*\b'.join(seq) + r'\b' for seq in insufficient_sequences]

# Function to analyze a description
def analyze_description(description):
    # Handle NaN or non-string input
    desc_str = str(description) if pd.notna(description) else ""
    desc_lower = desc_str.lower()

    # Regex matching
    suff_seq_matches = sum(1 for pattern in sufficient_patterns if re.search(pattern, desc_lower, re.IGNORECASE))
    insuff_seq_matches = sum(1 for pattern in insufficient_patterns if re.search(pattern, desc_lower, re.IGNORECASE))
    suff_kw_matches = sum(1 for kw in sufficient_keywords if isinstance(kw, str) and kw.lower() in desc_lower)
    insuff_kw_matches = sum(1 for kw in insufficient_keywords if isinstance(kw, str) and kw.lower() in desc_lower)
    total_suff_matches = suff_seq_matches + suff_kw_matches
    total_insuff_matches = insuff_seq_matches + insuff_kw_matches

    # Enhance description with regex matches
    enhanced_desc = desc_str
    for kw in sufficient_keywords:
        if isinstance(kw, str) and kw.lower() in desc_lower:
            enhanced_desc += " " + kw
    for seq in sufficient_sequences:
        if re.search(r'\b' + r'\b.*\b'.join(seq) + r'\b', desc_lower, re.IGNORECASE):
            enhanced_desc += " " + " ".join(seq)

    # Prepare documents for NLP
    sufficient_docs = [kw for kw in sufficient_keywords if isinstance(kw, str)] + [' '.join(seq) for seq in sufficient_sequences]
    insufficient_docs = [kw for kw in insufficient_keywords if isinstance(kw, str)] + [' '.join(seq) for seq in insufficient_sequences]
    all_docs = sufficient_docs + insufficient_docs + [enhanced_desc]

    # Check for empty or invalid input
    if not desc_str.strip() or (not sufficient_docs and not insufficient_docs):
        return "not sufficient", 0.0

    # Vectorize
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    desc_vector = tfidf_matrix[-1]
    suff_vectors = tfidf_matrix[:len(sufficient_docs)] if sufficient_docs else None
    insuff_vectors = tfidf_matrix[len(sufficient_docs):-1] if insufficient_docs else None

    # Calculate similarities
    suff_similarity = np.mean(cosine_similarity(desc_vector, suff_vectors)) if suff_vectors is not None else 0
    insuff_similarity = np.mean(cosine_similarity(desc_vector, insuff_vectors)) if insuff_vectors is not None else 0

    # Determine sufficiency
    base_score = suff_similarity - insuff_similarity if insuff_similarity > 0 else suff_similarity
    boost = min(total_suff_matches * 0.1, 0.5)
    final_score = min(base_score + boost, 1.0)
    is_sufficient = (final_score > 0.5) or (total_suff_matches > total_insuff_matches and final_score > 0)
    percentage = round(final_score * 100, 2)
    sufficiency_label = "sufficient" if is_sufficient else "not sufficient"

    return sufficiency_label, percentage

# Load CSV file
df = pd.read_csv('data.csv')

# Initialize lists to store results
sufficiency_labels = []
percentages = []

# Process each row
for index, row in df.iterrows():
    short_desc = row["short description"]
    sufficiency_label, percentage = analyze_description(short_desc)
    sufficiency_labels.append(sufficiency_label)
    percentages.append(percentage)

# Add new columns to DataFrame
df['regex matching'] = sufficiency_labels
df['percentage of sufficiency'] = percentages

# Save to Excel
df.to_excel('output_results.xlsx', index=False)
