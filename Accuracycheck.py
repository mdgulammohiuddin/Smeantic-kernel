import pandas as pd
import json
import re
import logging
from pathlib import Path
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Setup logging to file
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('keyword_network.log', mode='a')  # Append to existing log
    ]
)

# Download NLTK resources
nltk.download('punkt_tab')
nltk.download('stopwords')

# Configuration
KEYWORD_JSON = r"C:\Users\2000078212\OneDrive - Hexaware Technologies\Desktop\ito_copilot_keywords\keyword_network.json"
VALIDATION_FILE = r"C:\Users\2000078212\OneDrive - Hexaware Technologies\Desktop\ito_copilot_keywords\validation_descriptions.xlsx"  # Update with your file path
OUTPUT_FILE = r"C:\Users\2000078212\OneDrive - Hexaware Technologies\Desktop\ito_copilot_keywords\sufficiency_results.xlsx"
MIN_SUBGRAPH_SIZE = 3  # Minimum keywords for sufficiency

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

# Step 2: Load keywords and sequences from JSON
def load_keywords_and_sequences(json_path):
    logging.info(f"Loading keywords and sequences from {json_path}")
    if not Path(json_path).exists():
        logging.error(f"JSON file not found: {json_path}")
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    keyword_dict = {}
    sequence_dict = {}
    for category_data in data:
        category = category_data['Category']
        keyword_dict[category] = [
            {
                'keyword': item['keyword'],
                'nested_keywords': [nk['keyword'] for nk in item['nested_keywords']],
                'is_sufficient': item['is_sufficient']
            }
            for item in category_data['Keywords']
        ]
        sequence_dict[category] = [
            {
                'sequence': item['sequence'],
                'nested_sequences': item['nested_sequences'],
                'is_sufficient': item['is_sufficient']
            }
            for item in category_data['Sequences']
        ]
    
    logging.info(f"Loaded keywords and sequences for categories: {list(keyword_dict.keys())}")
    return keyword_dict, sequence_dict

# Step 3: Check description sufficiency
def check_description_sufficiency(description, keywords, sequences, min_subgraph_size=MIN_SUBGRAPH_SIZE):
    if not isinstance(description, str) or not description.strip():
        return False, [], []
    
    cleaned_desc = preprocess_text(description)
    matched_keywords = []
    matched_sequences = []
    
    # Check keywords
    for kw_item in keywords:
        keyword = kw_item['keyword']
        if re.search(r'\b' + re.escape(keyword) + r'\b', cleaned_desc, re.IGNORECASE):
            matched_keywords.append(keyword)
    
    # Check sequences
    for seq_item in sequences:
        sequence = seq_item['sequence']
        all_present = all(
            re.search(r'\b' + re.escape(kw) + r'\b', cleaned_desc, re.IGNORECASE)
            for kw in sequence
        )
        if all_present:
            matched_sequences.append(sequence)
    
    # Determine sufficiency
    sufficient_keyword_match = False
    for kw_item in keywords:
        if kw_item['is_sufficient']:
            matched_count = sum(
                1 for kw in [kw_item['keyword']] + kw_item['nested_keywords']
                if kw in matched_keywords
            )
            if matched_count >= min_subgraph_size:
                sufficient_keyword_match = True
                break
    
    sufficient_sequence_match = any(
        seq_item['is_sufficient'] and seq_item['sequence'] in matched_sequences
        for seq_item in sequences
    )
    
    is_sufficient = sufficient_keyword_match or sufficient_sequence_match
    
    return is_sufficient, matched_keywords, matched_sequences

# Step 4: Process validation file
def process_validation_file(validation_file, keyword_dict, sequence_dict):
    logging.info(f"Processing validation file: {validation_file}")
    if not Path(validation_file).exists():
        logging.error(f"Validation file not found: {validation_file}")
        raise FileNotFoundError(f"Validation file not found: {validation_file}")
    
    df = pd.read_excel(validation_file, sheet_name=0)
    logging.info(f"Loaded {len(df)} rows from {validation_file}")
    
    # Verify required columns
    required_columns = ['Short description', 'Description', 'Automatable']
    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        logging.error(f"Missing columns: {missing}")
        raise ValueError(f"Validation file must contain columns: {required_columns}")
    
    # Initialize result columns
    df['Is_Sufficient'] = False
    df['Matched_Keywords'] = ''
    df['Matched_Sequences'] = ''
    
    # Process each description
    for idx, row in df.iterrows():
        # Combine Short description and Description
        short_desc = row['Short description']
        desc = row['Description']
        combined_desc = ' '.join(
            [str(short_desc), str(desc)] if not pd.isna(short_desc) and not pd.isna(desc)
            else [str(short_desc)] if not pd.isna(short_desc)
            else [str(desc)] if not pd.isna(desc)
            else ['']
        )
        
        # Get category from Automatable column
        automatable = str(row['Automatable']).lower()
        category = 'automatable' if automatable == 'automatable' else 'non_automatable'
        
        # Select keywords and sequences
        if category in keyword_dict:
            keywords = keyword_dict[category]
            sequences = sequence_dict[category]
        else:
            logging.warning(f"Invalid Automatable value '{automatable}' at index {idx}, skipping")
            continue
        
        # Check sufficiency
        is_suff, matched_kws, matched_seqs = check_description_sufficiency(
            combined_desc, keywords, sequences
        )
        df.at[idx, 'Is_Sufficient'] = is_suff
        df.at[idx, 'Matched_Keywords'] = ', '.join(matched_kws)
        df.at[idx, 'Matched_Sequences'] = '; '.join([', '.join(seq) for seq in matched_seqs])
    
    logging.info(f"Processed {len(df)} descriptions")
    return df

# Step 5: Main processing
def main():
    try:
        # Load keywords and sequences
        keyword_dict, sequence_dict = load_keywords_and_sequences(KEYWORD_JSON)
        
        # Process validation file
        result_df = process_validation_file(VALIDATION_FILE, keyword_dict, sequence_dict)
        
        # Save results
        logging.info(f"Saving results to {OUTPUT_FILE}")
        output_dir = Path(OUTPUT_FILE).parent
        if output_dir != Path("."):
            output_dir.mkdir(parents=True, exist_ok=True)
        result_df.to_excel(OUTPUT_FILE, index=False)
    
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        raise

if __name__ == '__main__':
    main()
    logging.info("Sufficiency check complete. Results saved to 'sufficiency_results.xlsx'.")
