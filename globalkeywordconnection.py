import pandas as pd
import spacy
import json
from typing import List, Dict, Any

# Load the spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

def is_description_sufficient(description: str) -> tuple[bool, List[Dict[str, str]]]:
    """
    Determine if a description is sufficient using NLP and extract keywords.
    Sufficiency criteria: contains at least one entity or more than two noun chunks.
    Returns a tuple of (sufficient, keywords), where keywords are structured with text and type.
    """
    if not description or not isinstance(description, str) or len(description.strip()) < 20:
        return False, []
    
    # Process description with spaCy
    doc = nlp(description)
    
    # Extract entities (e.g., ORG, GPE, PRODUCT)
    entities = [{"text": ent.text.lower(), "type": ent.label_} for ent in doc.ents 
                if ent.label_ in ["ORG", "GPE", "PRODUCT"]]
    
    # Extract noun chunks that do not overlap with entities
    noun_chunks = [{"text": chunk.text.lower(), "type": "NOUN_CHUNK"} 
                   for chunk in doc.noun_chunks 
                   if all(token.ent_type_ == "" for token in chunk)]
    
    # Combine entities and noun chunks, preserving order
    keywords = entities + noun_chunks
    
    # Define sufficiency: at least one entity or more than two noun chunks
    sufficient = len(entities) > 0 or len(noun_chunks) > 2
    
    return sufficient, keywords

def process_excel_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process Excel file to extract 'NO' and 'Description', evaluate sufficiency.
    Returns a list of dictionaries with 'No', 'sufficient', and 'keywords'.
    """
    try:
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Normalize column names for case-insensitive matching
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces
        columns = {col.lower(): col for col in df.columns}
        
        # Check for 'NO' and 'Description' columns (case-insensitive)
        no_col = next((col for col in columns if col.lower() == 'no'), None)
        desc_col = next((col for col in columns if col.lower() == 'description'), None)
        
        if not no_col or not desc_col:
            raise ValueError("Excel file must contain 'NO' and 'Description' columns")
        
        results = []
        for _, row in df.iterrows():
            no = row[columns[no_col.lower()]]
            description = str(row[columns[desc_col.lower()]]) if pd.notna(row[columns[desc_col.lower()]]) else ""
            
            # Evaluate sufficiency and extract keywords
            sufficient, keywords = is_description_sufficient(description)
            
            # Append result
            results.append({
                "No": no,
                "sufficient": sufficient,
                "keywords": keywords
            })
        
        return results
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

def main():
    # Example usage
    file_path = r"C:\Users\2000078212\OneDrive - Hexaware Technologies\Desktop\ito_copilot_keywords\Automatable_Use_cases 5 (1).xlsx"  # Replace with your Excel file path
    results = process_excel_file(file_path)
    
    # Write results to a JSON file
    with open("output.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()
