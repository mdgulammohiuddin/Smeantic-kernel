import pandas as pd
import spacy
from typing import List, Dict, Any

# Load the spaCy model for NLP
nlp = spacy.load("en_core_web_sm")

def is_description_sufficient(description: str) -> tuple[bool, List[str]]:
    """
    Determine if a description is sufficient using NLP and extract keywords.
    Sufficiency criteria: length > 20 chars, contains entities or specific terms.
    """
    if not description or not isinstance(description, str) or len(description.strip()) < 20:
        return False, []
    
    # Process description with spaCy
    doc = nlp(description)
    
    # Extract entities (e.g., ORG, GPE, PRODUCT) and tokens
    entities = [ent.text.lower() for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PRODUCT"]]
    tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    
    # Define sufficiency: must have entities or meaningful tokens
    if len(entities) > 0 or len(tokens) > 5:
        # Extract keywords (entities + non-stop words)
        keywords = list(set(entities + tokens))
        return True, keywords[:5]  # Limit to 5 keywords for brevity
    return False, []

def process_excel_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process Excel file to extract 'NO' and 'Description', evaluate sufficiency.
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
    
    # Print results
    for result in results:
        print(f"No: {result['No']}, Sufficient: {result['sufficient']}, Keywords: {result['keywords']}")

if __name__ == "__main__":
    main()
