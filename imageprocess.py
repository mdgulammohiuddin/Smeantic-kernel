from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools import tool
 
import easyocr
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import os
from dotenv import load_dotenv
 
# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
 
# Initialize the OCR reader
reader = easyocr.Reader(['en'])
 
@tool
def extract_text_from_image(image_source: str) -> dict:
    """
    Extracts OCR text and key information from the provided image source.
    
    Args:
        image_source (str): URL or local path to the image file.
    
    Returns:
        dict: Extracted OCR text and primary key information.
    """
    try:
        # Load the image
        if image_source.startswith(("http://", "https://")):
            response = requests.get(image_source)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_source)
        
        # Convert image to numpy array
        image_np = np.array(image)
 
        # Perform OCR
        ocr_data = reader.readtext(image_np)
        ocr_text = [text[1] for text in ocr_data]
 
        return {
            "ocr_text": ocr_text if ocr_text else None,
            "key_information": {
                "primary_text": ocr_data[0][1] if ocr_data else None,
            }
        }
    except Exception as e:
        return {"error": str(e)}
 
# Create the OCR agent
ocr_extractor_agent = Agent(
    name="OCR Extractor Agent",
    model=Groq(id="llama-3.3-70b-versatile", api_key=GROQ_API_KEY),
    tools=[extract_text_from_image],
    instructions=["Extract all OCR text and primary key data from the image."],
    markdown=True
)
 
# Example usage
ocr_extractor_agent.print_response("Extract text from image ../../assets/invoice.png", stream=True)









































