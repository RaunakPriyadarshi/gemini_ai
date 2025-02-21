import os
import json
import io
from PIL import Image
import google.generativeai as genai

# Working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# Path of config_data file
config_file_path = os.path.join(working_dir, "config.json")

# Load configuration data
with open(config_file_path, "r") as config_file:
    config_data = json.load(config_file)

# Load the GOOGLE API key
GOOGLE_API_KEY = config_data.get("GOOGLE_API_KEY")

# Configure Google Generative AI API
genai.configure(api_key=GOOGLE_API_KEY)


def load_gemini_pro_model():
    """Load the Gemini Pro text model."""
    return genai.GenerativeModel("gemini-pro")


def gemini_pro_vision_response(prompt, image):
    """Generate a caption for an image using Gemini API."""
    
    # Convert PIL Image to bytes
    if isinstance(image, Image.Image):
        image_bytes_io = io.BytesIO()
        image.save(image_bytes_io, format="PNG")
        image_bytes = image_bytes_io.getvalue()
    elif isinstance(image, str):  # If it's a file path, read it
        with open(image, "rb") as img_file:
            image_bytes = img_file.read()
    else:
        raise ValueError("Invalid image input. Provide a file path or PIL image object.")

    # Initialize Gemini model (use gemini-1.5 models instead of gemini-pro-vision)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")

    # Convert image bytes to Google AI-compatible format
    image_parts = {
        "mime_type": "image/png",
        "data": image_bytes
    }

    # Get response from Gemini
    response = gemini_model.generate_content([prompt, image_parts])

    return response.text if response else "No response received."


def embeddings_model_response(input_text):
    """Get text embeddings for retrieval tasks."""
    embedding_model = "models/embedding-001"
    embedding = genai.embed_content(
        model=embedding_model,
        content=input_text,
        task_type="retrieval_document"
    )
    return embedding.get("embedding", [])


def gemini_pro_response(user_prompt):
    """Generate a text response using Gemini Pro."""
    gemini_model = genai.GenerativeModel("gemini-pro")
    response = gemini_model.generate_content(user_prompt)
    
    return response.text if response else "No response received."
