import base64
from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import io
import logging
from PIL import Image
import requests
from dotenv import load_dotenv
import uvicorn # Ensure uvicorn is imported for direct running

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Load environment variables
load_dotenv()

app = FastAPI()

# 2. Setup Static Files and Templates
# Static folder is set up but not strictly needed for this single-file app, 
# but we keep it for consistency.
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory="templates")

# 3. API Configuration
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    # This will prevent the app from starting if the key is missing.
    raise ValueError("GROQ_API_KEY is not set in the .env file")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Serves the main HTML interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
    """Handles image upload, queries the Llama model, and returns the response."""
    try:
        image_content = await image.read()
        if not image_content:
            raise HTTPException(status_code=400, detail="Empty file")

        # Validate image
        try:
            img = Image.open(io.BytesIO(image_content))
            img.verify()
        except Exception as e:
            logger.error(f"Invalid image format: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        # Encode the image in base64 for the API payload
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        # Construct the multi-modal message payload
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        # Make API request to the Llama 4 Scout model
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": messages,
                "max_tokens": 2000
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=45 # Increased timeout slightly for large model processing
        )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"Processed response from llama-4-scout API: {answer[:100]}...")
            return JSONResponse(status_code=200, content={"llama": answer})
        else:
            logger.error(f"Error from llama-4-scout API: {response.status_code} - {response.text}")
            # Try to extract the error message from the response JSON
            try:
                error_detail = response.json().get("error", {}).get("message", response.text)
            except:
                 error_detail = response.text
            raise HTTPException(status_code=response.status_code, detail=f"Error from Llama 4 Scout API: {error_detail}")

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he.detail)}")
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        # Raise a 500 internal server error for unexpected issues
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    # If the file is run directly using `python app.py`, we fall back to uvicorn.run
    # However, the preferred method is `uvicorn app:app --reload`
    uvicorn.run(app, port=8000)
