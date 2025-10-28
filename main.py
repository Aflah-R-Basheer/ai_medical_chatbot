# VERSION 1 OF CODE

# from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import HTMLResponse, JSONResponse
# import base64
# import requests
# import io
# from PIL import Image
# from dotenv import load_dotenv
# import os
# import logging


# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# load_dotenv()

# app = FastAPI()

# templates = Jinja2Templates(directory="templates")

# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY is not set in the .env file")

# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/upload_and_query")
# async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
#     try:
#         image_content = await image.read()
#         if not image_content:
#             raise HTTPException(status_code=400, detail="Empty file")
        
#         encoded_image = base64.b64encode(image_content).decode("utf-8")

#         try:
#             img = Image.open(io.BytesIO(image_content))
#             img.verify()
#         except Exception as e:
#             logger.error(f"Invalid image format: {str(e)}")
#             raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")


#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "text", "text": query},
#                     {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
#                 ]
#             }
#         ]

#         def make_api_request(model):
#             response = requests.post(
#                 GROQ_API_URL,
#                 json={
#                     "model": model,
#                     "messages": messages,
#                     "max_tokens": 1000
#                 },
#                 headers={
#                     "Authorization": f"Bearer {GROQ_API_KEY}",
#                     "Content-Type": "application/json"
#                 },
#                 timeout=30
#             )
#             return response

#         # Make requests to both models
#         llama_response = make_api_request("llama-3.2-11b-vision-preview")
#         llama_response = make_api_request("llama-3.2-90b-vision-preview")

#         # Process responses
#         responses = {}
#         for model, response in [("llama", llama_response), ("llama", llama_response)]:
#             if response.status_code == 200:
#                 result = response.json()
#                 answer = result["choices"][0]["message"]["content"]
#                 logger.info(f"Processed response from {model} API: {answer[:100]}...")
#                 responses[model] = answer
#             else:
#                 logger.error(f"Error from {model} API: {response.status_code} - {response.text}")
#                 responses[model] = f"Error from {model} API: {response.status_code}"

#         return JSONResponse(status_code=200, content=responses)

#     except HTTPException as he:
#         logger.error(f"HTTP Exception: {str(he)}")
#         raise he
#     except Exception as e:
#         logger.error(f"An unexpected error occurred: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)

# VERSIOB 2 OF CODE

# from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
# from fastapi.templating import Jinja2Templates
# from fastapi.responses import HTMLResponse, JSONResponse
# import base64
# import requests
# import io
# from PIL import Image
# from dotenv import load_dotenv
# import os
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# load_dotenv()

# app = FastAPI()

# templates = Jinja2Templates(directory="templates")

# GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# if not GROQ_API_KEY:
#     raise ValueError("GROQ_API_KEY is not set in the .env file")

# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/upload_and_query")
# async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
#     try:
#         # Read and validate image
#         image_content = await image.read()
#         if not image_content:
#             raise HTTPException(status_code=400, detail="Empty file")
        
#         try:
#             img = Image.open(io.BytesIO(image_content))
#             img.verify()
#         except Exception as e:
#             logger.error(f"Invalid image format: {str(e)}")
#             raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")
        
#         # Encode image in base64
#         encoded_image = base64.b64encode(image_content).decode("utf-8")

#         # Convert query + image into a single string (required by the model)
#         messages = [
#             {
#                 "role": "user",
#                 "content": f"Query: {query}\nImage (base64): {encoded_image}"
#             }
#         ]

#         # Function to make the API request
#         def make_api_request(model):
#             response = requests.post(
#                 GROQ_API_URL,
#                 json={
#                     "model": model,
#                     "messages": messages,
#                     "max_tokens": 1000
#                 },
#                 headers={
#                     "Authorization": f"Bearer {GROQ_API_KEY}",
#                     "Content-Type": "application/json"
#                 },
#                 timeout=30
#             )
#             return response

#         # Request the recommended model
#         response = make_api_request("llama-3.3-70b-versatile")

#         if response.status_code == 200:
#             result = response.json()
#             answer = result["choices"][0]["message"]["content"]
#             logger.info(f"Processed response from llama API: {answer[:100]}...")
#             return JSONResponse(status_code=200, content={"llama": answer})
#         else:
#             logger.error(f"Error from llama API: {response.status_code} - {response.text}")
#             return JSONResponse(status_code=500, content={"error": f"Error from llama API: {response.status_code}"})

#     except HTTPException as he:
#         logger.error(f"HTTP Exception: {str(he)}")
#         raise he
#     except Exception as e:
#         logger.error(f"An unexpected error occurred: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, port=8000)

# VERSION 3 OF CODE

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
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

# Static folder to serve uploaded images
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

templates = Jinja2Templates(directory="templates")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/upload_and_query")
async def upload_and_query(image: UploadFile = File(...), query: str = Form(...)):
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

        # Encode the image in base64
        encoded_image = base64.b64encode(image_content).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
                ]
            }
        ]

        # Make API request to the new Llama 4 Scout model
        response = requests.post(
            GROQ_API_URL,
            json={
                "model": "meta-llama/llama-4-scout-17b-16e-instruct",
                "messages": messages,
                "max_tokens": 2000 # Increased for a more capable model
            },
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            logger.info(f"Processed response from llama-4-scout API: {answer[:100]}...")
            return JSONResponse(status_code=200, content={"llama": answer})
        else:
            logger.error(f"Error from llama-4-scout API: {response.status_code} - {response.text}")
            error_detail = response.json().get("error", {}).get("message", response.text)
            raise HTTPException(status_code=response.status_code, detail=f"Error from Llama 4 Scout API: {error_detail}")

    except HTTPException as he:
        logger.error(f"HTTP Exception: {str(he.detail)}")
        raise he
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)

