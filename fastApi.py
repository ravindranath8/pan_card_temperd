from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize FastAPI and Jinja2Templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Function to highlight differences between original and tampered images
def highlight_difference(original_image_cv, tampered_image_cv):
    try:
        diff = cv2.absdiff(original_image_cv, tampered_image_cv)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresholded = cv2.threshold(diff_gray, 50, 255, cv2.THRESH_BINARY)
        tampered_image = tampered_image_cv.copy()
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 500:
                cv2.drawContours(tampered_image, [contour], -1, (0, 255, 0), 3)
        return tampered_image, contours, diff_gray
    except Exception as e:
        logger.error(f"Error in highlight_difference: {e}")
        raise HTTPException(status_code=500, detail="Error processing images")

# Function to process images and calculate tampering and SSIM
def detect_tampering(original_image: Image, tampered_image: Image):
    try:
        original_image_cv = np.array(original_image)
        original_image_cv = cv2.cvtColor(original_image_cv, cv2.COLOR_RGB2BGR)
        tampered_image_cv = np.array(tampered_image)
        tampered_image_cv = cv2.cvtColor(tampered_image_cv, cv2.COLOR_RGB2BGR)
        tampered_image_with_highlight, contours, diff_gray = highlight_difference(original_image_cv, tampered_image_cv)
        total_pixels = diff_gray.size
        differing_pixels = np.count_nonzero(diff_gray)
        tampering_percentage = (differing_pixels / total_pixels) * 100
        original_gray = cv2.cvtColor(original_image_cv, cv2.COLOR_BGR2GRAY)
        tampered_gray = cv2.cvtColor(tampered_image_cv, cv2.COLOR_BGR2GRAY)
        ssim_value = ssim(original_gray, tampered_gray)
        ssim_percentage = ssim_value * 100
        return tampered_image_with_highlight, contours, tampering_percentage, ssim_percentage
    except Exception as e:
        logger.error(f"Error in detect_tampering: {e}")
        raise HTTPException(status_code=500, detail="Error detecting tampering")

# HTML page for the frontend
@app.get("/", response_class=HTMLResponse)
async def get_upload_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Endpoint to handle image upload and process images
@app.post("/upload/")
async def upload_images(original: UploadFile = File(...), tampered: UploadFile = File(...)):
    try:
        original_image = Image.open(io.BytesIO(await original.read()))
        tampered_image = Image.open(io.BytesIO(await tampered.read()))
        if original_image.format not in ["JPEG", "PNG"] or tampered_image.format not in ["JPEG", "PNG"]:
            raise HTTPException(status_code=400, detail="Unsupported image format. Please upload JPEG or PNG images.")
        tampered_image = tampered_image.resize(original_image.size)  # Resize to match dimensions
        tampered_image_with_highlight, contours, tampering_percentage, ssim_percentage = detect_tampering(original_image, tampered_image)
        _, encoded_image = cv2.imencode('.PNG', tampered_image_with_highlight)
        highlighted_image = encoded_image.tobytes()
        return JSONResponse(content={
            "tampering_detected": len(contours) > 0,
            "tampering_percentage": tampering_percentage,
            "ssim_percentage": ssim_percentage,
            "highlighted_image": highlighted_image
        })
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image file")
    except Exception as e:
        logger.error(f"Error in upload_images: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")