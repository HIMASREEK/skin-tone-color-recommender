from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import cv2
import os
import io
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from typing import List, Dict
from image_processor import ImageProcessor

# -------------------------------
# Initialize FastAPI app
# -------------------------------
app = FastAPI(
    title="Dvaltor Fashion Recommendation API",
    description="Detect faces, analyze skin tone, and get color recommendations.",
    version="2.0.0"
)

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "https://dvaltor.com", "https://app.dvaltor.com"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Templates and Static Files
# -------------------------------
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# -------------------------------
# Load DNN face detection model
# -------------------------------
prototxt_path = "./Models/deploy.prototxt.txt"
model_path = "./Models/res10_300x300_ssd_iter_140000.caffemodel"

if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
    raise RuntimeError("Face detection model files missing.")

net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# -------------------------------
# Load skin tone recommendation CSV
# -------------------------------
try:
    df_recommendations = pd.read_csv("./Dataset/skin_tone_recommendations.csv")
except FileNotFoundError:
    raise RuntimeError("CSV file missing in './Dataset/'.")

def hex_to_rgb(hex_code: str) -> List[int]:
    hex_code = hex_code.lstrip('#')
    return [int(hex_code[i:i+2], 16) for i in (0, 2, 4)]

df_recommendations['RGB'] = df_recommendations['Hex Code'].apply(hex_to_rgb)
SKIN_TONE_RGB_LIST = np.array(df_recommendations['RGB'].tolist())

# -------------------------------
# FFmpeg Image Processor
# -------------------------------
image_processor = ImageProcessor(max_width=1920, max_height=1080, quality=85)

# -------------------------------
# Skin tone extraction
# -------------------------------
def get_average_skin_color_from_roi(face_roi: np.ndarray) -> List[int] | None:
    try:
        hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)

        # Skin mask
        mask = cv2.inRange(
            hsv,
            np.array([0, 48, 80]),
            np.array([20, 255, 255])
        )

        skin = cv2.bitwise_and(face_roi, face_roi, mask=mask)

        pixels = skin.reshape((-1, 3))
        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]

        if len(pixels) == 0:
            return None

        return np.mean(pixels, axis=0).astype(int).tolist()

    except Exception as e:
        print("Skin color extraction error:", e)
        return None

# -------------------------------
# Recommendation logic
# -------------------------------
def get_recommendation_json(avg_rgb_color: List[int]) -> Dict:
    distances = euclidean_distances([avg_rgb_color], SKIN_TONE_RGB_LIST)
    closest_index = np.argmin(distances)
    data = df_recommendations.iloc[closest_index]

    recommendations = [
        {
            "name": data[f"Rec_Color_{i}_Name"],
            "hex": data[f"Rec_Color_{i}_Hex"]
        }
        for i in range(1, 21)
        if pd.notna(data.get(f"Rec_Color_{i}_Name")) and pd.notna(data.get(f"Rec_Color_{i}_Hex"))
    ]

    return {
        "detected_average_rgb": avg_rgb_color,
        "closest_skin_tone": {
            "description": data['Skin Shade Description'],
            "hex_code": data['Hex Code']
        },
        "recommended_colors": recommendations
    }

# -------------------------------
# Frontend Home Page
# -------------------------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )

# -------------------------------
# Main Analyze Endpoint
# -------------------------------
@app.post("/analyze-fashion/")
async def detect_and_recommend(file: UploadFile = File(...)):
    contents = await file.read()

    # Decode uploaded image directly (faster than FFmpeg for analysis)
    np_arr = np.frombuffer(contents, np.uint8)
    original_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if original_image is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Resize large images for faster processing
    max_width = 800
    if original_image.shape[1] > max_width:
        scale = max_width / original_image.shape[1]
        new_height = int(original_image.shape[0] * scale)
        original_image = cv2.resize(original_image, (max_width, new_height))

    (h, w) = original_image.shape[:2]

    # Face detection
    blob = cv2.dnn.blobFromImage(
        cv2.resize(original_image, (300, 300)),
        1.0,
        (300, 300),
        (104, 177, 123)
    )
    net.setInput(blob)
    detections = net.forward()

    best_face = None
    best_confidence = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5 and confidence > best_confidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            roi = original_image[
                max(0, startY):min(h, endY),
                max(0, startX):min(w, endX)
            ]

            if roi.size > 0:
                avg_color = get_average_skin_color_from_roi(roi)

                if avg_color:
                    recommendation = get_recommendation_json(avg_color)

                    best_face = {
                        "success": True,
                        "face_confidence": round(float(confidence), 4),
                        "face_box": [int(startX), int(startY), int(endX), int(endY)],
                        "skin_tone": recommendation["closest_skin_tone"]["description"],
                        "skin_hex": recommendation["closest_skin_tone"]["hex_code"],
                        "average_rgb": recommendation["detected_average_rgb"],
                        "recommended_colors": recommendation["recommended_colors"]
                    }

                    best_confidence = confidence

    if not best_face:
        raise HTTPException(
            status_code=404,
            detail="No face detected or skin tone could not be extracted."
        )

    return JSONResponse(content=best_face)

# -------------------------------
# Image info endpoint
# -------------------------------
@app.post("/image-info/")
async def get_image_info(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        info = image_processor.get_image_info(contents)

        return JSONResponse(content={
            "image_info": info,
            "status": "success"
        })

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process image: {str(e)}"
        )

# -------------------------------
# Optimize image endpoint
# -------------------------------
@app.post("/optimize-image/")
async def optimize_uploaded_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        optimized = image_processor.optimize_image(contents)

        return StreamingResponse(
            io.BytesIO(optimized),
            media_type="image/jpeg",
            headers={
                "Content-Disposition": f"attachment; filename=optimized_{file.filename}"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to optimize image: {str(e)}"
        )