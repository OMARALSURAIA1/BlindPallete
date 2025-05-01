from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import numpy as np
# from catboost import CatBoostClassifier  # Changed import
# from .Functions import ExtractFeatureFromVideo
import joblib
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io






app = FastAPI()

# model = joblib.load("models\\")

import torch
import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from sklearn.cluster import KMeans

# Initialize the model once to reuse it across function calls
def setup_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"
    return DefaultPredictor(cfg)

predictor = setup_predictor()

# Extract dominant color from a masked object
def extract_dominant_color(image, mask, k=3):
    masked_pixels = image[mask == 1]
    if masked_pixels.size == 0:
        return None
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(masked_pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return dominant_color.astype(int)

# Main function
def get_dominant_colors(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    outputs = predictor(image_rgb)
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()

    dominant_colors = []
    for mask in masks:
        color = extract_dominant_color(image_rgb, mask)
        if color is not None:
            dominant_colors.append(color.tolist())  # Convert to list for easier JSON/log output

    return dominant_colors

# Simulated model function
def process_image(image: Image.Image) -> Image.Image:
    # Example: convert to grayscale (replace with your ML model output)
    return image.convert("L")


@app.post("/match")
async def match_image(image: UploadFile = File(...)):
    # Open the image with PIL
    pil_image = Image.open(image.file)

    # Process image with model
    result_image = process_image(pil_image)
    # cv_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
    # Convert result image to bytes
    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    # image_path = path  +'/tshirt/1125.jpg'
    # image = cv2.imread(image_path)
    
    # colors = get_dominant_colors(cv_image)
    # print("Dominant RGB colors:", colors)

    # Return image as a response
    return StreamingResponse(img_byte_arr, media_type="image/png")
    # return JSONResponse(content={
    #     "image": StreamingResponse(img_byte_arr, media_type="image/png"),
    #     "colors": colors
    # })

@app.post("/color")
async def match_image(image: UploadFile = File(...)):
    # Open the image with PIL
    pil_image = Image.open(image.file)

    
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    colors = get_dominant_colors(cv_image)
    print(colors)
    # print("Dominant RGB colors:", colors)
    return colors
