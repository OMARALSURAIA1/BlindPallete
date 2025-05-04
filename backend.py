from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import shutil
import os
import numpy as np
# from catboost import CatBoostClassifier  # Changed import
# from .Functions import ExtractFeatureFromVideo
import joblib
from fastapi import FastAPI, File, UploadFile,Form
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras import layers, Model
import os
import cv2
import numpy as np
import zipfile
from sklearn.metrics.pairwise import euclidean_distances






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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
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

def get_dominant_color(image_bgr, k=3):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    outputs = predictor(image_rgb)
    instances = outputs["instances"].to("cpu")
    masks = instances.pred_masks.numpy()

    if len(masks) == 0:
        return None

    # Merge all masks
    combined_mask = np.any(masks, axis=0)

    # Extract only masked pixels from the image
    masked_pixels = image_rgb[combined_mask]

    if masked_pixels.size == 0:
        return None

    # Apply KMeans clustering to get the dominant color
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(masked_pixels)
    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]

    return dominant_color.astype(int).tolist()

import math

color_names = {
    "black": [0, 0, 0], "white": [255, 255, 255], "red": [255, 0, 0],
    "green": [0, 128, 0], "blue": [0, 0, 255], "yellow": [255, 255, 0],
    "cyan": [0, 255, 255], "magenta": [255, 0, 255], "gray": [128, 128, 128],
    "silver": [192, 192, 192], "maroon": [128, 0, 0], "olive": [128, 128, 0],
    "purple": [128, 0, 128], "teal": [0, 128, 128], "navy": [0, 0, 128],
    "pink": [255, 192, 203], "baby pink": [244, 194, 194], "deep pink": [255, 20, 147],
    "hot pink": [255, 105, 180], "light pink": [255, 182, 193], "salmon": [250, 128, 114],
    "light salmon": [255, 160, 122], "dark salmon": [233, 150, 122],
    "orange": [255, 165, 0], "dark orange": [255, 140, 0], "coral": [255, 127, 80],
    "light coral": [240, 128, 128], "tomato": [255, 99, 71], "peach": [255, 229, 180],
    "beige": [245, 245, 220], "wheat": [245, 222, 179], "moccasin": [255, 228, 181],
    "khaki": [240, 230, 140], "gold": [255, 215, 0], "light yellow": [255, 255, 224],
    "light goldenrod": [250, 250, 210], "lavender": [230, 230, 250], "thistle": [216, 191, 216],
    "plum": [221, 160, 221], "violet": [238, 130, 238], "orchid": [218, 112, 214],
    "fuchsia": [255, 0, 255], "indigo": [75, 0, 130], "dark violet": [148, 0, 211],
    "blue violet": [138, 43, 226], "sky blue": [135, 206, 235], "light blue": [173, 216, 230],
    "powder blue": [176, 224, 230], "steel blue": [70, 130, 180],
    "dodger blue": [30, 144, 255], "royal blue": [65, 105, 225],
    "medium blue": [0, 0, 205], "midnight blue": [25, 25, 112],
    "spring green": [0, 255, 127], "lime green": [50, 205, 50],
    "forest green": [34, 139, 34], "light green": [144, 238, 144],
    "dark green": [0, 100, 0], "sea green": [46, 139, 87],
    "medium sea green": [60, 179, 113], "light sea green": [32, 178, 170],
    "pale green": [152, 251, 152], "turquoise": [64, 224, 208],
    "medium turquoise": [72, 209, 204], "light cyan": [224, 255, 255],
    "azure": [240, 255, 255], "alice blue": [240, 248, 255],
    "mint cream": [245, 255, 250], "honeydew": [240, 255, 240],
    "ivory": [255, 255, 240], "linen": [250, 240, 230],
    "seashell": [255, 245, 238], "snow": [255, 250, 250],
    "ghost white": [248, 248, 255], "floral white": [255, 250, 240],
    "light gray": [211, 211, 211], "dimgray": [105, 105, 105],
    "dark slate gray": [47, 79, 79]
}

def color_distance(rgb1, rgb2):
    return math.sqrt(sum([(c1 - c2) ** 2 for c1, c2 in zip(rgb1, rgb2)]))

# Function to find the closest color name
def closest_color_name(rgb_color):
    min_distance = float('inf')
    closest_name = None
    for name, color_rgb in color_names.items():
        distance = color_distance(rgb_color, color_rgb)
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name


#predict model
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import pickle
from sklearn.metrics.pairwise import euclidean_distances

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import pickle
from sklearn.metrics.pairwise import euclidean_distances
import time

def build_encoder(latent_dim):
    encoder_inputs = layers.Input(shape=(128, 128, 3))
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2D(128, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=(tf.shape(z_mean)[0], latent_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    return Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


def build_decoder(latent_dim):
    decoder_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(16 * 16 * 128, activation='relu')(decoder_inputs)
    x = layers.Reshape((16, 16, 128))(x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    decoder_outputs = layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)
    return Model(decoder_inputs, decoder_outputs, name="decoder")


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)


# Load saved data instead of preprocessing
print("Loading saved data and models...")
start_time = time.time()

saved_data_folder = 'saved_data'

# Load models
try:
    # Try to load the saved models first
    print("Loading encoder and decoder models...")
    encoder = tf.keras.models.load_model(os.path.join(saved_data_folder, 'encoder_model.h5'))
    decoder = tf.keras.models.load_model(os.path.join(saved_data_folder, 'decoder_model.h5'))
except:
    # If models aren't saved yet, build them and load weights
    print("Saved models not found. Building models and loading weights...")
    latent_dim = 32
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder)
    
    # Initialize with a dummy input to build the model
    dummy_input = tf.random.normal((1, 128, 128, 3))
    _ = vae(dummy_input)
    
    # Load original weights
    vae.load_weights('models/vae_final.weights.h5')

# Create VAE model
vae = VAE(encoder, decoder)

# Load preprocessed images
try:
    print("Loading preprocessed images...")
    # raise FileNotFoundError("Forcing fallback to reprocess images.")
    pants_train_images = np.load(os.path.join(saved_data_folder, 'pants_train_images2.npy'))
    tops_train_images = np.load(os.path.join(saved_data_folder, 'tops_train_images2.npy'))
except:
    print("Preprocessed images not found. Loading will be slower as images need to be processed.")
    # pants_train_folder = 'data_split_pants/train'    
    # tops_train_folder = 'data_split_top/train'
    pants_train_folder = 'newdata/data_split_pants/train'    
    tops_train_folder = 'newdata/data_split_top/data_split_top/train'
    
    # Define the original loading function
    def load_and_preprocess_images(folder_path, target_size=(128, 128)):
        images = []
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, target_size)
                    img = img.astype('float32') / 255.0  # Normalization
                    images.append(img)
        return np.array(images)
    
    # Load images using original function
    pants_train_images = load_and_preprocess_images(pants_train_folder)
    tops_train_images = load_and_preprocess_images(tops_train_folder)
    
    # Save for future use
    os.makedirs(saved_data_folder, exist_ok=True)
    np.save(os.path.join(saved_data_folder, 'pants_train_images2.npy'), pants_train_images)
    np.save(os.path.join(saved_data_folder, 'tops_train_images2.npy'), tops_train_images)

print(f"👖 Pants trained images:", pants_train_images.shape)
print(f"👕 Tops trained images:", tops_train_images.shape)

# Load file paths
try:
    print("Loading file paths...")
    # raise FileNotFoundError("Forcing fallback to reprocess images.")
    with open(os.path.join(saved_data_folder, 'tops_paths2.pkl'), 'rb') as f:
        tops_paths = pickle.load(f)
    
    with open(os.path.join(saved_data_folder, 'pants_paths2.pkl'), 'rb') as f:
        pants_paths = pickle.load(f)
except:
    print("File paths not found. Generating path lists...")
    # pants_train_folder = 'data_split_pants/train'    
    # tops_train_folder = 'data_split_top/train'
    pants_train_folder = 'newdata/data_split_pants/train'    
    tops_train_folder = 'newdata/data_split_top/data_split_top/train'
    
    def git_the_path_in_list(folder, paths):
        for filename in sorted(os.listdir(folder)):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                 paths.append(os.path.join(folder, filename))
        return paths
    
    tops_paths = []
    pants_paths = []
    git_the_path_in_list(tops_train_folder, tops_paths)
    git_the_path_in_list(pants_train_folder, pants_paths)
    
    # Save for future use
    os.makedirs(saved_data_folder, exist_ok=True)
    with open(os.path.join(saved_data_folder, 'tops_paths2.pkl'), 'wb') as f:
        pickle.dump(tops_paths, f)
    
    with open(os.path.join(saved_data_folder, 'pants_paths2.pkl'), 'wb') as f:
        pickle.dump(pants_paths, f)

print(f"Number of Tops:", len(tops_paths))
print(f"Number of Pants:", len(pants_paths))

# Load latent vectors
try:
    print("Loading latent vectors...")
    # raise FileNotFoundError("Forcing fallback to reprocess images.")
    tops_latent_vectors = np.load(os.path.join(saved_data_folder, 'tops_latent_vectors2.npy'))
    pants_latent_vectors = np.load(os.path.join(saved_data_folder, 'pants_latent_vectors2.npy'))
except:
    print("Latent vectors not found. Computing latent representations...")
    # Compute latent vectors
    tops_latent_vectors = encoder.predict(tops_train_images)[0]
    pants_latent_vectors = encoder.predict(pants_train_images)[0]
    
    # Save for future use
    os.makedirs(saved_data_folder, exist_ok=True)
    np.save(os.path.join(saved_data_folder, 'tops_latent_vectors2.npy'), tops_latent_vectors)
    np.save(os.path.join(saved_data_folder, 'pants_latent_vectors2.npy'), pants_latent_vectors)

print("Laten Space of Tops:", tops_latent_vectors.shape)
print("Laten Space of Pants:", pants_latent_vectors.shape)

end_time = time.time()
print(f"All resources loaded in {end_time - start_time:.2f} seconds")

# Keep original function definitions unchanged
def find_best_matching(img_array_input, latents, paths_of_prediect):
    z_mean, _, _ = encoder.predict(np.expand_dims(img_array_input, axis=0))
    distances = euclidean_distances(z_mean, latents)
    best_index = np.argmin(distances)
    return paths_of_prediect[best_index]


def preprocess_input(path_img):
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0
    return img


# This maintains backward compatibility with your existing code
def load_and_preprocess_images(folder_path, target_size=(128, 128)):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size)
                img = img.astype('float32') / 255.0  # Normalization
                images.append(img)
    return np.array(images)


def git_the_path_in_list(folder, paths):
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
             paths.append(os.path.join(folder, filename))
    return paths


print("READY!")
print("READY!")
@app.post("/match")
async def match_image(image: UploadFile = File(...),category: str = Form(...)):
    # Open the image with PIL
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    filename = image.filename  # You could use uuid.uuid4().hex + ".png" for uniqueness
    # print(category)
    # Get the extension of the uploaded image (if it exists, otherwise default to .png)
    file_extension = filename.split('.')[-1] if '.' in filename else 'png'
    
    # Construct the save path with the correct extension
    save_path = os.path.join(temp_dir, f"uploaded_image.{file_extension}")

    pil_image = Image.open(image.file)
    pil_image.save(save_path)

    # Process image with model
    # result_image = process_image(pil_image)
    process_img = preprocess_input(save_path)
    if category == "Bottoms":
        best_tops_path = find_best_matching(process_img, tops_latent_vectors, tops_paths)
    else:
        best_tops_path = find_best_matching(process_img, pants_latent_vectors, pants_paths)


    result_image = Image.open(best_tops_path)

    img_byte_arr = io.BytesIO()
    result_image.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)



    return StreamingResponse(img_byte_arr, media_type="image/png")


@app.post("/color")
async def match_image(image: UploadFile = File(...)):
    # Open the image with PIL
    pil_image = Image.open(image.file)

    
    cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    colors = get_dominant_color(cv_image)
    print(colors)
    color_name = closest_color_name(colors)
    print("Color name:", color_name)
    print("Dominant RGB colors:", colors)
    colors = [colors]

    return {"colors": colors, "color_name": color_name}
