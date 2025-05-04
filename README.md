# 👕🧠 BlindPallet: Colorblind-Friendly Outfit Recommendation using VAE

This project presents a **computer vision-based recommendation system** that suggests matching outfits for colorblind users using **visual similarity in latent space**.

---

## 🎯 Project Goal

To build an intelligent outfit recommendation system that:
- Understands clothing images (tops and pants),
- Encodes their **visual style** (especially color),
- Suggests **aesthetically matching combinations**, even if colors cannot be distinguished by the user,
- Helps colorblind individuals choose matching outfits confidently.

---

## 🚀 Features

- 🎨 Color Extraction: Extracts dominant colors using KMeans or histogram analysis.
- 🧩 Latent Space Embedding: Represents each clothing item's color numerically using UMAP.
- 🤝 Outfit Matching: Recommends matching wardrobe pieces based on color similarity (Euclidean distance).
- 🕹️ Colorblind Game Page: A small game to simulate colorblind perception.

---

## 🧩 Project Structure

| File/Folder | Description |
|-------------|-------------|
| `Model_Colorblind_VAE.ipynb` | The core VAE model: builds, trains, and evaluates the color-based autoencoder. |
| `vae_final.weights.h5` | Final trained weights for the VAE. |
| `How_to_run_model.ipynb` | Quick notebook to demonstrate how to use the model for prediction and outfit matching. |
| `color_extraction.ipynb` | Extracts dominant colors from each image (used for visualization in latent space). |
| `cleaned_men_pants_cloths.ipynb` / `cleaned_men_top_cloths.ipynb` | Preprocessing steps for preparing clean datasets. |
| `data_split_pants.zip` / `data_split_top.zip` | Zipped datasets split into training/testing sets. |
| `saved_data/` | Stores processed latent vectors and metadata. |
| `backend.py` | Backend logic (API-ready functions). |
| `frontend.py` / `gamepage.py` | (Optional) UI logic if integrated with a game or web interface. |
| `README.md` | You're reading it |

---

## 🧠 Model Details

- **Model Type**: Variational Autoencoder (VAE)
- **Latent Space**: 32-dimensional learned embeddings
- **Training**: Images of tops and pants, resized to 128x128
- **Loss**: Combination of reconstruction loss (MSE) and KL-divergence
- **Output**: Latent embeddings that represent each item's visual identity

---

## 🧪 How It Works

1. Each image is preprocessed and passed through the encoder.
2. The encoder outputs `z_mean` (latent vector), representing color/style.
3. For a given query (e.g., a shirt), the system compares its `z_mean` to all pants and retrieves the **most similar matches** using Euclidean distance.
4. Optionally, matching visuals can be shown using **merged outfit image**.

---

## 🚀 How to Run

1. Upload your own tops/pants images into `data_split_top/` and `data_split_pants/`.
2. Open and run `How_to_run_model.ipynb`.
3. The notebook:
   - Loads the model and weights
   - Encodes items
   - Optionally merges images for preview.

---

## 🧑‍💻 Example Use Case

> "Give me matching pants for this red t-shirt!"

The model:
- Encodes the t-shirt.
- Finds pants in your 'drawer' (preloaded dataset) with similar latent color style.

---

## 🛠️ Requirements

- Python 3.10+
- TensorFlow / Keras
- OpenCV, matplotlib, numpy, umap-learn
