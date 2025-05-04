import streamlit as st
import os
import time
import pandas as pd
import requests
from PIL import Image
import io
import base64
import json
import random
import webbrowser

# Set page configuration
st.set_page_config(
    page_title="BlindPalette",
    page_icon="🧥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS for styling
st.markdown("""
<style>
    /* Main styles */
    .main {
        background-color: #f8f9fa;
        padding: 2rem 3rem !important;
    }
    /* Add padding to the entire content area */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    /* Custom card style */
    .css-1r6slb0 {
        border-radius: 12px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05) !important;
    }
    /* Make expanders look better */
    .streamlit-expanderHeader {
        background-color: #f1f3f5;
        border-radius: 8px;
        padding: 1rem !important;
    }
    /* Footer styling */
    footer {
        visibility: hidden;
    }
    /* Custom button styling */
    .stButton>button {
        background-color: #4F46E5;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.75rem 1.25rem;
        font-weight: 500;
        transition: all 0.2s ease;
        margin: 0.5rem 0;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #4338CA;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
        transform: translateY(-2px);
    }
    /* Match button styling */
    .match-btn>button {
        background-color: #059669;
        font-size: 1.1rem;
        padding: 0.8rem 1.5rem;
    }
    .match-btn>button:hover {
        background-color: #047857;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.2);
    }
    /* Custom header */
    .custom-header {
        text-align: center;
        padding: 2.5rem 1.5rem;
        background: linear-gradient(135deg, #4F46E5, #7C3AED);
        color: white;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px rgba(79, 70, 229, 0.15);
    }
    .custom-header h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .custom-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    /* Card styles with better margins */
    .clothing-card {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid #e2e8f0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .clothing-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0, 0, 0, 0.12);
    }
    /* Category badges */
    .category-badge {
        background-color: #EDE9FE;
        color: #5B21B6;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 8px;
    }
    /* Better file uploader */
    .css-1cpxqw2, .css-1ekf893 {
        border: 2px dashed #4F46E5 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }
    /* Upload indicator styling */
    .upload-area {
        border: 2px dashed #4F46E5;
        border-radius: 12px;
        padding: 2.5rem;
        text-align: center;
        margin-bottom: 2rem;
        background-color: #F5F3FF;
    }
    .upload-area h3 {
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
        color: #4F46E5;
    }
    /* Better section division */
    .section-divider {
        height: 4px;
        background: linear-gradient(90deg, #4F46E5, transparent);
        margin: 1.5rem 0;
        border-radius: 4px;
    }
    /* Better form inputs */
    .stTextInput>div>div>input, .stSelectbox>div>div>select {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    /* Mobile responsive improvements */
    @media (max-width: 768px) {
        .main {
            padding: 1rem !important;
        }
        .custom-header {
            padding: 1.5rem 1rem;
            margin-bottom: 2rem;
        }
        .custom-header h1 {
            font-size: 1.8rem;
        }
    }
    /* Fix for select boxes */
    .stSelectbox label, .stTextInput label {
        font-size: 1rem;
        font-weight: 500;
        color: #4B5563;
        margin-bottom: 0.5rem;
    }
    /* Better spacing for containers */
    [data-testid="stVerticalBlock"] {
        gap: 1.5rem;
    }
    /* Stats Card */
    .stats-card {
        background-color: #EEF2FF;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.1);
    }
    /* Hide sidebar completely */
    [data-testid="stSidebar"] {
        display: none !important;
    }
    /* Make container wider */
    .block-container {
        max-width: 1400px !important;
        padding-right: 1rem !important;
        padding-left: 1rem !important;
    }
    /* Results section */
    .results-container {
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
        padding: 1.5rem;
        margin-top: 2rem;
    }
    /* Color display box */
    .color-box {
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        margin-top: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    /* Row spacing */
    .row-container {
        margin-bottom: 2rem;
        background-color: white;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
    }
    /* Beautiful color name display */
    .color-name-display {
        margin-top: 12px;
        padding: 10px 15px;
        background: linear-gradient(135deg, rgba(255,255,255,0.8), rgba(255,255,255,0.4));
        border-radius: 20px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        font-weight: 600;
        text-align: center;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.5);
    }
    /* Loading animation under button */
    .loading-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-top: 15px;
    }
    .loading-dots {
        display: flex;
        justify-content: center;
        margin-top: 5px;
    }
    .loading-dots span {
        width: 10px;
        height: 10px;
        margin: 0 5px;
        background-color: #059669;
        border-radius: 50%;
        display: inline-block;
        animation: bounce 1.5s infinite ease-in-out both;
    }
    .loading-dots span:nth-child(1) {
        animation-delay: -0.3s;
    }
    .loading-dots span:nth-child(2) {
        animation-delay: -0.15s;
    }
    @keyframes bounce {
        0%, 80%, 100% { 
            transform: scale(0);
        }
        40% { 
            transform: scale(1.0);
        }
    }
    /* Color panel with gradient border */
    .color-panel {
        background-color: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
        margin-top: 10px;
        position: relative;
    }
    .color-panel::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #4F46E5, #7C3AED, #10B981, #059669);
        border-radius: 14px;
        z-index: -1;
    }
</style>
""", unsafe_allow_html=True)

if not os.path.exists("temp"):
    os.makedirs("temp")

# Initialize session state for color results
if 'original_color' not in st.session_state:
    st.session_state.original_color = None
if 'matched_color' not in st.session_state:
    st.session_state.matched_color = None
if 'matched_image' not in st.session_state:
    st.session_state.matched_image = None
# Add loading state
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False

# Function to call backend for matching
def find_matching_clothes(uploaded_image, category):
    # Save the uploaded image temporarily
    temp_path = "temp/temp_upload.jpg"
    uploaded_image.convert("RGB").save(temp_path)
    with open(temp_path, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/match",
            files={"image": f},
            data={"category": category}
        )
        f.seek(0)
        response2 = requests.post(
            "http://127.0.0.1:8000/color",
            files={"image": f}
        )
    matches = Image.open(io.BytesIO(response.content))
    img_byte_arr = io.BytesIO()
    matches.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    response3 = requests.post(
        "http://127.0.0.1:8000/color",
        files={"image": ("matched_image.jpg", img_byte_arr, "image/jpeg")}
    )
    data = response2.json()
    data2 = response3.json()
    return matches, data["colors"], data2["colors"], data["color_name"], data2["color_name"]

# Category icons mapping
category_icons = {
    "Tops": "👕",
    "Bottoms": "👖",
}

# Custom header
st.markdown("""
<div class="custom-header">
    <h1>🎨 BlindPalette</h1>
    <p>Upload an item and discover complementary pieces from your wardrobe</p>
    <div style="position: absolute; right: 30px; top: 30px;">
        <!-- SVG icon omitted for brevity -->
    </div>
</div>
""", unsafe_allow_html=True)

# Process match button click
def process_match():
    if st.session_state.get('match_button'):
        st.session_state.is_loading = True
        st.rerun()

# ROW 1: Tips and upload section
st.markdown('<div class="row-container">', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1,1,1])

with col1:
    st.markdown("""
    <div class="upload-area">
      <h3>Upload Your Item</h3>
      <p>Select an image of a clothing item you'd like to match</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        label="Upload image",
        type=["jpg", "jpeg", "png"],
        key="main_upload",
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image

with col2:
    st.markdown("""
    <h3>Matching Tips</h3>
    <ul>
      <li>Upload clear images with good lighting</li>
      <li>Try to capture the true color of your item</li>
      <li>For best results, use images with solid backgrounds</li>
      <li>Add more items to your drawer for better matching</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px; padding-top:15px; border-top:1px solid #e2e8f0;">
      <h4>Not sure about colors?</h4>
      <p>Take a quick color blindness test to understand your color perception</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🔍 Take Color Blindness Test", key="color_blind_test"):
        webbrowser.open_new_tab("http://localhost:8081")

    st.markdown("""
    <style>
      [data-testid="stButton"][aria-describedby="color_blind_test"] button {
        background-color: #8B5CF6;
      }
      [data-testid="stButton"][aria-describedby="color_blind_test"] button:hover {
        background-color: #7C3AED;
      }
    </style>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <h3>How It Works</h3>
    <ol>
      <li><strong>Upload</strong> - Add your clothing item photo</li>
      <li><strong>Select</strong> - Choose the correct category</li>
      <li><strong>Match</strong> - Find complementary items</li>
      <li><strong>Style</strong> - Get outfit recommendations</li>
    </ol>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ROW 2: After upload
if 'uploaded_image' in st.session_state:
    st.markdown('<div class="row-container">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1,1])

    with c1:
        st.image(st.session_state.uploaded_image, use_container_width=True)

    with c2:
        category = st.selectbox(
            "Item Category",
            options=["-- Select a category --"] + list(category_icons),
            format_func=lambda x: f"{category_icons.get(x,'❓')} {x}" if x!="-- Select a category --" else x,
            key="category_select"
        )
        if st.session_state.get('original_color'):
            r,g,b = st.session_state.original_color[0]
            st.markdown(f"""
            <div class="color-panel">
              <div style="height:80px; background-color:rgb({r},{g},{b});"></div>
              <div class="color-name-display">{st.session_state.original_color_name}</div>
            </div>
            """, unsafe_allow_html=True)

    with c3:
        if st.session_state.category_select!="-- Select a category --":
            if st.button("✨ Find Matching Items", key="match_button", on_click=process_match):
                pass
            if st.session_state.is_loading:
                st.markdown("<p>Finding your perfect matches...</p>", unsafe_allow_html=True)
                matches, c1, c2, n1, n2 = find_matching_clothes(
                    st.session_state.uploaded_image,
                    st.session_state.category_select
                )
                st.session_state.matched_image = matches
                st.session_state.original_color = c1
                st.session_state.matched_color = c2
                st.session_state.original_color_name = n1
                st.session_state.matched_color_name = n2
                st.session_state.is_loading = False
                st.rerun()
        else:
            st.warning("Please select a category")

    st.markdown('</div>', unsafe_allow_html=True)

# ROW 3: Results
if st.session_state.matched_image is not None:
    st.markdown('<div class="row-container">', unsafe_allow_html=True)
    d1,d2,d3 = st.columns([1,1,1])

    with d1:
        st.image(st.session_state.matched_image, use_container_width=True)
    with d2:
        r2,g2,b2 = st.session_state.matched_color[0]
        st.markdown(f"""
        <div class="color-panel">
          <div style="height:80px; background-color:rgb({r2},{g2},{b2});"></div>
          <div class="color-name-display">{st.session_state.matched_color_name}</div>
        </div>
        """, unsafe_allow_html=True)
    with d3:
        st.markdown("""
        <div style="padding:1rem; background:#F0FDF4; border-left:4px solid #10B981;">
          <h4>Perfect Match!</h4>
          <p>These items create a harmonious outfit.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
