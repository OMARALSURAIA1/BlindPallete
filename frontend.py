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


# Function to convert image to base64 for display
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Function to call backend for matching
def find_matching_clothes(uploaded_image, category):
    # Save the uploaded image temporarily
    temp_path = "temp/temp_upload.jpg"
    uploaded_image.convert("RGB").save(temp_path)
    
    with open(temp_path, "rb") as f:
        response = requests.post("http://127.0.0.1:8000/match",
                                files={"image": f},
                                data={"category": category}  
                                )
        f.seek(0)
        response2 = requests.post("http://127.0.0.1:8000/color", files={"image": f})

    # Load returned image 
    matches = Image.open(io.BytesIO(response.content))
    img_byte_arr = io.BytesIO()
    matches.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    
    response3 = requests.post(
        "http://127.0.0.1:8000/color",
        files={"image": ("matched_image.jpg", img_byte_arr, "image/jpeg")}
    )
    data = response2.json()
    color = data["colors"]
    name = data["color_name"]
    data2 = response3.json()
    color2 = data2["colors"]
    name2 = data2["color_name"]
    print(f"\n\n========{color}\n{name}\n\n{color2}\n{name2}\n\n========")
    return matches, color, color2, name, name2

# Category icons mapping
category_icons = {
    "Tops": "👕",
    "Bottoms": "👖",
}


# Custom header with enhanced design
st.markdown("""
<div class="custom-header">
    <h1>🎨 BlindPalette</h1>
    <p>Upload an item and discover complementary pieces from your wardrobe</p>
    <div style="position: absolute; right: 30px; top: 30px;">
        <svg width="80" height="80" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="45" stroke="rgba(255,255,255,0.2)" stroke-width="2"/>
            <path d="M65,40 C65,40 75,50 65,60" stroke="rgba(255,255,255,0.6)" stroke-width="3" stroke-linecap="round"/>
            <path d="M35,40 C35,40 25,50 35,60" stroke="rgba(255,255,255,0.6)" stroke-width="3" stroke-linecap="round"/>
            <path d="M50,35 C50,35 60,45 50,55 C40,45 50,35 50,35" fill="rgba(255,255,255,0.6)"/>
        </svg>
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
row1_col1, row1_col2, row1_col3 = st.columns([1, 1, 1])

with row1_col1:
    st.markdown("""
    <div class="upload-area">
        <h3>Upload Your Item</h3>
        <p>Select an image of a clothing item you'd like to match</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="main_upload")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image

with row1_col2:
    st.markdown("""
    <h3>Matching Tips</h3>
    <ul style="margin-top: 1rem; color: #4B5563;">
        <li>Upload clear images with good lighting</li>
        <li>Try to capture the true color of your item</li>
        <li>For best results, use images with solid backgrounds</li>
        <li>Add more items to your drawer for better matching</li>
    </ul>
    """, unsafe_allow_html=True)
    
    # Add color blindness test button
    st.markdown("""
    <div style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #e2e8f0;">
        <h4>Not sure about colors?</h4>
        <p style="font-size: 0.9rem; color: #6B7280; margin-bottom: 10px;">
            Take a quick color blindness test to understand your color perception
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Define the button to open EnChroma test in a new tab
    if st.button("🔍 Take Color Blindness Test", key="color_blind_test"):
        webbrowser.open_new_tab("https://enchroma.com/pages/color-blindness-test")
        
    # Add custom styling for this button
    st.markdown("""
    <style>
        [data-testid="stButton"][aria-describedby="color_blind_test"] button {
            background-color: #8B5CF6;
            border-color: #8B5CF6;
        }
        [data-testid="stButton"][aria-describedby="color_blind_test"] button:hover {
            background-color: #7C3AED;
            border-color: #7C3AED;
        }
    </style>
    """, unsafe_allow_html=True)

with row1_col3:
    st.markdown("""
    <h3>How It Works</h3>
    <ol style="margin-top: 1rem; color: #4B5563;">
        <li><strong>Upload</strong> - Add your clothing item photo</li>
        <li><strong>Select</strong> - Choose the correct category</li>
        <li><strong>Match</strong> - Find complementary items</li>
        <li><strong>Style</strong> - Get outfit recommendations</li>
    </ol>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ROW 2: Only show if we have an uploaded file
if uploaded_file is not None:
    st.markdown('<div class="row-container">', unsafe_allow_html=True)
    row2_col1, row2_col2, row2_col3 = st.columns([1, 1, 1])
    
    with row2_col1:
        st.markdown("<h3>Your Item</h3>", unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    
    with row2_col2:
        st.markdown("<h3>Select Category</h3>", unsafe_allow_html=True)
        # Add category selection
        category_options = ["-- Select a category --"] + list(category_icons.keys())
        category = st.selectbox(
            "Item Category",
            options=category_options,
            format_func=lambda x: f"{category_icons.get(x, '❓')} {x}" if x != "-- Select a category --" else x,
            key="category_select"
        )
        
        # Display original color if available after matching
        if st.session_state.get('original_color'):
            st.markdown("<h4>Your Item Color</h4>", unsafe_allow_html=True)
            rgb = st.session_state.original_color
            if rgb:
                r, g, b = rgb[0]
                # Create color panel with gradient border
                st.markdown(f"""
                <div class="color-panel">
                    <div style="height: 80px; background-color: rgb({r}, {g}, {b}); border-radius: 8px;"></div>
                    <div class="color-name-display" style="border-bottom: 3px solid rgb({r}, {g}, {b});">
                        {st.session_state.original_color_name}
                    </div>
                    <div style="text-align: center; margin-top: 8px; color: #6B7280; font-size: 0.9rem;">
                        RGB({r}, {g}, {b})
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with row2_col3:
        st.markdown("<h3>Find Matches</h3>", unsafe_allow_html=True)
        if 'category_select' in st.session_state and st.session_state.category_select != "-- Select a category --":
            st.markdown('<div class="match-btn">', unsafe_allow_html=True)
            match_button = st.button("✨ Find Matching Items", key="match_button", on_click=process_match)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show loading animation under the button
            if st.session_state.is_loading:
                st.markdown("""
                <div class="loading-container">
                    <div style="color: #059669; font-weight: 500;">Finding your perfect matches...</div>
                    <div class="loading-dots">
                        <span></span>
                        <span></span>
                        <span></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Process the matching in the background
                matches, color, color2, name, name2 = find_matching_clothes(
                    st.session_state.uploaded_image, 
                    category=st.session_state.category_select
                )
                st.session_state.matched_image = matches
                st.session_state.original_color = color
                st.session_state.matched_color = color2
                st.session_state.original_color_name = name
                st.session_state.matched_color_name = name2
                st.session_state.is_loading = False
                st.rerun()
        else:
            st.warning("Please select a category to continue")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ROW 3: Display results if matching was performed
if st.session_state.get('matched_image') is not None:
    st.markdown('<div class="row-container">', unsafe_allow_html=True)
    row3_col1, row3_col2, row3_col3 = st.columns([1, 1, 1])
    
    with row3_col1:
        st.markdown("<h3>Matched Item</h3>", unsafe_allow_html=True)
        st.image(st.session_state.matched_image, use_container_width=True)
    
    with row3_col2:
        st.markdown("<h3>Matched Item Color</h3>", unsafe_allow_html=True)
        rgb2 = st.session_state.matched_color
        if rgb2:
            r2, g2, b2 = rgb2[0]
            # Create color panel with gradient border for matched color
            st.markdown(f"""
            <div class="color-panel">
                <div style="height: 80px; background-color: rgb({r2}, {g2}, {b2}); border-radius: 8px;"></div>
                <div class="color-name-display" style="border-bottom: 3px solid rgb({r2}, {g2}, {b2});">
                    {st.session_state.matched_color_name}
                </div>
                <div style="text-align: center; margin-top: 8px; color: #6B7280; font-size: 0.9rem;">
                    RGB({r2}, {g2}, {b2})
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with row3_col3:
        st.markdown("<h3>Style Recommendations</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style="padding: 1rem; background-color: #F0FDF4; border-radius: 8px; border-left: 4px solid #10B981;">
            <h4 style="color: #047857;">Perfect Match!</h4>
            <p style="color: #4B5563;">These items create a harmonious outfit based on color theory principles.</p>
        </div>
        
        <div style="margin-top: 1rem;">
            <p><strong>Styling Tips:</strong></p>
            <ul>
                <li>Add minimal accessories to complete the look</li>
                <li>Consider complementary footwear</li>
                <li>This pairing works well for casual settings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)