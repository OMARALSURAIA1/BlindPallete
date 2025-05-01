

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

# Set page configuration
st.set_page_config(
    page_title="Clothing Matcher",
    page_icon="👔",
    layout="wide",
    initial_sidebar_state="expanded"
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
    
    /* Delete button styling */
    .delete-btn>button {
        background-color: #DC2626;
    }
    
    .delete-btn>button:hover {
        background-color: #B91C1C;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2);
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
        margin-bottom: 3rem;
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
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4F46E5 !important;
        color: white !important;
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
    
    /* Enhanced sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        box-shadow: 2px 0 5px rgba(0, 0, 0, 0.05);
        padding: 2rem 1rem;
    }
    
    /* Fixing the sidebar width */
    [data-testid="stSidebar"] {
        min-width: 250px !important;
        max-width: 300px !important;
    }
    
    /* Better section division */
    .section-divider {
        height: 4px;
        background: linear-gradient(90deg, #4F46E5, transparent);
        margin: 3rem 0 2rem;
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
</style>
""", unsafe_allow_html=True)

# Create directories if they don't exist
if not os.path.exists("clothing_db"):
    os.makedirs("clothing_db")

if not os.path.exists("temp"):
    os.makedirs("temp")

# Initialize session state for database
if 'clothing_db' not in st.session_state:
    if os.path.exists("clothing_db/database.json"):
        with open("clothing_db/database.json", "r") as f:
            st.session_state.clothing_db = json.load(f)
    else:
        st.session_state.clothing_db = []

# Function to save current database to file
def save_database():
    with open("clothing_db/database.json", "w") as f:
        json.dump(st.session_state.clothing_db, f)

# Function to add item to database
def add_to_database(image, category, description, color):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"clothing_db/{timestamp}.jpg"
    
    # Save the image
    image.save(filename)
    
    # Add entry to database
    item = {
        "id": timestamp,
        "path": filename,
        "category": category,
        "description": description,
        "color": color,
        # "season": season,
        "added_date": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.session_state.clothing_db.append(item)
    save_database()
    return item

# Function to delete item from database
def delete_from_database(item_id):
    for i, item in enumerate(st.session_state.clothing_db):
        if item["id"] == item_id:
            # Remove the file
            if os.path.exists(item["path"]):
                os.remove(item["path"])
            
            # Remove from database
            st.session_state.clothing_db.pop(i)
            save_database()
            break

# Function to convert image to base64 for display
def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Function to call backend for matching
def find_matching_clothes(uploaded_image):
    # This is where you would call your backend API
    # For demonstration, we'll just return a mock response with random matches
    
    # Save the uploaded image temporarily
    temp_path = "temp/temp_upload.jpg"
    uploaded_image.convert("RGB").save(temp_path)
    
    # In a real application, you would send this to your backend:
    # response = requests.post(
    #     "https://your-backend-api.com/match",
    #     files={"image": open(temp_path, "rb")}
    # )
    # matches = response.json()
    with open(temp_path, "rb") as f:
        response = requests.post("http://127.0.0.1:8000/match", files={"image": f})
        f.seek(0)
        response2 = requests.post("http://127.0.0.1:8000/color", files={"image": f})

    # Load returned image 
    # data = response.json()
    matches = Image.open(io.BytesIO(response.content))
    color = response2.content
    # print(color)

    # # For demo purposes, just select 3 random items from the database
    # if len(st.session_state.clothing_db) > 0:
    #     num_matches = min(3, len(st.session_state.clothing_db))
    #     matches = random.sample(st.session_state.clothing_db, num_matches)
    # else:
    #     matches = []
    
    # Simulate a loading delay
    time.sleep(1.5)
    
    return matches,color

# Category icons mapping
category_icons = {
    "Tops": "👕",
    "Bottoms": "👖",
    # "Dresses": "👗",
    # "Outerwear": "🧥",
    # "Shoes": "👞",
    # "Accessories": "👔"
}

# Color options with hex codes for visual display
color_options = {
    "Black": "#000000",
    "White": "#FFFFFF",
    "Red": "#DC2626",
    "Blue": "#2563EB",
    "Green": "#10B981",
    "Yellow": "#FBBF24",
    "Purple": "#8B5CF6",
    "Pink": "#EC4899",
    "Gray": "#6B7280",
    "Brown": "#92400E",
    "Orange": "#F97316",
    "Multicolor": "linear-gradient(90deg, red, orange, yellow, green, blue, indigo, violet)"
}

# Season options
# season_options = ["Spring", "Summer", "Fall", "Winter", "All Seasons"]

# App navigation in sidebar with custom icons
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem 0; margin-bottom: 2rem; border-bottom: 1px solid #E5E7EB;">
    <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f454.svg" width="50" style="margin-bottom: 1rem;">
    <h1 style="font-size: 1.8rem; font-weight: bold; margin-bottom: 0.5rem; background: linear-gradient(90deg, #4F46E5, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        Clothing Matcher
    </h1>
    <p style="color: #6B7280; font-size: 0.9rem;">Your personal style assistant</p>
</div>
""", unsafe_allow_html=True)

# Navigation options
nav_options = {
    "Main Page": "Find matches for your clothing items",
    "Clothing Drawer": "Manage your wardrobe collection"
}

# Create radio buttons for navigation
page = st.sidebar.radio("Navigation", list(nav_options.keys()))

# Stats in sidebar with better styling
st.sidebar.markdown("""
<div style="background-color: #EEF2FF; padding: 1.5rem; border-radius: 12px; margin-top: 2.5rem; box-shadow: 0 4px 12px rgba(79, 70, 229, 0.1);">
    <h3 style="font-size: 1.1rem; color: #4F46E5; margin-bottom: 1rem; font-weight: 600;">Wardrobe Stats</h3>
""", unsafe_allow_html=True)

total_items = len(st.session_state.clothing_db)
st.sidebar.markdown(f"**Total Items:** {total_items}")

if total_items > 0:
    # Count items by category
    category_counts = {}
    for item in st.session_state.clothing_db:
        category = item["category"]
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1
    
    # Display category breakdown
    for category, count in category_counts.items():
        icon = category_icons.get(category, "📦")
        st.sidebar.markdown(f"{icon} **{category}:** {count}")

st.sidebar.markdown("</div>", unsafe_allow_html=True)

# # Footer in sidebar with improved styling
# st.sidebar.markdown("""
# <div style="position: absolute; bottom: 30px; left: 20px; right: 20px; text-align: center; padding: 1rem; color: #6B7280; font-size: 0.85rem; background-color: #F8FAFC; border-top: 1px solid #E5E7EB; border-radius: 12px;">
#     <p style="margin-bottom: 0.5rem;">© 2025 Clothing Matcher</p>
#     <p style="margin-bottom: 0;">Your Personal Style Assistant</p>
# </div>
# """, unsafe_allow_html=True)







if page == "Main Page":
    # Custom header with enhanced design
    st.markdown("""
    <div class="custom-header">
        <h1>Find Your Perfect Match</h1>
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
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="upload-area">
            <h3>Upload Your Item</h3>
            <p>Select an image of a clothing item you'd like to match</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="main_upload")
        
        if uploaded_file is not None:
            # Display the uploaded image in a card-like container
            st.markdown("""
            <div style="background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 1rem;">
                <h3 style="margin-bottom: 1rem;">Your Item</h3>
            """, unsafe_allow_html=True)
            
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            # Match button with custom styling
            st.markdown('<div class="match-btn">', unsafe_allow_html=True)
            match_button = st.button("✨ Find Matching Items", key="match_button")
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); height: 100%;">
            <h3>Matching Tips</h3>
            <ul style="margin-top: 1rem; color: #4B5563;">
                <li>Upload clear images with good lighting</li>
                <li>Try to capture the true color of your item</li>
                <li>For best results, use images with solid backgrounds</li>
                <li>Add more items to your drawer for better matching</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Results section
    if uploaded_file is not None and 'match_button' in st.session_state and st.session_state.match_button:
        # Show loading animation
        with st.spinner("Finding your perfect matches..."):
            matches,color = find_matching_clothes(image)
        
        # Display results
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        if matches:
            # st.markdown("""
            # <h2 style="text-align: center; margin-bottom: 2rem;">🎯 Your Perfect Matches</h2>
            # """, unsafe_allow_html=True)
            
            # cols = st.columns(min(3, len(matches)))
            # for i, match in enumerate(matches):
            #     with cols[i % len(cols)]:
            #         st.markdown(f"""
            #         <div class="clothing-card">
            #             <h4>{match["description"]}</h4>
            #         """, unsafe_allow_html=True)
                    
            #         st.image(match["path"], use_container_width=True)
            #         st.markdown(f"""
            #             <div style="display: flex; align-items: center; margin-top: 0.5rem;">
            #                 <div class="category-badge">{category_icons.get(match["category"], "📦")} {match["category"]}</div>
            #                 <div style="margin-left: auto; font-size: 0.8rem; color: #6B7280;">{match.get("season", "All Seasons")}</div>
            #             </div>
                        
            #             <div style="display: flex; align-items: center; margin-top: 0.5rem;">
            #                 <div style="background-color: {color_options.get(match.get("color", "Black"), "#000000")}; 
            #                     width: 15px; height: 15px; border-radius: 50%; margin-right: 5px;"></div>
            #                 <span style="font-size: 0.8rem;">{match.get("color", "Black")}</span>
            #             </div>
            #         """, unsafe_allow_html=True)
                    
            #         st.markdown("</div>", unsafe_allow_html=True)
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
            st.markdown("""
            <h2 style="text-align: center; margin-bottom: 2rem;">🎯 Your Processed Image</h2>
            """, unsafe_allow_html=True)

            st.image(matches, use_container_width=True)

            import webcolors

            def closest_color(requested_color):
                min_colors = {}
                for hex_code, name in webcolors.CSS3_HEX_TO_NAMES.items():
                    r_c, g_c, b_c = webcolors.hex_to_rgb(hex_code)
                    rd = (r_c - requested_color[0]) ** 2
                    gd = (g_c - requested_color[1]) ** 2
                    bd = (b_c - requested_color[2]) ** 2
                    min_colors[(rd + gd + bd)] = name
                return min_colors[min(min_colors.keys())]
            rgb = json.loads(color.decode())
            # st.write(rgb)
            # Unpack RGB
            if rgb:
                r, g, b = rgb[0]

                # Create and display image
                img = Image.new("RGB", (100, 100), (r, g, b))
                st.image(img, caption=f"RGB: ({r}, {g}, {b})")
                rgb_str = rgb.decode("utf-8")
                import ast
                # Step 2: Convert string to list
                rgbt = ast.literal_eval(rgb_str)  # Now rgb = [120, 200, 150]
                color_name = closest_color(rgbt)
                st.write(f"This color is closest to: **{color_name}**")
            else:
                st.write("no color found")
            st.markdown("""
            <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-top: 2rem;">
                <h3 style="margin-bottom: 1rem;">🌟 Style Advice</h3>
                <p>This result has been generated by the model. Consider pairing it with complementary colors or styles based on your preferences!</p>
            </div>
            """, unsafe_allow_html=True)
            # Style advice
            st.markdown("""
            <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-top: 2rem;">
                <h3 style="margin-bottom: 1rem;">🌟 Style Advice</h3>
                <p>These items were selected based on color harmony, style compatibility, and seasonal appropriateness. 
                   Try pairing them with your uploaded item for a perfectly coordinated look!</p>
            </div>
            """, unsafe_allow_html=True)
            
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 0;">
                <h3>No matches found</h3>
                <p style="color: #6B7280;">Add more clothes to your drawer for better matching results!</p>
                <a href="#" style="display: inline-block; background-color: #4F46E5; color: white; padding: 0.5rem 1rem; 
                   border-radius: 8px; text-decoration: none; margin-top: 1rem;">Add Items Now</a>
            </div>
            """, unsafe_allow_html=True)








elif page == "Clothing Drawer":
    # Custom header with enhanced design
    st.markdown("""
    <div class="custom-header">
        <h1>Your Clothing Drawer</h1>
        <p>Manage your wardrobe collection</p>
        <div style="position: absolute; right: 30px; top: 30px;">
            <svg width="80" height="80" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                <rect x="25" y="25" width="50" height="60" rx="5" stroke="rgba(255,255,255,0.6)" stroke-width="3"/>
                <line x1="25" y1="45" x2="75" y2="45" stroke="rgba(255,255,white,0.6)" stroke-width="3"/>
                <line x1="25" y1="65" x2="75" y2="65" stroke="rgba(255,255,white,0.6)" stroke-width="3"/>
                <circle cx="37" cy="35" r="3" fill="rgba(255,255,white,0.6)"/>
                <circle cx="37" cy="55" r="3" fill="rgba(255,255,white,0.6)"/>
                <circle cx="37" cy="75" r="3" fill="rgba(255,255,white,0.6)"/>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for better organization
    tab1, tab2 = st.tabs(["📥 Add New Items", "👕 View Collection"])
    
    with tab1:
        # Add new clothing item section
        st.markdown("""
        <h3 style="margin-bottom: 1rem;">Add a New Clothing Item</h3>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.markdown("""
            <div class="upload-area">
                <h3>Upload Image</h3>
                <p>Select a clear image of your clothing item</p>
            </div>
            """, unsafe_allow_html=True)
            
            new_item = st.file_uploader("", type=["jpg", "jpeg", "png"], key="drawer_upload")
                
        with col2:
            if new_item is not None:
                image = Image.open(new_item)
                st.image(image, caption="Preview", use_container_width=True)
        
        # Item details form
        if new_item is not None:
            st.markdown("""
            <div style="background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-top: 1rem;">
                <h3 style="margin-bottom: 1rem;">Item Details</h3>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                category = st.selectbox("Category", list(category_icons.keys()), format_func=lambda x: f"{category_icons[x]} {x}")
                description = st.text_input("Description", placeholder="E.g., Blue striped shirt")
            
            with col2:
                color = st.selectbox("Primary Color", list(color_options.keys()))
                # season = st.selectbox("Season", season_options)
            
            # Color preview
            color_hex = color_options[color]
            st.markdown(f"""
            <div style="display: flex; align-items: center; margin-top: 0.5rem;">
                <div style="background: {color_hex}; width: 30px; height: 30px; border-radius: 50%; margin-right: 10px; border: 1px solid #E5E7EB;"></div>
                <span>Selected color: {color}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Add button with custom styling
            st.markdown('<div style="margin-top: 1.5rem;">', unsafe_allow_html=True)
            if st.button("➕ Add to My Collection", key="add_item"):
                item = add_to_database(image, category, description, color)
                st.success(f"Added {description} to your collection!")
                time.sleep(1)  # Show success message briefly
                st.experimental_rerun()
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        # Filter options with more visual approach
        st.markdown("""
        <div style="background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-bottom: 1.5rem;">
            <h3 style="margin-bottom: 1rem;">Filter Your Collection</h3>
        """, unsafe_allow_html=True)
        
        col1, col3 = st.columns(2)
        
        with col1:
            filter_category = st.selectbox(
                "Category", 
                ["All"] + list(category_icons.keys()),
                format_func=lambda x: x if x == "All" else f"{category_icons[x]} {x}",
                key="filter_category"
            )
        
        # with col2:
        #     filter_season = st.selectbox("Season", ["All"] + season_options, key="filter_season")
        
        with col3:
            filter_color = st.selectbox("Color", ["All"] + list(color_options.keys()), key="filter_color")
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Apply filters
        filtered_items = st.session_state.clothing_db
        
        if filter_category != "All":
            filtered_items = [item for item in filtered_items if item["category"] == filter_category]
        
        # if filter_season != "All":
        #     filtered_items = [item for item in filtered_items if item.get("season", "All Seasons") == filter_season]
        
        if filter_color != "All":
            filtered_items = [item for item in filtered_items if item.get("color", "") == filter_color]
        
        # Collection stats
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <p>Showing {len(filtered_items)} of {len(st.session_state.clothing_db)} items</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Display items in a grid with improved styling
        if filtered_items:
            # Use 4 columns for display on wide screens
            num_cols = 4
            rows = (len(filtered_items) + num_cols - 1) // num_cols  # Ceiling division
            
            for row in range(rows):
                cols = st.columns(num_cols)
                for col in range(num_cols):
                    idx = row * num_cols + col
                    if idx < len(filtered_items):
                        item = filtered_items[idx]
                        with cols[col]:
                            st.markdown(f"""
                            <div class="clothing-card">
                                <h4>{item["description"]}</h4>
                            """, unsafe_allow_html=True)
                            
                            st.image(item["path"], use_container_width=True)
                            
                            # Item details
                            st.markdown(f"""
                                <div style="display: flex; align-items: center; margin-top: 0.5rem;">
                                    <div class="category-badge">{category_icons.get(item["category"], "📦")} {item["category"]}</div>
                                    <div style="margin-left: auto; font-size: 0.8rem; color: #6B7280;">{item.get("season", "All Seasons")}</div>
                                </div>
                                
                                <div style="display: flex; align-items: center; margin-top: 0.5rem;">
                                    <div style="background-color: {color_options.get(item.get("color", "Black"), "#000000")}; 
                                        width: 15px; height: 15px; border-radius: 50%; margin-right: 5px;"></div>
                                    <span style="font-size: 0.8rem;">{item.get("color", "Black")}</span>
                                    <span style="margin-left: auto; font-size: 0.8rem; color: #6B7280;">Added: {item["added_date"].split()[0]}</span>
                                </div>
                                
                                <div style="margin-top: 1rem;">
                            """, unsafe_allow_html=True)
                            
                            # Delete button with custom styling
                            st.markdown('<div class="delete-btn">', unsafe_allow_html=True)
                            if st.button(f"🗑 Delete", key=f"delete_{item['id']}"):
                                delete_from_database(item["id"])
                                st.success("Item removed from your collection!")
                                time.sleep(1)  # Show success message briefly
                                st.experimental_rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown("</div></div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align: center; padding: 3rem 0; background-color: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
                <img src="https://cdn.jsdelivr.net/gh/twitter/twemoji@14.0.2/assets/svg/1f4ce.svg" width="60" style="margin-bottom: 1rem;">
                <h3>No items found</h3>
                <p style="color: #6B7280; margin-bottom: 1rem;">Try changing your filters or add new items to start building your collection</p>
                <p>Switch to the "Add New Items" tab to expand your wardrobe</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Pagination (for future implementation with larger wardrobes)
        if len(filtered_items) > 20:
            st.markdown("""
            <div style="display: flex; justify-content: center; margin-top: 2rem;">
                <div style="display: flex; border: 1px solid #E5E7EB; border-radius: 8px; overflow: hidden;">
                    <button style="padding: 0.5rem 1rem; background-color: #F3F4F6; border: none; cursor: pointer;">Previous</button>
                    <button style="padding: 0.5rem 1rem; background-color: #4F46E5; color: white; border: none; cursor: pointer;">1</button>
                    <button style="padding: 0.5rem 1rem; background-color: #F3F4F6; border: none; cursor: pointer;">2</button>
                    <button style="padding: 0.5rem 1rem; background-color: #F3F4F6; border: none; cursor: pointer;">3</button>
                    <button style="padding: 0.5rem 1rem; background-color: #F3F4F6; border: none; cursor: pointer;">Next</button>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =----------------------===========================----------------------------------======================----------------------------=

