import streamlit as st
import tensorflow as tf
import numpy as np

# TensorFlow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page - Home
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM 🔍")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    # 🌿 Welcome to My Plant Disease Recognition System! 🔍  
   
    Hi there! 👋 I'm here to help you identify plant diseases quickly and accurately. Just upload an image of your plant, and my system will analyze it to detect any signs of disease. Let’s work together to protect your crops and ensure a healthier harvest! 🌱💚  

    ## 🌟 How It Works  
    🔹 **Upload Image**: Navigate to the **Disease Recognition** page and upload a clear image of the affected plant.  
    🔹 **AI Analysis**: My AI-powered system scans the image and detects potential diseases.  
    🔹 **Instant Results**: Receive a diagnosis along with expert recommendations for treatment.  

    ## 💡 Why Choose This System?  
    ✅ **Highly Accurate** – Uses advanced machine learning for precise disease detection.  
    ✅ **User-Friendly** – A simple and intuitive interface—just upload an image and get results!  
    ✅ **Fast & Efficient** – Receive instant insights so you can take action without delay.  

    ## 🌍 Supported Plant Diseases  
    This system can detect a wide range of plant diseases across various crops, including:  

    **🌱 Cassava:**  
    - Bacterial Blight  
    - Brown Streak Disease  
    - Green Mottle  
    - Mosaic Disease  
    - **Healthy** ✅  

    **🌶️ Chili:**  
    - Leaf Curl  
    - Leaf Spot  
    - Whitefly  
    - Yellowish  
    - **Healthy** ✅  

    **🌽 Corn:**  
    - Common Rust  
    - Gray Leaf Spot  
    - Northern Leaf Blight  
    - **Healthy** ✅  

    **🥒 Cucumber:**  
    - Diseased  
    - **Healthy** ✅  

    **🍐 Guava:**  
    - Diseased  
    - **Healthy** ✅  

    **🍇 Grapes:**  
    - Black Measles  
    - Black Rot  
    - Leaf Blight (Isariopsis Leaf Spot)  
    - **Healthy** ✅  

    **🍈 Jamun:**  
    - Diseased  
    - **Healthy** ✅  

    **🍋 Lemon:**  
    - Diseased  
    - **Healthy** ✅  

    **🥭 Mango:**  
    - Diseased  
    - **Healthy** ✅  

    **🫑 Pepper Bell:**  
    - Bacterial Spot  
    - **Healthy** ✅  

    **🥔 Potato:**  
    - Early Blight  
    - Late Blight  
    - **Healthy** ✅  

    **🌾 Rice:**  
    - Brown Spot  
    - Hispa  
    - Leaf Blast  
    - Neck Blast  
    - **Healthy** ✅  

    **🍅 Tomato:**  
    - Bacterial Spot  
    - Early Blight  
    - Late Blight  
    - Leaf Mold  
    - Mosaic Virus  
    - Septoria Leaf Spot  
    - Spider Mites (Two-Spotted Spider Mite)  
    - Target Spot  
    - Yellow Leaf Curl Virus  
    - **Healthy** ✅  

    ## 🚀 Get Started  
    Click on the **Disease Recognition** page in the sidebar to upload an image and start detecting plant diseases effortlessly!  

    ## 📌 About Me  
    Learn more about this project and my mission to help farmers and gardeners on the **About** page.  
""")


# About Page
elif app_mode == "About":
    st.header("🌱 About This Project")  
    st.markdown("""
    Hi there! 👋 I'm excited to share my **Plant Disease Recognition System** with you.  
    As someone passionate about technology and agriculture, I built this tool to help farmers, gardeners, and plant lovers detect diseases early and take action quickly.  

    ### 🌟 Why I Created This  
    - **Early Detection Saves Crops** – Spot diseases before they spread.  
    - **AI-Powered Accuracy** – My system uses deep learning to analyze plant images.  
    - **Fast & Easy to Use** – Just upload an image, and I’ll give you instant results!  

    ### 🚀 My Mission  
    I want to make plant disease detection simple and accessible for everyone, ensuring healthier crops and better yields.  

    ### 🔍 How You Can Use It  
    1. Go to the **Disease Recognition** page.  
    2. Upload a clear image of your plant.  
    3. Get instant results along with expert recommendations.  

    Let’s work together to keep plants healthy and thriving! 🌍🌱  
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown("## 🌿 Plant Disease Recognition 🔍")  
    st.markdown("### Upload an Image to Detect Plant Diseases Instantly! 📸")  

    # Upload Image  
    test_image = st.file_uploader("📤 **Choose a Plant Image:**", type=["jpg", "png", "jpeg"])  

    if test_image:  
        st.success("✅ Image Uploaded Successfully!")  

        # Show Image  
        if st.button("🖼 Show Image"):  
            st.image(test_image, use_column_width=True, caption="Uploaded Image")  

        # Define class names before prediction
        class_names = [
            'Cassava__bacterial_blight', 'Cassava__brown_streak_disease', 'Cassava__green_mottle', 'Cassava__healthy',
            'Cassava__mosaic_disease', 'Chili__healthy', 'Chili__leaf curl', 'Chili__leaf spot', 'Chili__whitefly',
            'Chili__yellowish', 'Corn__common_rust', 'Corn__gray_leaf_spot', 'Corn__healthy', 'Corn__northern_leaf_blight',
            'Cucumber__diseased', 'Cucumber__healthy', 'Gauva__diseased', 'Gauva__healthy', 'Grape__black_measles',
            'Grape__black_rot', 'Grape__healthy', 'Grape__leaf_blight_(isariopsis_leaf_spot)', 'Jamun__diseased',
            'Jamun__healthy', 'Lemon__diseased', 'Lemon__healthy', 'Mango__diseased', 'Mango__healthy', 'Pepper_bell__bacterial_spot',
            'Pepper_bell__healthy', 'Potato__early_blight', 'Potato__healthy', 'Potato__late_blight', 'Rice__brown_spot',
            'Rice__healthy', 'Rice__hispa', 'Rice__leaf_blast', 'Rice__neck_blast', 'Tomato__bacterial_spot', 'Tomato__early_blight',
            'Tomato__healthy', 'Tomato__late_blight', 'Tomato__leaf_mold', 'Tomato__mosaic_virus', 'Tomato__septoria_leaf_spot',
            'Tomato__spider_mites_(two_spotted_spider_mite)', 'Tomato__target_spot', 'Tomato__yellow_leaf_curl_virus'
        ]

        # Predict Disease  
        if st.button("🔍 Predict Disease"):  
            with st.spinner("⏳ Analyzing the image... Please wait!"):  
                result_index = model_prediction(test_image)  

            # Fun animation  
            st.snow()  

            # Display prediction result  
            detected_disease = class_names[result_index]
            st.markdown("### 🌟 Prediction Result:")  
            st.success(f"🩺 **Detected Disease:** {detected_disease}")  

            # Disease solutions dictionary
            disease_solutions = [
    {"disease": "Cassava__bacterial_blight", "solution": "Use disease-free planting material, practice crop rotation, and apply copper-based fungicides."},
    {"disease": "Cassava__brown_streak_disease", "solution": "Plant resistant varieties, control whiteflies, and remove infected plants."},
    {"disease": "Cassava__green_mottle", "solution": "Use virus-free cuttings, control insect vectors, and practice field sanitation."},
    {"disease": "Cassava__mosaic_disease", "solution": "Use resistant varieties, control whiteflies, and remove diseased plants."},
    {"disease": "Chili__leaf curl", "solution": "Control whiteflies with neem oil or insecticides, and plant resistant varieties."},
    {"disease": "Chili__leaf spot", "solution": "Apply copper-based fungicides and avoid overhead irrigation."},
    {"disease": "Chili__whitefly", "solution": "Use yellow sticky traps and apply insecticidal soap or neem oil."},
    {"disease": "Chili__yellowish", "solution": "Ensure proper fertilization, manage pests, and improve soil drainage."},
    {"disease": "Corn__common_rust", "solution": "Plant resistant varieties and apply fungicides when necessary."},
    {"disease": "Corn__gray_leaf_spot", "solution": "Rotate crops, improve air circulation, and apply fungicides."},
    {"disease": "Corn__northern_leaf_blight", "solution": "Use resistant hybrids and apply foliar fungicides."},
    {"disease": "Cucumber__diseased", "solution": "Apply appropriate fungicides, control pests, and practice crop rotation."},
    {"disease": "Gauva__diseased", "solution": "Prune affected areas, apply fungicides, and improve air circulation."},
    {"disease": "Grape__black_measles", "solution": "Prune infected vines, apply fungicides, and improve vineyard sanitation."},
    {"disease": "Grape__black_rot", "solution": "Remove infected leaves, apply fungicides, and ensure proper vine spacing."},
    {"disease": "Grape__leaf_blight_(isariopsis_leaf_spot)", "solution": "Use resistant varieties, remove infected leaves, and apply fungicides."},
    {"disease": "Jamun__diseased", "solution": "Prune affected areas, apply organic fungicides, and ensure good soil health."},
    {"disease": "Lemon__diseased", "solution": "Use copper-based fungicides, prune infected branches, and ensure proper irrigation."},
    {"disease": "Mango__diseased", "solution": "Apply copper-based fungicides, improve orchard sanitation, and prune infected parts."},
    {"disease": "Pepper_bell__bacterial_spot", "solution": "Use resistant varieties, avoid overhead watering, and apply copper-based sprays."},
    {"disease": "Potato__early_blight", "solution": "Use certified disease-free seeds, rotate crops, and apply fungicides."},
    {"disease": "Potato__late_blight", "solution": "Apply fungicides like Mancozeb or Metalaxyl and remove infected plants."},
    {"disease": "Rice__brown_spot", "solution": "Use resistant varieties, apply balanced fertilization, and ensure proper water management."},
    {"disease": "Rice__hispa", "solution": "Handpick larvae, apply neem-based insecticides, and introduce biological control agents."},
    {"disease": "Rice__leaf_blast", "solution": "Use resistant varieties, avoid excessive nitrogen, and apply fungicides."},
    {"disease": "Rice__neck_blast", "solution": "Apply fungicides like Tricyclazole, use resistant varieties, and ensure proper field sanitation."},
    {"disease": "Tomato__bacterial_spot", "solution": "Use disease-free seeds, apply copper-based sprays, and avoid overhead irrigation."},
    {"disease": "Tomato__early_blight", "solution": "Rotate crops, remove infected leaves, and use fungicides like Chlorothalonil."},
    {"disease": "Tomato__late_blight", "solution": "Apply fungicides such as Mancozeb and remove infected plants promptly."},
    {"disease": "Tomato__leaf_mold", "solution": "Increase air circulation, avoid excessive humidity, and apply fungicides."},
    {"disease": "Tomato__mosaic_virus", "solution": "Use virus-free seeds, control aphids, and remove infected plants."},
    {"disease": "Tomato__septoria_leaf_spot", "solution": "Use copper fungicides, practice crop rotation, and ensure good air circulation."},
    {"disease": "Tomato__spider_mites_(two_spotted_spider_mite)", "solution": "Use neem oil, introduce predatory mites, and maintain proper watering."},
    {"disease": "Tomato__target_spot", "solution": "Apply fungicides, avoid overhead watering, and remove infected leaves."},
    {"disease": "Tomato__yellow_leaf_curl_virus", "solution": "Control whiteflies, plant resistant varieties, and remove infected plants."}]


            # Convert list to dictionary
            disease_dict = {item["disease"]: item["solution"] for item in disease_solutions}

# Display prediction result  
            detected_disease = class_names[result_index]
            st.markdown("### 🌟 Prediction Result:")  
            st.success(f"🩺 **Detected Disease:** {detected_disease}")  

# Display Solution  
            solution = disease_dict.get(detected_disease, "No solution found. Please consult an expert.")
            st.success(f"💡 **Solution:** {solution}")  

