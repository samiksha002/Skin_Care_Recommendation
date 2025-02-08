import cv2
import os
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from collections import Counter
import pandas as pd
from recommender import SkinCareRecommender  # Import the recommendation system

class CLAHEPreprocessor:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def apply_clahe(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced_lab = cv2.merge((l, a, b))
        enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        return enhanced_img

class YOLOProcessor:
    def __init__(self, model_path, df):
        self.model = YOLO(model_path)
        self.preprocessor = CLAHEPreprocessor()
        self.df = df
        self.recommender = SkinCareRecommender(df)
        self.severity_map = {"Low": 1, "Medium": 2, "High": 3}

    def draw_predictions(self, image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        enhanced_img = self.preprocessor.apply_clahe(image)
        
        results = self.model(enhanced_img)
        predictions = []
        
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    class_name = result.names[class_id]
                    
                    cv2.rectangle(image, (x1, y1), (x2, y2), (25, 0, 255), 2)
                    
                    label = f"{class_name} ({confidence:.2f})"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 60, 15),1)
                    
                    predictions.append(class_name)
        
        detection_counts = Counter(predictions)
        summary = ", ".join([f"{count} {cls}" for cls, count in detection_counts.items()])
        
        if not detection_counts:
            severity_level, overall_score = None, None
            recommendations, warning_message = pd.DataFrame(), "No detections found."
        else:
            severity_level, overall_score = self.calculate_severity(detection_counts)
            recommendations, warning_message = self.get_recommendations(detection_counts, severity_level)
        
        print(f"Overall Score: {overall_score}")
        print(f"Input to get_recommendations: Skin Concerns={list(detection_counts.keys())}, Severity={severity_level}")
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), summary, severity_level, recommendations, warning_message

    def calculate_severity(self, detected_issues):
        total_severity = 0
        total_count = 0
    
        base_severity_map = {
            "blackheads": 1,
            "papules": 2,
            "nodules": 3,
            "pustules": 4,
            "dark spots": 2,
            "whiteheads": 1,
        }
    
        for issue, count in detected_issues.items():
            base_severity = base_severity_map.get(issue, 0)
            adjusted_severity = base_severity + (count - 1) * 0.5
            total_severity += adjusted_severity * count
            total_count += count
    
        overall_score = total_severity / total_count if total_count > 0 else 0
    
        if overall_score >= 3:
            return "High", overall_score
        elif overall_score >= 2:
            return "Medium", overall_score
        else:
            return "Low", overall_score

    def get_recommendations(self, detected_issues, severity):
        skin_concerns = list(detected_issues.keys())
    
        recommendations = self.recommender.get_recommendations(skin_concern=skin_concerns, severity=severity, price_range=(0, 1000), top_n=5)
    
        warning_message = ""
        if severity == "High":
            warning_message = "Your skin concerns are severe. It is recommended to consult a dermatologist before trying any skincare products."
    
        return recommendations, warning_message

# Load dataset
df = pd.read_csv(r'C:\Users\Samiksha Bhatia\Acne_gpu\myvenv\SkinCare_Recommendation_Final\skincare_products_1500_unique.csv')

# Load YOLO Model
model_path = r"C:\Users\Samiksha Bhatia\Acne_gpu\runs\detect\train15\weights\best.pt"
yolo_processor = YOLOProcessor(model_path, df)

# Streamlit UI
st.title("UPLOAD YOUR IMAGE")

uploaded_file = st.file_uploader("Upload an image of your face", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    processed_image, summary, severity, recommendations, warning_message = yolo_processor.draw_predictions(image)
    
    st.image(processed_image, caption="Processed Image with Detections", use_column_width=True)
    
    st.subheader("Detection Results:")
    st.write(summary if summary else "No detections found.")
    
    st.subheader("Overall Severity Level:")
    st.write(severity if severity else "None")
    
    st.subheader("Recommended Products:")
    if not recommendations.empty:
        st.dataframe(recommendations)
    else:
        st.write("No recommendations available.")
    
    if warning_message and warning_message != "No detections found.":
        st.warning(warning_message)