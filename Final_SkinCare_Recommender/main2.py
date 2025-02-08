import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas as pd
import time
from collections import defaultdict
import mediapipe as mp
from recommender import SkinCareRecommender  # Importing recommendation system

# Face Detection using MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

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

    def detect_face(self, frame):
        """ Detects face using MediaPipe, draws bounding box, and returns face ROI and coordinates. """
        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x1, y1, x2, y2 = (
                        int(bboxC.xmin * w), int(bboxC.ymin * h),
                        int((bboxC.xmin + bboxC.width) * w), int((bboxC.ymin + bboxC.height) * h)
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    return frame[y1:y2, x1:x2], frame, (x1, y1, x2, y2)  # Return cropped face, frame, and coordinates
        return None, frame, (0, 0, 0, 0)

    def draw_predictions(self, frame, detections, face_coords):
        """ Draw bounding boxes and labels on the frame relative to face position. """
        face_x1, face_y1, _, _ = face_coords
        for result in detections:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 += face_x1
                y1 += face_y1
                x2 += face_x1
                y2 += face_y1
                
                confidence = box.conf.item()
                class_id = int(box.cls.item())
                class_name = result.names[class_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} ({confidence:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        return frame

    def process_frame(self, frame):
        """ Processes a frame: detects face, applies CLAHE, runs YOLO, and returns detections. """
        face_roi, processed_frame, face_coords = self.detect_face(frame)
        if face_roi is not None:
            enhanced_face = self.preprocessor.apply_clahe(face_roi)
            detections = self.model(enhanced_face)  # Run YOLO only on the face
            return detections, processed_frame, face_coords
        return None, frame, (0, 0, 0, 0)

# Load dataset
df = pd.read_csv(r'C:\Users\Samiksha Bhatia\Acne_gpu\myvenv\SkinCare_Recommendation_Final\skincare_products_1500_unique.csv')

# Load YOLO Model
model_path = r"C:\Users\Samiksha Bhatia\Acne_gpu\runs\detect\train15\weights\best.pt"
yolo_processor = YOLOProcessor(model_path, df)

# Streamlit UI
st.title("Real-Time Acne Detection")

if st.button("Start Video", key="start_video"):
    cap = cv2.VideoCapture(0)  # Open webcam
    video_placeholder = st.empty()
    detection_placeholder = st.empty()
    
    start_time = time.time()
    detection_log = defaultdict(list)  # Store detection timestamps

    while time.time() - start_time < 15:  # Run for 10 seconds
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture video.")
            break

        detections, processed_frame, face_coords = yolo_processor.process_frame(frame)

        if detections:
            for result in detections:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    class_name = result.names[class_id]
                    detection_log[class_name].append(time.time() - start_time)

            processed_frame = yolo_processor.draw_predictions(processed_frame, detections, face_coords)

        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

    cap.release()
    cv2.destroyAllWindows()
    video_placeholder.empty()

    # Filter detections that were present for at least 2 seconds
    final_detections = {key: len([t for t in times if times.count(t) >= 1]) for key, times in detection_log.items() if len(times) >= 1}

    # Remove skin concerns with zero count
    filtered_detections = {k: v for k, v in final_detections.items() if v > 5}
    # Calculate severity
    severity, score = yolo_processor.recommender.calculate_severity(filtered_detections)
    
    # Only pass detected concerns with count > 0
    skin_concern = list(filtered_detections.keys())

    processed_skin_concerns = [concern.strip() for entry in filtered_detections for concern in entry.split(",")]
    
    

    # Get recommendations only if there are valid skin concerns
    recommendations = yolo_processor.recommender.get_recommendations(skin_concern=processed_skin_concerns, severity=severity) 
    


    # Display results
    detection_placeholder.subheader("Final Detection Summary:")
    st.write(f"Skin Concerns: {processed_skin_concerns}")
   

    
    # st.subheader("SkinCare Recommendations:")
    # for rec in recommendations:
    #     st.write(f"• {rec}")

    st.subheader("Overall Severity Level:")
    st.write(severity if severity else "None")
    
    st.subheader("Recommended Products:")
    if not recommendations.empty:
        st.dataframe(recommendations)
    else:
        st.write("No recommendations available.")
    
    
