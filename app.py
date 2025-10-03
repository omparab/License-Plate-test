import streamlit as st
import torch.serialization
from ultralytics.nn.tasks import DetectionModel
import cv2
import tempfile
import os
from pathlib import Path
import pandas as pd
import time
from ultralytics import YOLO
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv
import numpy as np
from database import LicensePlateDB, initialize_sample_data

# Allow YOLO model loading
torch.serialization.add_safe_globals([DetectionModel])

# Initialize database
@st.cache_resource
def get_database():
    return LicensePlateDB()

# Page config
st.set_page_config(
    page_title="License Plate Detection System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database
db = get_database()

# Title and description
st.title("🚗 License Plate Detection & Verification System")
st.markdown("Upload a video to detect vehicles and verify license plates against registered database.")

# Sidebar navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Go to", ["🎥 Video Analysis", "🗄️ Database Management", "📊 Reports & Alerts"])

if page == "🗄️ Database Management":
    st.header("🗄️ Registered Vehicles Database")
    
    # Tabs for different database operations
    tab1, tab2, tab3 = st.tabs(["View Vehicles", "Add Vehicle", "Bulk Import"])
    
    with tab1:
        st.subheader("All Registered Vehicles")
        vehicles = db.get_all_registered_vehicles()
        
        if vehicles:
            df_vehicles = pd.DataFrame(vehicles, columns=[
                'ID', 'License Plate', 'Owner Name', 'Vehicle Type', 
                'Status', 'Registered Date', 'Notes'
            ])
            st.dataframe(df_vehicles, width="stretch")
            
            # Delete vehicle option
            st.subheader("Remove Vehicle")
            plate_to_delete = st.selectbox(
                "Select license plate to remove",
                options=[v[1] for v in vehicles]
            )
            if st.button("🗑️ Delete Vehicle", type="secondary"):
                if db.delete_vehicle(plate_to_delete):
                    st.success(f"Vehicle {plate_to_delete} removed successfully!")
                    st.rerun()
                else:
                    st.error("Failed to remove vehicle")
        else:
            st.info("No vehicles registered yet. Add some vehicles to get started!")
            if st.button("➕ Initialize Sample Data"):
                initialize_sample_data()
                st.success("Added 5 sample vehicles!")
                st.rerun()
    
    with tab2:
        st.subheader("Add New Vehicle")
        with st.form("add_vehicle_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_plate = st.text_input("License Plate *", placeholder="ABC1234").upper()
                owner_name = st.text_input("Owner Name", placeholder="John Doe")
                vehicle_type = st.selectbox("Vehicle Type", 
                    ["Car", "SUV", "Van", "Truck", "Motorcycle", "Other"])
            
            with col2:
                status = st.selectbox("Status", 
                    ["active", "suspended", "stolen", "expired"])
                notes = st.text_area("Notes", placeholder="Additional information...")
            
            submitted = st.form_submit_button("➕ Add Vehicle", type="primary")
            
            if submitted:
                if new_plate:
                    success, message = db.add_registered_vehicle(
                        new_plate, owner_name, vehicle_type, status, notes
                    )
                    if success:
                        st.success(message)
                        st.balloons()
                    else:
                        st.error(message)
                else:
                    st.error("License plate is required!")
    
    with tab3:
        st.subheader("Bulk Import Vehicles")
        st.markdown("Upload a CSV file with columns: `license_plate, owner_name, vehicle_type, status, notes`")
        
        uploaded_csv = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df, width="stretch")
            
            if st.button("📥 Import All Vehicles"):
                vehicles_data = []
                for _, row in df.iterrows():
                    vehicles_data.append((
                        str(row.get('license_plate', '')).upper(),
                        row.get('owner_name', None),
                        row.get('vehicle_type', None),
                        row.get('status', 'active'),
                        row.get('notes', None)
                    ))
                
                added, skipped = db.add_multiple_vehicles(vehicles_data)
                st.success(f"✅ Added {added} vehicles, skipped {skipped} duplicates")

elif page == "📊 Reports & Alerts":
    st.header("📊 Detection Reports & Alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Recent Detections")
        detections = db.get_detection_history(limit=50)
        
        if detections:
            df_detections = pd.DataFrame(detections, columns=[
                'ID', 'License Plate', 'Detection Time', 'Video Name',
                'Frame Number', 'Confidence', 'Registered', 'Match Type'
            ])
            st.dataframe(df_detections, width="stretch")
        else:
            st.info("No detections yet. Process a video to see results here.")
    
    with col2:
        st.subheader("⚠️ Active Alerts")
        alerts = db.get_active_alerts()
        
        if alerts:
            df_alerts = pd.DataFrame(alerts, columns=[
                'ID', 'License Plate', 'Alert Type', 'Alert Time', 'Description'
            ])
            st.dataframe(df_alerts, width="stretch")
        else:
            st.success("No active alerts")

else:  # Video Analysis page
    st.header("🎥 Video Analysis")

# Sidebar for settings
st.sidebar.header("⚙️ Detection Settings")
frame_skip = st.sidebar.slider("Process every Nth frame", 1, 20, 10, 
                               help="Higher values = faster processing but may miss some detections")
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.5, 
                                        help="Minimum confidence for detections")
enable_db_check = st.sidebar.checkbox("✅ Enable Database Verification", value=True,
                                     help="Check detected plates against database")
fuzzy_match = st.sidebar.checkbox("🔍 Enable Fuzzy Matching", value=True,
                                  help="Find similar plates even with OCR errors")
fuzzy_threshold = st.sidebar.slider("Fuzzy Match Threshold", 0.5, 1.0, 0.8,
                                   help="Similarity threshold for fuzzy matching")

# Cache models to avoid reloading
@st.cache_resource
def load_models():
    with st.spinner("Loading AI models... This may take a minute."):
        coco_model = YOLO('yolov8n.pt')
        license_plate_detector = YOLO('license_plate_detector.pt')
    return coco_model, license_plate_detector

# Main processing function
def process_video(video_path, coco_model, license_plate_detector, frame_skip, enable_db_check=True, fuzzy_match=False, fuzzy_threshold=0.75):
    results = {}
    mot_tracker = Sort()
    license_plate_cache = {}
    vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    
    # Statistics
    registered_count = 0
    unregistered_count = 0
    fuzzy_matches = []
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    frame_nmr = -1
    ret = True
    start_time = time.time()
    processed_frames = 0
    ocr_calls = 0
    
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        
        if ret and frame_nmr % frame_skip == 0:
            processed_frames += 1
            
            # Update progress
            progress = frame_nmr / total_frames
            progress_bar.progress(progress)
            
            elapsed = time.time() - start_time
            fps_processing = processed_frames / elapsed if elapsed > 0 else 0
            
            status_text.text(f"Processing frame {frame_nmr}/{total_frames} | "
                           f"Speed: {fps_processing:.2f} fps | "
                           f"Time: {elapsed:.1f}s")
            
            stats_col1.metric("Frames", f"{processed_frames}/{total_frames//frame_skip}")
            stats_col2.metric("OCR Calls", ocr_calls)
            stats_col3.metric("✅ Registered", registered_count)
            stats_col4.metric("❌ Unregistered", unregistered_count)
            
            results[frame_nmr] = {}
            
            # Detect vehicles
            detections = coco_model(frame)[0]
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                if int(class_id) in vehicles and score >= confidence_threshold:
                    detections_.append([x1, y1, x2, y2, score])
            
            # Track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))
            
            # Detect license plates
            license_plates = license_plate_detector(frame)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate
                
                if score < confidence_threshold:
                    continue
                
                # Assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                
                if car_id != -1:
                    # Check cache first
                    if car_id in license_plate_cache:
                        license_plate_text, license_plate_text_score, db_result = license_plate_cache[car_id]
                    else:
                        # Crop and process license plate
                        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(
                            license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                        )
                        
                        # Read license plate
                        license_plate_text, license_plate_text_score = read_license_plate(
                            license_plate_crop_thresh
                        )
                        ocr_calls += 1
                        
                        # Check against database
                        db_result = None
                        if enable_db_check and license_plate_text:
                            # First try exact match
                            db_result = db.check_license_plate(license_plate_text)
                            
                            # If not found and fuzzy matching enabled
                            if not db_result['found'] and fuzzy_match:
                                matches = db.fuzzy_match(license_plate_text, threshold=fuzzy_threshold)
                                if matches:
                                    db_result = {
                                        'found': True,
                                        'fuzzy': True,
                                        'matches': matches,
                                        'best_match': matches[0]
                                    }
                                    fuzzy_matches.append({
                                        'detected': license_plate_text,
                                        'matched': matches[0]['plate'],
                                        'similarity': matches[0]['similarity']
                                    })
                            
                            # Update statistics
                            if db_result and db_result['found']:
                                registered_count += 1
                                # Log detection
                                db.log_detection(
                                    license_plate_text,
                                    video_name="uploaded_video",
                                    frame_number=frame_nmr,
                                    confidence_score=license_plate_text_score,
                                    is_registered=True,
                                    match_type='fuzzy' if db_result.get('fuzzy') else 'exact'
                                )
                            else:
                                unregistered_count += 1
                                # Create alert for unregistered vehicle
                                db.create_alert(
                                    license_plate_text,
                                    alert_type='unregistered',
                                    description=f'Detected at frame {frame_nmr}'
                                )
                                # Log detection
                                db.log_detection(
                                    license_plate_text,
                                    video_name="uploaded_video",
                                    frame_number=frame_nmr,
                                    confidence_score=license_plate_text_score,
                                    is_registered=False,
                                    match_type='none'
                                )
                        
                        # Cache result
                        if license_plate_text is not None:
                            license_plate_cache[car_id] = (license_plate_text, license_plate_text_score, db_result)
                    
                    if license_plate_text is not None:
                        result_entry = {
                            'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                            'license_plate': {
                                'bbox': [x1, y1, x2, y2],
                                'text': license_plate_text,
                                'bbox_score': score,
                                'text_score': license_plate_text_score
                            }
                        }
                        
                        # Add database check result
                        if db_result:
                            result_entry['database'] = db_result
                        
                        results[frame_nmr][car_id] = result_entry
    
    cap.release()
    progress_bar.progress(1.0)
    
    total_time = time.time() - start_time
    status_text.text(f"✅ Processing complete! Total time: {total_time:.2f}s")
    
    return results, total_time, ocr_calls, len(license_plate_cache), registered_count, unregistered_count, fuzzy_matches
                


# File uploader
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Display video
    st.video(uploaded_file)
    
    # Process button
    if st.button("🚀 Start Processing", type="primary"):
        try:
            # Load models
            coco_model, license_plate_detector = load_models()
            
            # Process video
            st.header("📊 Processing Progress")
            results, total_time, ocr_calls, unique_cars = process_video(
                video_path, coco_model, license_plate_detector, frame_skip
            )
            
            # Display results
            st.success(f"Processing completed in {total_time:.2f} seconds!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total OCR Calls", ocr_calls)
            col2.metric("Unique Vehicles", unique_cars)
            col3.metric("Processing Speed", f"{total_time:.1f}s")
            
            # Save results to CSV
            csv_path = 'results.csv'
            write_csv(results, csv_path)
            
            # Read and display CSV
            if os.path.exists(csv_path):
                st.header("📋 Detection Results")
                df = pd.read_csv(csv_path)
                st.dataframe(df, width="stretch")
                
                # Download button
                with open(csv_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=f,
                        file_name="license_plate_results.csv",
                        mime="text/csv"
                    )
                
                # Statistics
                st.header("📈 Statistics")
                stat_col1, stat_col2 = st.columns(2)
                
                with stat_col1:
                    st.subheader("License Plates Detected")
                    if 'license_plate_text' in df.columns:
                        plate_counts = df['license_plate_text'].value_counts()
                        st.bar_chart(plate_counts)
                
                with stat_col2:
                    st.subheader("Detection Confidence")
                    if 'license_plate_text_score' in df.columns:
                        st.line_chart(df['license_plate_text_score'])
        
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
            st.exception(e)
        
        finally:
            # Cleanup - close file handles first
            try:
                if os.path.exists(video_path):
                    # Give it a moment for cv2 to release the file
                    time.sleep(0.5)
                    os.unlink(video_path)
            except PermissionError:
                # File still in use, skip cleanup
                pass

else:
    st.info("👆 Please upload a video file to get started")
    
    # Instructions
    st.header("📖 How to Use")
    st.markdown("""
    1. **Upload a video** using the file uploader above
    2. **Adjust settings** in the sidebar (optional)
        - Frame skip: Higher = faster processing
        - Confidence threshold: Higher = fewer false positives
    3. **Click "Start Processing"** to analyze the video
    4. **Download results** as CSV when complete
    
    **Note:** First run will download AI models (~50MB), which may take a few minutes.
    """)
    
    st.header("⚡ Performance Tips")
    st.markdown("""
    - **CPU processing is slow**: Processing 1 minute of video may take 5-10 minutes
    - **Increase frame skip**: Set to 15-20 for faster processing
    - **GPU recommended**: Much faster with CUDA-enabled GPU
    - **Smaller videos**: Test with shorter clips first
    """)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit 🎈 | Powered by YOLOv8 & EasyOCR")