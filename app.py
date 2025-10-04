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
from add_missing_data import interpolate_bounding_boxes
from visualize import create_annotated_video
import csv

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

st.sidebar.header("🎬 Video Output Settings")
create_video = st.sidebar.checkbox("📹 Create Annotated Video", value=True,
                                   help="Generate output video with bounding boxes")
show_registration_in_video = st.sidebar.checkbox("🏷️ Show Registration Status", value=True,
                                                  help="Display registration status in video")

# Cache models to avoid reloading
@st.cache_resource
def load_models():
    with st.spinner("Loading AI models... This may take a minute."):
        try:
            # YOLOv8n will auto-download if not present
            coco_model = YOLO('yolov8n.pt')
            
            # Check if custom license plate detector exists
            if os.path.exists('license_plate_detector.pt'):
                license_plate_detector = YOLO('license_plate_detector.pt')
            else:
                st.warning("⚠️ Custom license plate detector not found. Using YOLOv8n for both detection tasks.")
                st.info("To use a custom model, upload 'license_plate_detector.pt' to your repository.")
                license_plate_detector = YOLO('yolov8n.pt')
            
            return coco_model, license_plate_detector
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            st.stop()

# Main processing function
def process_video(video_path, coco_model, license_plate_detector, frame_skip, enable_db_check, fuzzy_match, fuzzy_threshold):
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
            results, total_time, ocr_calls, unique_cars, registered_count, unregistered_count, fuzzy_matches = process_video(
                video_path, coco_model, license_plate_detector, frame_skip, 
                enable_db_check, fuzzy_match, fuzzy_threshold
            )
            
            # Display results
            st.success(f"Processing completed in {total_time:.2f} seconds!")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total OCR Calls", ocr_calls)
            col2.metric("Unique Vehicles", unique_cars)
            col3.metric("✅ Registered", registered_count, delta="Authorized")
            col4.metric("❌ Unregistered", unregistered_count, delta="Alert", delta_color="inverse")
            
            # Show verification results
            if enable_db_check:
                st.header("🔍 Verification Results")
                
                # Registered vehicles found
                if registered_count > 0:
                    st.success(f"✅ Found {registered_count} registered vehicle(s)")
                    
                    registered_plates = []
                    for frame_results in results.values():
                        for car_result in frame_results.values():
                            if 'database' in car_result and car_result['database']['found']:
                                db_info = car_result['database']
                                if not db_info.get('fuzzy'):
                                    registered_plates.append({
                                        'License Plate': car_result['license_plate']['text'],
                                        'Owner': db_info.get('owner_name', 'N/A'),
                                        'Vehicle Type': db_info.get('vehicle_type', 'N/A'),
                                        'Status': db_info.get('status', 'N/A'),
                                        'Confidence': f"{car_result['license_plate']['text_score']:.2%}"
                                    })
                    
                    if registered_plates:
                        df_registered = pd.DataFrame(registered_plates).drop_duplicates()
                        st.dataframe(df_registered, width="stretch")
                
                # Fuzzy matches
                if fuzzy_matches:
                    st.warning(f"🔍 Found {len(fuzzy_matches)} fuzzy match(es) (OCR corrections)")
                    df_fuzzy = pd.DataFrame(fuzzy_matches)
                    df_fuzzy['similarity'] = df_fuzzy['similarity'].apply(lambda x: f"{x:.2%}")
                    df_fuzzy.columns = ['Detected (OCR)', 'Matched (Database)', 'Similarity']
                    st.dataframe(df_fuzzy, width="stretch")
                
                # Unregistered vehicles
                if unregistered_count > 0:
                    st.error(f"⚠️ Detected {unregistered_count} unregistered vehicle(s)")
                    
                    unregistered_plates = []
                    for frame_results in results.values():
                        for car_result in frame_results.values():
                            if 'database' not in car_result or not car_result['database']['found']:
                                unregistered_plates.append({
                                    'License Plate': car_result['license_plate']['text'],
                                    'Confidence': f"{car_result['license_plate']['text_score']:.2%}",
                                    'Alert': '⚠️ Unauthorized'
                                })
                    
                    if unregistered_plates:
                        df_unregistered = pd.DataFrame(unregistered_plates).drop_duplicates()
                        st.dataframe(df_unregistered, width="stretch")
            
            # Save results to CSV with registration status
            csv_path = 'results.csv'
            
            # Enhanced write_csv with registration info
            csv_data = []
            for frame_nmr, frame_data in results.items():
                for car_id, data in frame_data.items():
                    # Check registration status
                    is_registered = False
                    match_type = 'none'
                    owner_name = ''
                    vehicle_status = ''
                    
                    if 'database' in data and data['database']['found']:
                        is_registered = True
                        if data['database'].get('fuzzy'):
                            match_type = 'fuzzy'
                            best_match = data['database']['best_match']
                            # Get details from database
                            db_details = db.check_license_plate(best_match['plate'])
                            owner_name = db_details.get('owner_name', '')
                            vehicle_status = db_details.get('status', '')
                        else:
                            match_type = 'exact'
                            owner_name = data['database'].get('owner_name', '')
                            vehicle_status = data['database'].get('status', '')
                    
                    # Convert bounding boxes to string format for interpolation
                    car_bbox_list = data['car']['bbox']
                    license_bbox_list = data['license_plate']['bbox']
                    
                    csv_data.append({
                        'frame_nmr': frame_nmr,
                        'car_id': car_id,
                        'car_bbox': f"[{' '.join(map(str, car_bbox_list))}]",  # Convert to string format
                        'license_plate_bbox': f"[{' '.join(map(str, license_bbox_list))}]",  # Convert to string format
                        'license_plate_text': data['license_plate']['text'],
                        'license_plate_bbox_score': data['license_plate']['bbox_score'],
                        'license_plate_text_score': data['license_plate']['text_score'],
                        'is_registered': 'YES' if is_registered else 'NO',
                        'match_type': match_type,
                        'owner_name': owner_name,
                        'vehicle_status': vehicle_status
                    })
            
            # Save to CSV
            df_output = pd.DataFrame(csv_data)
            df_output.to_csv(csv_path, index=False)
            
            # Read and display CSV
            if os.path.exists(csv_path):
                st.header("📋 Complete Detection Results")
                df = pd.read_csv(csv_path)
                
                # Reorder columns for better display
                column_order = [
                    'frame_nmr', 'car_id', 'license_plate_text', 'is_registered',
                    'match_type', 'owner_name', 'vehicle_status',
                    'license_plate_text_score', 'license_plate_bbox_score'
                ]
                
                # Only include columns that exist
                display_columns = [col for col in column_order if col in df.columns]
                df_display = df[display_columns]
                
                # Style the dataframe
                def highlight_registration(row):
                    if row['is_registered'] == 'YES':
                        return ['background-color: #d4edda'] * len(row)
                    else:
                        return ['background-color: #f8d7da'] * len(row)
                
                st.dataframe(
                    df_display.style.apply(highlight_registration, axis=1),
                    width="stretch"
                )
                
                # Summary statistics
                st.subheader("📊 Summary")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                registered_in_csv = (df['is_registered'] == 'YES').sum()
                unregistered_in_csv = (df['is_registered'] == 'NO').sum()
                total_detections = len(df)
                
                summary_col1.metric("Total Detections", total_detections)
                summary_col2.metric("✅ Registered", registered_in_csv)
                summary_col3.metric("❌ Unregistered", unregistered_in_csv)
                
                # Download button
                with open(csv_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download Results (CSV)",
                        data=f,
                        file_name="license_plate_results.csv",
                        mime="text/csv"
                    )
                
                # Statistics
                st.header("📈 Detection Statistics")
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
            
            # Generate annotated video if requested
            if create_video:
                st.header("🎬 Generating Annotated Video")
                
                with st.spinner("Creating annotated video... This may take a few minutes."):
                    try:
                        # Step 1: Interpolate missing frames
                        st.info("Step 1/2: Interpolating frames for smooth tracking...")
                        
                        # Convert DataFrame to list of dicts for interpolation
                        csv_data_for_interp = df_output.to_dict('records')
                        interpolated_data = interpolate_bounding_boxes(csv_data_for_interp)
                        
                        # Save interpolated data
                        interp_csv_path = 'results_interpolated.csv'
                        header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 
                                  'license_plate_bbox_score', 'license_plate_text', 'license_plate_text_score',
                                  'is_registered', 'match_type', 'owner_name', 'vehicle_status']
                        
                        with open(interp_csv_path, 'w', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=header)
                            writer.writeheader()
                            writer.writerows(interpolated_data)
                        
                        st.success(f"✅ Interpolated {len(interpolated_data)} frames")
                        
                        # Step 2: Create annotated video
                        st.info("Step 2/2: Creating annotated video with bounding boxes...")
                        
                        output_video_path = 'output_annotated.mp4'
                        create_annotated_video(
                            input_csv=interp_csv_path,
                            video_path=video_path,
                            output_path=output_video_path,
                            show_registration_status=show_registration_in_video
                        )
                        
                        st.success("✅ Annotated video created successfully!")
                        
                        # Display video
                        if os.path.exists(output_video_path):
                            st.subheader("📹 Annotated Video Output")
                            st.video(output_video_path)
                            
                            # Download button for video
                            with open(output_video_path, 'rb') as video_file:
                                st.download_button(
                                    label="📥 Download Annotated Video",
                                    data=video_file,
                                    file_name="annotated_video.mp4",
                                    mime="video/mp4"
                                )
                        
                    except Exception as e:
                        st.error(f"Error creating video: {str(e)}")
                        st.exception(e)
        
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
