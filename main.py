import torch.serialization
from ultralytics.nn.tasks import DetectionModel
import time
torch.serialization.add_safe_globals([DetectionModel])
from ultralytics import YOLO
import cv2
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}
mot_tracker = Sort()
license_plate_cache = {}  # Cache plates per car

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
cap = cv2.VideoCapture('./sample.mp4')
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f"Total frames: {total_frames}, FPS: {fps}")

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
start_time = time.time()
processed_frames = 0
ocr_calls = 0

while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    
    # Process every 10th frame for much faster processing
    if ret and frame_nmr % 10 == 0:
        processed_frames += 1
        if processed_frames % 5 == 0:
            elapsed = time.time() - start_time
            fps_processing = processed_frames / elapsed
            print(f"Frame {frame_nmr}/{total_frames} | Speed: {fps_processing:.2f} fps | Time: {elapsed:.1f}s | OCR calls: {ocr_calls}")
        
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # Check cache first
                if car_id in license_plate_cache:
                    license_plate_text, license_plate_text_score = license_plate_cache[car_id]
                else:
                    # crop license plate
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                    # process license plate
                    license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number (this is the slow part)
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    ocr_calls += 1
                    
                    # Cache the result
                    if license_plate_text is not None:
                        license_plate_cache[car_id] = (license_plate_text, license_plate_text_score)

                if license_plate_text is not None:
                    results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                  'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}

cap.release()
print(f"\nProcessing complete! Total time: {time.time() - start_time:.2f}s")
print(f"Total OCR calls: {ocr_calls}")
print(f"Unique cars detected: {len(license_plate_cache)}")

# write results
write_csv(results, './test.csv')