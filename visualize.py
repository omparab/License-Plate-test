"""
Creates an annotated video with bounding boxes and license plate overlays
Reads from results_interpolated.csv and creates out.mp4
"""

import ast
import cv2
import numpy as np
import pandas as pd


def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, 
                line_length_x=200, line_length_y=200):
    """
    Draw decorative corner borders around bounding boxes
    
    Args:
        img: Image to draw on
        top_left: (x1, y1) coordinates
        bottom_right: (x2, y2) coordinates
        color: Border color (B, G, R)
        thickness: Line thickness
        line_length_x: Horizontal line length
        line_length_y: Vertical line length
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-left corner
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)

    # Bottom-left corner
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)

    # Top-right corner
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)

    # Bottom-right corner
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def parse_bbox(bbox_str):
    """Parse bounding box string to coordinates"""
    return ast.literal_eval(
        bbox_str.replace('[ ', '[')
                .replace('   ', ' ')
                .replace('  ', ' ')
                .replace(' ', ',')
    )


def create_annotated_video(input_csv='results_interpolated.csv', 
                          video_path='sample.mp4', 
                          output_path='out.mp4',
                          show_registration_status=True):
    """
    Create annotated video with bounding boxes and license plates
    
    Args:
        input_csv: Path to interpolated CSV file
        video_path: Path to input video
        output_path: Path to output video
        show_registration_status: Show registration status in overlay
    """
    print(f"Loading results from {input_csv}...")
    results = pd.read_csv(input_csv)
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video properties: {width}x{height} @ {fps} fps")
    
    # Create video writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get best license plate crop for each car
    print("Extracting best license plate crops...")
    license_plate = {}
    
    for car_id in np.unique(results['car_id']):
        # Find frame with highest confidence for this car
        car_data = results[results['car_id'] == car_id]
        max_score_idx = car_data['license_plate_text_score'].astype(float).idxmax()
        best_row = car_data.loc[max_score_idx]
        
        license_plate_text = best_row['license_plate_text']
        if license_plate_text == '0':
            license_plate_text = 'UNKNOWN'
        
        license_plate[car_id] = {
            'license_crop': None,
            'license_plate_text': license_plate_text,
            'is_registered': best_row.get('is_registered', 'UNKNOWN'),
            'owner_name': best_row.get('owner_name', ''),
            'vehicle_status': best_row.get('vehicle_status', '')
        }
        
        # Extract license plate crop from best frame
        frame_number = int(best_row['frame_nmr'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            try:
                x1, y1, x2, y2 = parse_bbox(best_row['license_plate_bbox'])
                license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
                
                # Resize license plate crop for display
                if license_crop.size > 0:
                    license_crop = cv2.resize(
                        license_crop, 
                        (int((x2 - x1) * 400 / (y2 - y1)), 400)
                    )
                    license_plate[car_id]['license_crop'] = license_crop
            except Exception as e:
                print(f"Warning: Could not extract license crop for car {car_id}: {e}")
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Process video frames
    print("Creating annotated video...")
    frame_nmr = -1
    ret = True
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while ret:
        ret, frame = cap.read()
        frame_nmr += 1
        
        if ret:
            # Get detections for this frame
            df_ = results[results['frame_nmr'] == frame_nmr]
            
            for row_indx in range(len(df_)):
                row = df_.iloc[row_indx]
                car_id = row['car_id']
                
                # Determine color based on registration status
                if show_registration_status and row.get('is_registered') == 'YES':
                    car_color = (0, 255, 0)  # Green for registered
                    plate_color = (0, 255, 0)
                elif show_registration_status and row.get('is_registered') == 'NO':
                    car_color = (0, 0, 255)  # Red for unregistered
                    plate_color = (0, 0, 255)
                else:
                    car_color = (0, 255, 0)  # Default green
                    plate_color = (0, 0, 255)  # Default red for plate
                
                # Draw car bounding box with corners
                try:
                    car_x1, car_y1, car_x2, car_y2 = parse_bbox(row['car_bbox'])
                    draw_border(
                        frame, 
                        (int(car_x1), int(car_y1)), 
                        (int(car_x2), int(car_y2)), 
                        car_color, 
                        25,
                        line_length_x=200, 
                        line_length_y=200
                    )
                except Exception as e:
                    print(f"Warning: Could not draw car bbox at frame {frame_nmr}: {e}")
                    continue
                
                # Draw license plate rectangle
                try:
                    x1, y1, x2, y2 = parse_bbox(row['license_plate_bbox'])
                    cv2.rectangle(
                        frame, 
                        (int(x1), int(y1)), 
                        (int(x2), int(y2)), 
                        plate_color, 
                        12
                    )
                except Exception as e:
                    print(f"Warning: Could not draw plate bbox at frame {frame_nmr}: {e}")
                    continue
                
                # Overlay license plate crop and text
                if car_id in license_plate and license_plate[car_id]['license_crop'] is not None:
                    try:
                        license_crop = license_plate[car_id]['license_crop']
                        H, W, _ = license_crop.shape
                        
                        # Calculate position for overlay (above car)
                        y_offset = 100
                        crop_y1 = int(car_y1) - H - y_offset
                        crop_y2 = int(car_y1) - y_offset
                        crop_x1 = int((car_x2 + car_x1 - W) / 2)
                        crop_x2 = int((car_x2 + car_x1 + W) / 2)
                        
                        # Check if overlay fits in frame
                        if crop_y1 > 0 and crop_x1 > 0 and crop_x2 < width:
                            # Place license plate crop
                            frame[crop_y1:crop_y2, crop_x1:crop_x2, :] = license_crop
                            
                            # Create white background for text
                            text_bg_y1 = crop_y1 - 300
                            text_bg_y2 = crop_y1
                            if text_bg_y1 > 0:
                                frame[text_bg_y1:text_bg_y2, crop_x1:crop_x2, :] = (255, 255, 255)
                                
                                # Draw license plate text
                                plate_text = license_plate[car_id]['license_plate_text']
                                (text_width, text_height), _ = cv2.getTextSize(
                                    plate_text,
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    4.3,
                                    17
                                )
                                
                                text_x = int((car_x2 + car_x1 - text_width) / 2)
                                text_y = int(crop_y1 - 150 + (text_height / 2))
                                
                                cv2.putText(
                                    frame,
                                    plate_text,
                                    (text_x, text_y),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    4.3,
                                    (0, 0, 0),
                                    17
                                )
                                
                                # Add registration status
                                if show_registration_status:
                                    status = license_plate[car_id]['is_registered']
                                    status_text = f"✓ {status}" if status == 'YES' else f"✗ {status}"
                                    status_color = (0, 128, 0) if status == 'YES' else (0, 0, 128)
                                    
                                    cv2.putText(
                                        frame,
                                        status_text,
                                        (text_x, text_y + 100),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        2.0,
                                        status_color,
                                        8
                                    )
                    except Exception as e:
                        pass  # Skip overlay if it doesn't fit
            
            # Write frame to output video
            out.write(frame)
            
            # Progress indicator
            if frame_nmr % 30 == 0:
                progress = (frame_nmr / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_nmr}/{total_frames} frames)")
    
    # Cleanup
    out.release()
    cap.release()
    
    print(f"✅ Annotated video saved to {output_path}")


if __name__ == '__main__':
    create_annotated_video()
