#!/usr/bin/env python3
import os
import sys
import time
import cv2
import numpy as np
import torch
import json
from pathlib import Path
import math

# Set MPS fallback for operations not supported on Apple Silicon
if hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Import our modules
from detection_model import ObjectDetector
from depth_model import DepthEstimator
from bbox3d_utils import BBox3DEstimator, BirdEyeView
from load_camera_params import load_camera_params, apply_camera_params_to_estimator

def calculate_velocity(prev_position, current_position, time_delta, prev_velocity=None):
    """
    Calculate velocity components and acceleration based on position change.
    
    Args:
        prev_position: Dictionary with previous x, y, z coordinates
        current_position: Dictionary with current x, y, z coordinates
        time_delta: Time difference in seconds
        prev_velocity: Optional dictionary with previous velocity data for acceleration calculation
        
    Returns:
        Dictionary with speed, velocity components, and acceleration components
    """
    if time_delta <= 0:
        return {
            "speed": 0.0,
            "lateral_velocity": 0.0,
            "longitudinal_velocity": 0.0,
            "lateral_acceleration": 0.0,
            "longitudinal_acceleration": 0.0
        }
    
    # Calculate displacement in each direction
    dx = current_position["x"] - prev_position["x"]  # lateral displacement (left/right)
    dy = current_position["y"] - prev_position["y"]  # vertical displacement (up/down)
    dz = current_position["z"] - prev_position["z"]  # longitudinal displacement (forward/backward)
    
    # Calculate velocities (distance / time)
    lateral_velocity = dx / time_delta  # meters per second
    vertical_velocity = dy / time_delta  # meters per second
    longitudinal_velocity = dz / time_delta  # meters per second
    
    # Calculate total speed (magnitude of the velocity vector)
    speed = math.sqrt(lateral_velocity**2 + vertical_velocity**2 + longitudinal_velocity**2)
    
    # Initialize acceleration values
    lateral_acceleration = 0.0
    longitudinal_acceleration = 0.0
    
    # Calculate accelerations if previous velocity is available
    if prev_velocity is not None:
        # Change in velocity
        delta_v_lateral = lateral_velocity - prev_velocity["lateral_velocity"]
        delta_v_longitudinal = longitudinal_velocity - prev_velocity["longitudinal_velocity"]
        
        # Acceleration (a = Δv/Δt)
        lateral_acceleration = delta_v_lateral / time_delta  # m/s²
        longitudinal_acceleration = delta_v_longitudinal / time_delta  # m/s²
    
    return {
        "speed": round(speed, 2),                       # total speed in m/s
        "speed_kmh": round(speed * 3.6, 2),             # speed in km/h
        "lateral_velocity": round(lateral_velocity, 2),  # left/right velocity in m/s
        "longitudinal_velocity": round(longitudinal_velocity, 2),  # forward/backward velocity in m/s
        "lateral_acceleration": round(lateral_acceleration, 2),  # left/right acceleration in m/s²
        "longitudinal_acceleration": round(longitudinal_acceleration, 2)  # forward/backward acceleration in m/s²
    }

def calculate_occlusion(bbox1, bbox2, depth1, depth2):
    """
    Calculate the occlusion level between two bounding boxes.
    
    Args:
        bbox1: 2D bounding box of first object [x1, y1, x2, y2]
        bbox2: 2D bounding box of second object [x1, y1, x2, y2]
        depth1: Depth value of first object
        depth2: Depth value of second object
        
    Returns:
        Float: occlusion level (0.0-1.0) where 0 means no occlusion
    """
    # Extract coordinates
    x1_a, y1_a, x2_a, y2_a = bbox1
    x1_b, y1_b, x2_b, y2_b = bbox2
    
    # Calculate areas of each bounding box
    area_a = (x2_a - x1_a) * (y2_a - y1_a)
    area_b = (x2_b - x1_b) * (y2_b - y1_b)
    
    if area_a <= 0 or area_b <= 0:
        return 0.0
    
    # Calculate intersection area
    x_overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
    y_overlap = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
    intersection_area = x_overlap * y_overlap
    
    # If there's no overlap, there's no occlusion
    if intersection_area <= 0:
        return 0.0
    
    # Determine which object is in front based on depth
    if depth1 > depth2:  # Object 2 is closer to the camera
        # Object 1 is occluded by Object 2
        occlusion_level = intersection_area / area_a
    else:  # Object 1 is closer to the camera
        # Object 2 is occluded by Object 1
        occlusion_level = intersection_area / area_b
    
    # Round to 2 decimal places
    return round(float(occlusion_level), 2)

def calculate_distance_3d(position1, position2):
    """
    Calculate the 3D Euclidean distance between two positions.
    
    Args:
        position1: Dictionary or list containing x, y, z coordinates
        position2: Dictionary or list containing x, y, z coordinates
        
    Returns:
        Float: distance in meters
    """
    # Extract coordinates based on input type
    if isinstance(position1, dict):
        x1, y1, z1 = position1["x"], position1["y"], position1["z"]
    else:  # assuming list or array
        x1, y1, z1 = position1[0], position1[1], position1[2]
        
    if isinstance(position2, dict):
        x2, y2, z2 = position2["x"], position2["y"], position2["z"]
    else:  # assuming list or array
        x2, y2, z2 = position2[0], position2[1], position2[2]
    
    # Calculate Euclidean distance
    distance = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
    
    # Print debug info
    print(f"Distance calculation: ({x1:.2f}, {y1:.2f}, {z1:.2f}) to ({x2:.2f}, {y2:.2f}, {z2:.2f}) = {distance:.2f}m")
    
    return distance

def find_closest_vehicle_pair(vehicles):
    """
    Find the closest pair of vehicles from a list of vehicles.
    
    Args:
        vehicles: List of dictionaries containing vehicle data with 3D positions
        
    Returns:
        Tuple: (index1, index2, distance) for the closest pair,
               or None if there are fewer than 2 vehicles
    """
    if len(vehicles) < 2:
        return None
    
    min_distance = float('inf')
    closest_pair = None
    
    # Compare each pair of vehicles
    for i in range(len(vehicles)):
        for j in range(i+1, len(vehicles)):
            vehicle1 = vehicles[i]
            vehicle2 = vehicles[j]
            
            # Calculate distance between them
            distance = calculate_distance_3d(
                vehicle1["vehicle_position"], 
                vehicle2["vehicle_position"]
            )
            
            # Update closest pair if this distance is smaller
            if distance < min_distance:
                min_distance = distance
                closest_pair = (i, j, distance)
    
    return closest_pair

def draw_star(img, x, y, size=10, color=(0, 255, 255), thickness=1):
    """Draw a star marker at the specified position."""
    # Draw a star (cross + x)
    half_size = size // 2
    
    # Draw cross
    cv2.line(img, (x - half_size, y), (x + half_size, y), color, thickness)
    cv2.line(img, (x, y - half_size), (x, y + half_size), color, thickness)
    
    # Draw X
    cv2.line(img, (x - half_size, y - half_size), (x + half_size, y + half_size), color, thickness)
    cv2.line(img, (x - half_size, y + half_size), (x + half_size, y - half_size), color, thickness)
    
    # Add circle in the center
    cv2.circle(img, (x, y), 2, color, thickness)

def main():
    """Main function."""
    # Configuration variables (modify these as needed)
    # ===============================================
    
    # Input/Output
    video_path = "video3.mp4"  # Path to input video file
    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found")
        return
        
    source = video_path  # Use the video file as source
    output_path = "output.mp4"  # Path to output video file
    
    # Model settings
    yolo_model_size = "nano"  # YOLOv11 model size: "nano", "small", "medium", "large", "extra"
    depth_model_size = "small"  # Depth Anything v2 model size: "small", "base", "large"
    
    # Device settings
    device = 'cpu'  # Force CPU for stability
    
    # Detection settings
    conf_threshold = 0.25  # Confidence threshold for object detection
    iou_threshold = 0.45  # IoU threshold for NMS
    classes = None  # Filter by class, e.g., [0, 1, 2] for specific classes, None for all classes
    
    # Feature toggles
    enable_tracking = True  # Enable object tracking
    enable_bev = True  # Enable Bird's Eye View visualization
    enable_pseudo_3d = True  # Enable pseudo-3D visualization
    
    # Camera parameters - simplified approach
    camera_params_file = None  # Path to camera parameters file (None to use default parameters)
    # ===============================================
    
    print(f"Using device: {device}")
    
    # Initialize models
    print("Initializing models...")
    try:
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device=device
        )
    except Exception as e:
        print(f"Error initializing object detector: {e}")
        print("Falling back to CPU for object detection")
        detector = ObjectDetector(
            model_size=yolo_model_size,
            conf_thres=conf_threshold,
            iou_thres=iou_threshold,
            classes=classes,
            device='cpu'
        )
    
    try:
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device=device
        )
    except Exception as e:
        print(f"Error initializing depth estimator: {e}")
        print("Falling back to CPU for depth estimation")
        depth_estimator = DepthEstimator(
            model_size=depth_model_size,
            device='cpu'
        )
    
    # Initialize 3D bounding box estimator with default parameters
    # Simplified approach - focus on 2D detection with depth information
    bbox3d_estimator = BBox3DEstimator()
    
    # Open video source
    try:
        if isinstance(source, str) and source.isdigit():
            source = int(source)  # Convert string number to integer for webcam
    except ValueError:
        pass  # Keep as string (for video file)
    
    print(f"Opening video source: {source}")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:  # Sometimes happens with webcams
        fps = 30
    
    # Initialize Bird's Eye View if enabled
    if enable_bev:
        # Use a scale that works well for the 1-5 meter range
        bev = BirdEyeView(size=(300, 300), scale=60, camera_height=1.2)  # Using correct parameters
    
    # Initialize video writer with a more reliable codec
    # Calculate the total width for side-by-side view (original frame + BEV)
    total_width = width * 2  # Double width for side-by-side view
    
    # Generate a unique output filename with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"output_{timestamp}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create video writer with MJPG codec")
        # Try another widely supported codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, height))
        
        if not out.isOpened():
            print("Error: Could not create video writer with any codec")
            return
        else:
            print(f"Successfully created video writer with XVID codec: {output_path}")
    else:
        print(f"Successfully created video writer with MJPG codec: {output_path}")
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = "FPS: --"
    
    # Initialize an empty list to store the centroid data
    centroids_data = []
    
    # Dictionary to store the previous position of each vehicle for velocity calculation
    previous_positions = {}
    
    # Dictionary to store the previous velocity of each vehicle for acceleration calculation
    previous_velocities = {}
    
    print("Starting processing...")
    
    # Main loop
    while True:
        # Check for key press at the beginning of each loop
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
            print("Exiting program...")
            break
            
        try:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Make copies for different visualizations
            original_frame = frame.copy()
            detection_frame = frame.copy()
            depth_frame = frame.copy()
            result_frame = frame.copy()
            
            # Step 1: Object Detection
            try:
                detection_frame, detections = detector.detect(detection_frame, track=enable_tracking)
            except Exception as e:
                print(f"Error during object detection: {e}")
                detections = []
                cv2.putText(detection_frame, "Detection Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 2: Depth Estimation
            try:
                depth_map = depth_estimator.estimate_depth(original_frame)
                depth_colored = depth_estimator.colorize_depth(depth_map)
            except Exception as e:
                print(f"Error during depth estimation: {e}")
                # Create a dummy depth map
                depth_map = np.zeros((height, width), dtype=np.float32)
                depth_colored = np.zeros((height, width, 3), dtype=np.uint8)
                cv2.putText(depth_colored, "Depth Error", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Step 3: 3D Bounding Box Estimation
            boxes_3d = []
            active_ids = []
            
            # Frame data dictionary to store information for this frame
            frame_data = {
                "frame_number": frame_count,
                "timestamp": time.time(),
                "objects": []
            }
            
            # Keep track of vehicle objects for inter-vehicle distance calculation
            vehicle_objects = []
            
            for detection in detections:
                try:
                    bbox, score, class_id, obj_id = detection
                    
                    # Get class name
                    class_name = detector.get_class_names()[class_id]
                    
                    # Get depth in the region of the bounding box
                    # Try different methods for depth estimation
                    if class_name.lower() in ['person', 'cat', 'dog']:
                        # For people and animals, use the center point depth
                        center_x = int((bbox[0] + bbox[2]) / 2)
                        center_y = int((bbox[1] + bbox[3]) / 2)
                        depth_value = depth_estimator.get_depth_at_point(depth_map, center_x, center_y)
                        depth_method = 'center'
                    else:
                        # For other objects, use the median depth in the region
                        depth_value = depth_estimator.get_depth_in_region(depth_map, bbox, method='median')
                        depth_method = 'median'
                    
                    # Use the bbox3d_estimator to compute 3D box information
                    full_box_3d = bbox3d_estimator.estimate_3d_box(
                        bbox_2d=bbox,
                        depth_value=depth_value,
                        class_name=class_name,
                        object_id=obj_id
                    )
                    
                    # Add additional information to the box
                    full_box_3d['depth_value'] = depth_value
                    full_box_3d['depth_method'] = depth_method
                    full_box_3d['score'] = score
                    
                    # Extract needed information
                    dimensions_3d = full_box_3d['dimensions'].tolist()
                    location_3d = full_box_3d['location'].tolist()
                    orientation = full_box_3d['orientation']
                    
                    # Get 2D centroid
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    centroid_2d = (center_x, center_y)
                    
                    # Get 2D dimensions
                    width_2d = bbox[2] - bbox[0]
                    height_2d = bbox[3] - bbox[1]
                    dimensions_2d = (width_2d, height_2d)
                    
                    # Round values for JSON serialization
                    centroid_3d_rounded = [round(float(val), 2) for val in location_3d]
                    dimensions_3d_rounded = [round(float(val), 2) for val in dimensions_3d]
                    
                    # Add centroid_2d to the full_box_3d for visualization
                    full_box_3d['centroid_2d'] = centroid_2d
                    
                    # Add centroid_3d to the full_box_3d for visualization
                    full_box_3d['centroid_3d'] = centroid_3d_rounded
                    
                    # Calculate vehicle position if it's a vehicle
                    vehicle_position = None
                    velocity_data = None
                    is_vehicle = any(vehicle_type in class_name.lower() for vehicle_type in 
                                    ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'vehicle'])
                    
                    # Print detected vehicles for debugging
                    if is_vehicle:
                        print(f"Detected vehicle: {class_name}, ID: {obj_id}, Score: {score:.2f}")
                    
                    if is_vehicle:
                        # Current vehicle position
                        vehicle_position = {
                            "x": centroid_3d_rounded[0],          # lateral position
                            "y": centroid_3d_rounded[1],          # vertical position
                            "z": centroid_3d_rounded[2],          # longitudinal position (depth)
                            "orientation": round(float(orientation), 2),  # orientation in radians
                            "time": round(frame_count / fps if fps > 0 else 0, 2)  # time in seconds
                        }
                        
                        # Calculate velocity if we have previous position data for this vehicle
                        if obj_id is not None and obj_id in previous_positions:
                            prev_pos = previous_positions[obj_id]
                            # Time difference between current and previous frame
                            time_delta = vehicle_position["time"] - prev_pos["time"]
                            
                            # Get previous velocity for acceleration calculation
                            prev_velocity = previous_velocities.get(obj_id, None)
                            
                            # Calculate velocity components and acceleration
                            velocity_data = calculate_velocity(prev_pos, vehicle_position, time_delta, prev_velocity)
                            
                            # Store current velocity for next frame's acceleration calculation
                            previous_velocities[obj_id] = velocity_data
                        
                        # Store current position for next frame's velocity calculation
                        if obj_id is not None:
                            previous_positions[obj_id] = vehicle_position
                    
                    # Store object data in the frame data
                    object_data = {
                        "object_id": obj_id if obj_id is not None else -1,
                        "class": class_name,
                        "score": float(score),
                        "bbox_2d": [float(x) for x in bbox],
                        "centroid_2d": [float(x) for x in centroid_2d],
                        "dimensions_2d": [float(x) for x in dimensions_2d],
                        "depth_value": float(depth_value),
                        "centroid_3d": centroid_3d_rounded,
                        "dimensions_3d": dimensions_3d_rounded,
                        "orientation": round(float(orientation), 2)
                    }
                    
                    # Add vehicle position data if it's a vehicle
                    if is_vehicle and vehicle_position:
                        object_data["vehicle_position"] = vehicle_position
                        # Add velocity data if available
                        if velocity_data:
                            object_data["vehicle_velocity"] = velocity_data
                        # Keep track of vehicles for inter-vehicle distance
                        vehicle_objects.append(object_data)
                    
                    frame_data["objects"].append(object_data)
                    
                    boxes_3d.append(full_box_3d)
                    
                    # Keep track of active IDs for tracker cleanup
                    if obj_id is not None:
                        active_ids.append(obj_id)
                except Exception as e:
                    print(f"Error processing detection: {e}")
                    continue
            
            # Calculate inter-vehicle distances if we have multiple vehicles
            if len(vehicle_objects) >= 2:
                # Print number of vehicles for debugging
                print(f"Frame {frame_count}: Found {len(vehicle_objects)} vehicles for distance calculation")
                for i, vehicle in enumerate(vehicle_objects):
                    print(f"  Vehicle {i+1}: ID={vehicle['object_id']}, Class={vehicle['class']}, Position={vehicle['vehicle_position']}")
                
                # Calculate occlusion levels between all pairs of vehicles
                for i in range(len(vehicle_objects)):
                    # Initialize occlusion data for this vehicle
                    vehicle_objects[i]["occlusion"] = {
                        "level": 0.0,
                        "occluded_by": None
                    }
                    
                    # Compare with all other vehicles
                    for j in range(len(vehicle_objects)):
                        if i == j:
                            continue
                            
                        # Calculate occlusion between these two vehicles
                        occlusion_level = calculate_occlusion(
                            vehicle_objects[i]["bbox_2d"],
                            vehicle_objects[j]["bbox_2d"],
                            vehicle_objects[i]["depth_value"],
                            vehicle_objects[j]["depth_value"]
                        )
                        
                        # If significant occlusion is detected
                        if occlusion_level > 0.1:  # More than 10% overlap
                            # Get object IDs
                            id_i = vehicle_objects[i]["object_id"]
                            id_j = vehicle_objects[j]["object_id"]
                            
                            # If depth of vehicle i is greater than vehicle j, then i is occluded by j
                            if vehicle_objects[i]["depth_value"] > vehicle_objects[j]["depth_value"]:
                                if occlusion_level > vehicle_objects[i]["occlusion"]["level"]:
                                    vehicle_objects[i]["occlusion"]["level"] = occlusion_level
                                    vehicle_objects[i]["occlusion"]["occluded_by"] = id_j
                                    print(f"  Vehicle {id_i} is occluded by Vehicle {id_j} at level {occlusion_level:.2f}")
                
                # Find the closest pair of vehicles
                closest_pair = find_closest_vehicle_pair(vehicle_objects)
                
                if closest_pair:
                    idx1, idx2, distance = closest_pair
                    vehicle1 = vehicle_objects[idx1]
                    vehicle2 = vehicle_objects[idx2]
                    
                    print(f"  Closest pair: Vehicle {vehicle1['object_id']} and Vehicle {vehicle2['object_id']} at {distance:.2f}m")
                    
                    # Add inter-vehicle distance to both vehicles in the JSON
                    vehicle1["closest_vehicle"] = {
                        "object_id": vehicle2["object_id"],
                        "distance": round(distance, 2)
                    }
                    
                    vehicle2["closest_vehicle"] = {
                        "object_id": vehicle1["object_id"],
                        "distance": round(distance, 2)
                    }
                    
                    # Draw a line between the two closest vehicles on the result frame
                    try:
                        # Get 2D positions for both vehicles
                        center1 = (int(vehicle1["centroid_2d"][0]), int(vehicle1["centroid_2d"][1]))
                        center2 = (int(vehicle2["centroid_2d"][0]), int(vehicle2["centroid_2d"][1]))
                        
                        # Calculate midpoint for distance text
                        mid_x = (center1[0] + center2[0]) // 2
                        mid_y = (center1[1] + center2[1]) // 2
                        
                        # Draw line between vehicles (yellow) - make it thicker and brighter
                        cv2.line(result_frame, center1, center2, (0, 255, 255), 3)
                        
                        # Draw endpoints with distinctive markers for clarity
                        cv2.drawMarker(result_frame, center1, (0, 165, 255), cv2.MARKER_CROSS, 15, 3)  # Orange cross
                        cv2.drawMarker(result_frame, center2, (0, 165, 255), cv2.MARKER_CROSS, 15, 3)  # Orange cross
                        
                        # Add distance text with background for better visibility
                        distance_text = f"{distance:.2f}m"
                        text_size = cv2.getTextSize(distance_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        
                        # Draw text background
                        cv2.rectangle(
                            result_frame, 
                            (mid_x - text_size[0]//2 - 5, mid_y - text_size[1]//2 - 5),
                            (mid_x + text_size[0]//2 + 5, mid_y + text_size[1]//2 + 5),
                            (0, 0, 0), -1
                        )
                        
                        # Draw text - make it larger and brighter
                        cv2.putText(
                            result_frame, distance_text,
                            (mid_x - text_size[0]//2, mid_y + text_size[1]//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
                        )
                        
                        # Print debug info
                        print(f"Drawing distance line between vehicles: {vehicle1['object_id']} and {vehicle2['object_id']}, distance: {distance:.2f}m")
                    except Exception as e:
                        print(f"Error drawing inter-vehicle distance: {e}")
            
            # Add the frame data to the centroids data
            centroids_data.append(frame_data)
            
            # Clean up trackers for objects that are no longer tracked
            ids_to_remove = []
            for obj_id in previous_positions:
                if obj_id not in active_ids:
                    ids_to_remove.append(obj_id)
            
            for obj_id in ids_to_remove:
                del previous_positions[obj_id]
                if obj_id in previous_velocities:
                    del previous_velocities[obj_id]
            
            bbox3d_estimator.cleanup_trackers(active_ids)
            
            # Step 4: Visualization
            # Draw boxes on the result frame
            for box_3d in boxes_3d:
                try:
                    # Determine color based on class
                    class_name = box_3d['class_name'].lower()
                    if 'car' in class_name or 'vehicle' in class_name:
                        color = (0, 0, 255)  # Red
                    elif 'person' in class_name:
                        color = (0, 255, 0)  # Green
                    elif 'bicycle' in class_name or 'motorcycle' in class_name:
                        color = (255, 0, 0)  # Blue
                    elif 'potted plant' in class_name or 'plant' in class_name:
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (255, 255, 255)  # White
                    
                    # Draw box with depth information
                    result_frame = bbox3d_estimator.draw_box_3d(result_frame, box_3d, color=color)
                    
                    # Draw 2D centroid as a filled circle - make it larger and more visible
                    centroid_2d = box_3d['centroid_2d']
                    cx, cy = int(centroid_2d[0]), int(centroid_2d[1])
                    # Draw a larger, more visible circle
                    cv2.circle(result_frame, (cx, cy), 7, (255, 255, 0), -1)  # Filled yellow circle
                    cv2.circle(result_frame, (cx, cy), 7, (0, 0, 0), 2)  # Thicker black outline
                    
                    # Display centroid coordinates near the point
                    centroid_3d = box_3d['centroid_3d']
                    centroid_text = f"({centroid_3d[0]:.2f}, {centroid_3d[1]:.2f}, {centroid_3d[2]:.2f})"
                    
                    # Add background to text for better visibility
                    text_size = cv2.getTextSize(centroid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(
                        result_frame,
                        (cx + 10, cy - text_size[1] - 5),
                        (cx + 10 + text_size[0], cy + 5),
                        (0, 0, 0), -1
                    )
                    
                    cv2.putText(result_frame, centroid_text, 
                               (cx + 10, cy), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                except Exception as e:
                    print(f"Error drawing box: {e}")
                    continue
            
            # Draw BEV
            try:
                # Reset BEV and draw objects
                bev.reset()
                for box_3d in boxes_3d:
                    # Convert depth to float if it's an integer
                    if isinstance(box_3d['depth_value'], int):
                        box_3d['depth_value'] = float(box_3d['depth_value'])
                    bev.draw_box(box_3d)
                
                # Get BEV image
                bev_frame = bev.get_image()
                
                # Add centroids to the BEV image
                for box_3d in boxes_3d:
                    try:
                        # Calculate position in BEV
                        depth = float(box_3d['depth_value'])
                        # Map depth value (0-1) to a range of 1-5 meters
                        depth_meters = 1.0 + depth * 4.0
                        
                        # Calculate BEV coordinates similar to what's done in the BirdEyeView.draw_box method
                        bev_y = bev.origin_y - int(depth_meters * bev.scale)
                        
                        # Get the 2D centroid for x-position
                        centroid_2d = box_3d['centroid_2d']
                        rel_x = (centroid_2d[0] / width) - 0.5
                        bev_x = bev.origin_x + int(rel_x * bev.width * 0.6)
                        
                        # Ensure the point stays within the visible area
                        bev_x = max(10, min(bev_x, bev.width - 10))
                        bev_y = max(10, min(bev_y, bev.origin_y - 10))
                        
                        # Draw centroid on BEV as a star marker with label
                        draw_star(bev_frame, bev_x, bev_y, size=8, color=(255, 255, 0))
                        
                        # Get object ID if available
                        obj_id = box_3d.get('object_id', None)
                        if obj_id is not None:
                            cv2.putText(bev_frame, f"ID:{obj_id}", 
                                      (bev_x + 10, bev_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    except Exception as e:
                        print(f"Error drawing centroid on BEV: {e}")
                
                # Create a copy of the result frame for the output video
                output_frame = result_frame.copy()
                
                if bev_frame is not None:
                    # Convert to uint8 if needed
                    if bev_frame.dtype != np.uint8:
                        bev_frame = (bev_frame * 255).astype(np.uint8)
                    # Resize to match frame dimensions
                    bev_frame = cv2.resize(bev_frame, (width, height))
                    # Stack frames horizontally
                    try:
                        combined_frame = np.hstack((output_frame, bev_frame))
                        # Ensure the combined frame has the correct dimensions
                        if combined_frame.shape[1] != total_width:
                            print(f"Warning: Combined frame width ({combined_frame.shape[1]}) doesn't match expected width ({total_width})")
                            # Resize to match expected dimensions
                            combined_frame = cv2.resize(combined_frame, (total_width, height))
                        # Write the combined frame
                        out.write(combined_frame)
                        
                        # Add debug info for the output
                        print(f"Frame {frame_count}: Successfully wrote combined frame with size {combined_frame.shape}")
                    except Exception as e:
                        print(f"Error creating combined frame: {e}")
                        # Fallback: create a blank frame of the correct size
                        combined_frame = np.zeros((height, total_width, 3), dtype=np.uint8)
                        # Copy the original frame to the left side
                        combined_frame[:, :width, :] = output_frame
                        out.write(combined_frame)
                else:
                    # If BEV frame is None, create a blank frame for the right side
                    combined_frame = np.zeros((height, total_width, 3), dtype=np.uint8)
                    # Copy the original frame to the left side
                    combined_frame[:, :width, :] = output_frame
                    out.write(combined_frame)
            except Exception as e:
                print(f"Error in BEV visualization: {e}")
                # Create a safe frame to write
                safe_frame = np.zeros((height, total_width, 3), dtype=np.uint8)
                if frame.shape[0] == height and frame.shape[1] == width:
                    safe_frame[:, :width, :] = frame
                out.write(safe_frame)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 10 == 0:  # Update FPS every 10 frames
                end_time = time.time()
                elapsed_time = end_time - start_time
                fps_value = frame_count / elapsed_time
                fps_display = f"FPS: {fps_value:.1f}"
            
            # Add FPS and device info to the result frame
            cv2.putText(result_frame, f"{fps_display} | Device: {device}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            
            # Add depth map to the corner of the result frame
            try:
                depth_height = height // 4
                depth_width = depth_height * width // height
                depth_resized = cv2.resize(depth_colored, (depth_width, depth_height))
                result_frame[0:depth_height, 0:depth_width] = depth_resized
            except Exception as e:
                print(f"Error adding depth map to result: {e}")
            
            # Display frames
            cv2.imshow("3D Object Detection", result_frame)
            cv2.imshow("Depth Map", depth_colored)
            cv2.imshow("Object Detection", detection_frame)
            
            # Check for key press again at the end of the loop
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                print("Exiting program...")
                break
        
        except Exception as e:
            print(f"Error processing frame: {e}")
            # Also check for key press during exception handling
            key = cv2.waitKey(1)
            if key == ord('q') or key == 27 or (key & 0xFF) == ord('q') or (key & 0xFF) == 27:
                print("Exiting program...")
                break
            continue
    
    # Clean up
    print("Cleaning up resources...")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Save centroids data to JSON file
    json_output_path = os.path.splitext(output_path)[0] + "_centroids.json"
    with open(json_output_path, 'w') as f:
        json.dump(centroids_data, f, indent=2)
    
    print(f"Processing complete. Output saved to {output_path}")
    print(f"Centroids data saved to {json_output_path}")
    print("Important: If the output video doesn't play properly, try converting it with:")
    print(f"ffmpeg -i {output_path} -c:v libx264 -preset medium -crf 23 converted_output.mp4")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user (Ctrl+C)")
        # Clean up OpenCV windows
        cv2.destroyAllWindows() 