"""
Advanced Clock Time Recognition System
Utilizes computer vision techniques for precise time detection from analog clocks.
"""
import cv2
import numpy as np
import math
from typing import Tuple, Optional, List


def enhance_hand_visibility(image_bgr, region_mask=None):
    """Advanced preprocessing pipeline for optimal hand detection."""
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply contrast limited adaptive histogram equalization
    clahe_processor = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast_enhanced = clahe_processor.apply(grayscale)
    
    # Gentle smoothing to reduce noise artifacts
    smoothed = cv2.GaussianBlur(contrast_enhanced, (5, 5), 1)
    
    # Morphological operations for texture suppression
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphologically_closed = cv2.morphologyEx(smoothed, cv2.MORPH_CLOSE, structuring_element)
    
    # Intelligent adaptive thresholding
    binary_result = cv2.adaptiveThreshold(morphologically_closed, 255,
                                          cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 25, 8)
    
    # Apply region constraint if provided
    if region_mask is not None:
        binary_result = cv2.bitwise_and(binary_result, binary_result, mask=region_mask)
    
    # Final noise reduction
    binary_result = cv2.morphologyEx(binary_result, cv2.MORPH_OPEN, structuring_element)
    
    return binary_result


def extract_line_skeleton(binary_image):
    """Extract skeletal structure from binary image using morphological operations."""
    working_image = binary_image.copy()
    working_image[working_image > 0] = 1
    skeleton_result = np.zeros(working_image.shape, np.uint8)
    cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    
    iteration_count = 0
    while True:
        iteration_count += 1
        eroded_image = cv2.erode(working_image, cross_kernel)
        dilated_temp = cv2.dilate(eroded_image, cross_kernel)
        skeleton_component = cv2.subtract(working_image, dilated_temp)
        skeleton_result = cv2.bitwise_or(skeleton_result, skeleton_component)
        working_image = eroded_image.copy()
        
        if cv2.countNonZero(working_image) == 0:
            break
    
    return (skeleton_result * 255).astype(np.uint8)


def locate_clock_boundary(image_input):
    """
    Locate clock boundary using advanced ellipse fitting techniques.
    Returns (center_x, center_y, major_axis, minor_axis, rotation_angle) if found, None otherwise.
    """
    grayscale_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY)
    noise_reduced = cv2.GaussianBlur(grayscale_input, (5, 5), 1.5)
    edge_map = cv2.Canny(noise_reduced, 50, 150)
    
    # Extract contours and perform ellipse fitting
    detected_contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_ellipses = []
    image_height, image_width = image_input.shape[:2]
    
    for contour in detected_contours:
        if len(contour) >= 5:
            fitted_ellipse = cv2.fitEllipse(contour)
            (ellipse_x, ellipse_y), (major_axis, minor_axis), rotation_angle = fitted_ellipse
            aspect_ratio = major_axis / minor_axis
            if 0.6 < aspect_ratio < 1.4 and 50 < major_axis < image_height:
                candidate_ellipses.append(((ellipse_x, ellipse_y), (major_axis, minor_axis), rotation_angle))
    
    if not candidate_ellipses:
        return None
    
    # Select primary ellipse (closest to image center)
    image_center = (image_width / 2, image_height / 2)
    primary_ellipse = min(candidate_ellipses, 
                        key=lambda e: math.hypot(e[0][0] - image_center[0], e[0][1] - image_center[1]))
    (center_x, center_y), (major_axis, minor_axis), rotation_angle = primary_ellipse
    
    return (int(center_x), int(center_y), int(major_axis), int(minor_axis), int(rotation_angle))


def identify_hand_segments(image_input, ellipse_parameters):
    """
    Identify clock hand segments using advanced skeletonization and line detection.
    Returns list of detected line segments.
    """
    center_x, center_y, major_axis, minor_axis, rotation_angle = ellipse_parameters
    image_height, image_width = image_input.shape[:2]
    
    # Create elliptical region mask
    region_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    cv2.ellipse(region_mask, (center_x, center_y), (major_axis // 2, minor_axis // 2), 
                rotation_angle, 0, 360, 255, -1)
    
    # Apply advanced preprocessing
    processed_image = enhance_hand_visibility(image_input, region_mask)
    if cv2.countNonZero(processed_image) < 50:
        processed_image = enhance_hand_visibility(image_input)
    
    # Extract skeletal structure
    skeleton_image = extract_line_skeleton(processed_image)
    
    # Detect line segments using probabilistic Hough transform with optimized parameters
    detected_lines = cv2.HoughLinesP(skeleton_image, 1, np.pi / 180, 25,  # Reduced threshold for better detection
                                    minLineLength=int(min(major_axis, minor_axis) * 0.12),  # Slightly shorter minimum length
                                    maxLineGap=8)  # Reduced gap for better continuity
    
    if detected_lines is None:
        return []
    
    # Filter segments within elliptical boundary
    valid_segments = []
    for line_segment in detected_lines:
        x1, y1, x2, y2 = line_segment[0]
        midpoint_x, midpoint_y = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Transform coordinates to ellipse coordinate system
        transformed_x = (midpoint_x - center_x) * math.cos(math.radians(rotation_angle)) + \
                       (midpoint_y - center_y) * math.sin(math.radians(rotation_angle))
        transformed_y = -(midpoint_x - center_x) * math.sin(math.radians(rotation_angle)) + \
                       (midpoint_y - center_y) * math.cos(math.radians(rotation_angle))
        
        # Check if segment is within ellipse
        ellipse_constraint = (transformed_x ** 2) / ((major_axis / 2) ** 2) + \
                           (transformed_y ** 2) / ((minor_axis / 2) ** 2)
        if ellipse_constraint <= 1:
            valid_segments.append((x1, y1, x2, y2))
    
    return valid_segments


def group_hand_candidates(line_segments, ellipse_parameters):
    """
    Group hand candidates by directional similarity and return clustered data.
    Returns list of (angle, length, tip_position) tuples.
    """
    if not line_segments:
        return []
    
    center_x, center_y, major_axis, minor_axis, rotation_angle = ellipse_parameters
    candidate_data = []
    
    for (x1, y1, x2, y2) in line_segments:
        direction_vector = (x2 - x1, y2 - y1)
        angle_degrees = (math.degrees(math.atan2(direction_vector[1], direction_vector[0])) + 360) % 360
        segment_length = math.hypot(direction_vector[0], direction_vector[1])
        
        # Determine tip position (farther from center)
        distance_from_center_1 = math.hypot(x1 - center_x, y1 - center_y)
        distance_from_center_2 = math.hypot(x2 - center_x, y2 - center_y)
        tip_position = (x1, y1) if distance_from_center_1 > distance_from_center_2 else (x2, y2)
        
        candidate_data.append({
            'angle': angle_degrees, 
            'vector': direction_vector, 
            'length': segment_length, 
            'tip': tip_position
        })
    
    ANGLE_TOLERANCE = 12  # Reduced tolerance for more precise clustering
    processed_flags = [False] * len(candidate_data)
    clustered_groups = []
    
    for i, primary_candidate in enumerate(candidate_data):
        if processed_flags[i]:
            continue
        
        current_group = [primary_candidate]
        processed_flags[i] = True
        
        for j, secondary_candidate in enumerate(candidate_data):
            if not processed_flags[j]:
                angle_difference = abs(primary_candidate['angle'] - secondary_candidate['angle'])
                angle_difference = min(angle_difference, 360 - angle_difference)  # Handle wraparound
                
                if angle_difference <= ANGLE_TOLERANCE:
                    current_group.append(secondary_candidate)
                    processed_flags[j] = True
        
        # Calculate group statistics
        total_group_length = sum(candidate['length'] for candidate in current_group)
        average_group_angle = sum(candidate['angle'] for candidate in current_group) / len(current_group)
        average_tip_position = np.mean([candidate['tip'] for candidate in current_group], axis=0)
        
        clustered_groups.append((average_group_angle, total_group_length, tuple(average_tip_position)))
    
    # Sort by length (longer segments are more likely to be hands)
    clustered_groups.sort(key=lambda group: group[1], reverse=True)
    
    return clustered_groups


def compute_time_from_hand_data(hand_groups, ellipse_parameters):
    """
    Compute clock time from detected hand groups.
    Returns (hour, minute) if successful, None otherwise.
    """
    if len(hand_groups) < 1:
        return None
    
    center_x, center_y, major_axis, minor_axis, rotation_angle = ellipse_parameters
    
    # Extract primary and secondary hand data with improved logic
    if len(hand_groups) >= 2:
        # Use the two longest segments
        minute_angle, minute_length, minute_tip = hand_groups[0]
        hour_angle, hour_length, hour_tip = hand_groups[1]
        
        # Double-check: ensure minute hand is actually longer
        if minute_length < hour_length:
            minute_angle, minute_length, minute_tip = hand_groups[1]
            hour_angle, hour_length, hour_tip = hand_groups[0]
    else:
        # Only one hand detected - assume it's the minute hand
        minute_angle, minute_length, minute_tip = hand_groups[0]
        hour_angle, hour_length, hour_tip = hand_groups[0]
    
    # Normalize vectors
    def normalize_vector(vector):
        magnitude = math.hypot(vector[0], vector[1]) + 1e-12
        return (vector[0] / magnitude, vector[1] / magnitude)
    
    minute_direction = normalize_vector((minute_tip[0] - center_x, minute_tip[1] - center_y))
    hour_direction = normalize_vector((hour_tip[0] - center_x, hour_tip[1] - center_y))
    
    # Calculate hand lengths
    minute_hand_length = min(minute_length, major_axis * 0.9)
    hour_hand_length = min(hour_length, major_axis * 0.9)
    
    # Calculate hand endpoints
    minute_endpoint = (int(center_x + minute_direction[0] * minute_hand_length), 
                      int(center_y + minute_direction[1] * minute_hand_length))
    hour_endpoint = (int(center_x + hour_direction[0] * hour_hand_length), 
                    int(center_y + hour_direction[1] * hour_hand_length))
    
    # Convert to clock angles (clockwise from 12 o'clock)
    def convert_to_clock_angle(vector):
        # vector is (x, y) from center to tip
        # math.atan2(y, x) gives angle from positive x-axis, counter-clockwise
        math_angle = math.degrees(math.atan2(vector[1], vector[0]))
        
        # Convert to clockwise angle from 12 o'clock (top) for image coordinates
        # In image coordinates: Y points down, so we need to adjust
        # math_angle: -90° = 12 o'clock, 0° = 3 o'clock, 90° = 6 o'clock, 180° = 9 o'clock
        # clock_angle: 0° = 12 o'clock, 90° = 3 o'clock, 180° = 6 o'clock, 270° = 9 o'clock
        clock_angle = (math_angle + 90 + 360) % 360
        return clock_angle
    
    minute_clock_angle = convert_to_clock_angle((minute_endpoint[0] - center_x, minute_endpoint[1] - center_y))
    hour_clock_angle = convert_to_clock_angle((hour_endpoint[0] - center_x, hour_endpoint[1] - center_y))
    
    # Calculate time values with improved precision
    detected_minutes = int(round(minute_clock_angle / 6)) % 60
    detected_hours = int(round(hour_clock_angle / 30)) % 12
    
    # Handle hour correction for 12-hour format
    if detected_hours == 0:
        detected_hours = 12
    
    # Additional validation: ensure minute hand is longer than hour hand
    if minute_length < hour_length:
        # Swap if minute hand appears shorter (likely detection error)
        detected_hours, detected_minutes = detected_minutes // 5, detected_hours * 5
        if detected_hours == 0:
            detected_hours = 12
        detected_minutes = min(detected_minutes, 59)
    
    return detected_hours, detected_minutes


def detect_clock_time(img: np.ndarray) -> Tuple[Optional[int], Optional[int], np.ndarray]:
    """
    Main function to detect clock time from image.
    Returns (hour, minute, result_image_with_detections).
    """
    # Locate clock boundary using ellipse detection
    ellipse_parameters = locate_clock_boundary(img)
    if ellipse_parameters is None:
        return None, None, img.copy()
    
    center_x, center_y, major_axis, minor_axis, rotation_angle = ellipse_parameters
    
    # Identify hand segments
    hand_segments = identify_hand_segments(img, ellipse_parameters)
    if not hand_segments:
        return None, None, img.copy()
    
    # Group hand candidates
    hand_groups = group_hand_candidates(hand_segments, ellipse_parameters)
    if not hand_groups:
        return None, None, img.copy()
    
    # Compute time
    time_result = compute_time_from_hand_data(hand_groups, ellipse_parameters)
    if time_result is None:
        return None, None, img.copy()
    
    detected_hours, detected_minutes = time_result
    
    # Create visualization
    visualization_image = img.copy()
    
    # Draw clock boundary ellipse
    cv2.ellipse(visualization_image, (center_x, center_y), (major_axis // 2, minor_axis // 2), 
                rotation_angle, 0, 360, (255, 255, 0), 2)  # Cyan ellipse
    
    # Draw detected hands
    minute_angle, minute_length, minute_tip = hand_groups[0]
    hour_angle, hour_length, hour_tip = hand_groups[1] if len(hand_groups) > 1 else hand_groups[0]
    
    def normalize_vector(vector):
        magnitude = math.hypot(vector[0], vector[1]) + 1e-12
        return (vector[0] / magnitude, vector[1] / magnitude)
    
    minute_direction = normalize_vector((minute_tip[0] - center_x, minute_tip[1] - center_y))
    hour_direction = normalize_vector((hour_tip[0] - center_x, hour_tip[1] - center_y))
    
    minute_hand_length = min(minute_length, major_axis * 0.9)
    hour_hand_length = min(hour_length, major_axis * 0.9)
    
    clock_center = (center_x, center_y)
    minute_endpoint = (int(center_x + minute_direction[0] * minute_hand_length), 
                      int(center_y + minute_direction[1] * minute_hand_length))
    hour_endpoint = (int(center_x + hour_direction[0] * hour_hand_length), 
                    int(center_y + hour_direction[1] * hour_hand_length))
    
    # Draw hands with distinct colors
    cv2.line(visualization_image, clock_center, minute_endpoint, (0, 255, 255), 5)  # Yellow minute hand
    cv2.line(visualization_image, clock_center, hour_endpoint, (255, 0, 255), 7)   # Magenta hour hand
    
    # Draw center point
    cv2.circle(visualization_image, clock_center, 5, (0, 255, 0), -1)  # Green center
    
    # Add time display
    time_display_text = f"Detected Time: {detected_hours:02d}:{detected_minutes:02d}"
    cv2.putText(visualization_image, time_display_text, (30, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return detected_hours, detected_minutes, visualization_image
