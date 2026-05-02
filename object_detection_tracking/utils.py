"""
Utility functions for drawing detections and statistics on video frames.
"""

import cv2
import time
import numpy as np
from typing import Dict, Tuple, List, Any

LIVING_COLOR = (0, 0, 220)    # Red   in BGR (OpenCV format)
NONLIVING_COLOR = (160, 160, 160) # Gray  in BGR

def draw_rounded_rect(img, pt1, pt2, color, thickness, r, d):
    """Draws a rectangle with rounded corners."""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Draw straight lines
    cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
    cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)

    # Draw arcs
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

def draw_detections(frame: np.ndarray, tracked_objects: Dict[int, Tuple[int, int, List[int]]], class_names: Dict[int, str], confidences: Dict[int, float], is_living_dict: Dict[int, bool]):
    """
    Draws bounding boxes, labels, and tracking IDs on the frame.
    
    Args:
        frame: The OpenCV image/frame.
        tracked_objects: Dict from tracker mapping ID to (cX, cY, bbox).
        class_names: Dict mapping object ID to class name.
        confidences: Dict mapping object ID to confidence score.
        is_living_dict: Dict mapping object ID to living boolean.
    """
    for object_id, info in tracked_objects.items():
        cx, cy, bbox = info
        x1, y1, x2, y2 = bbox
        
        class_name = class_names.get(object_id, "unknown")
        is_living = is_living_dict.get(object_id, False)
        
        color = LIVING_COLOR if is_living else NONLIVING_COLOR
        icon = "🔴" if is_living else "⬜"
        
        # Draw smooth rounded corner boxes
        draw_rounded_rect(frame, (x1, y1), (x2, y2), color, 2, 10, 10)
        
        # Label format: "Dog #4 🔴" for living, "Chair #7 ⬜" for non-living
        label = f"{class_name.capitalize()} #{object_id} {icon}"
            
        # Filled label background rectangle for readability
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        
        # Text
        cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def calculate_fps(prev_time: float) -> Tuple[float, float]:
    """
    Calculates current FPS.
    
    Args:
        prev_time: Timestamp of the previous frame.
        
    Returns:
        Tuple[float, float]: (current_fps, current_time)
    """
    current_time = time.time()
    fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0.0
    return fps, current_time

def draw_fps(frame: np.ndarray, fps: float):
    """
    Overlays FPS counter on top-left corner.
    """
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def draw_stats(frame: np.ndarray, living_counts: Dict[str, int], nonliving_counts: Dict[str, int]):
    """
    Draws a small panel bottom-left showing count per class.
    
    Args:
        frame: OpenCV image/frame.
        living_counts: Dictionary mapping living class names to counts.
        nonliving_counts: Dictionary mapping non-living class names to counts.
    """
    h, w = frame.shape[:2]
    
    living_text = "🔴 Living  →  " + "  ".join([f"{k.capitalize()}s: {v}" for k, v in living_counts.items() if v > 0])
    if not living_counts:
        living_text = "🔴 Living  →  None"
        
    nonliving_text = "⬜ Non-living  →  " + "  ".join([f"{k.capitalize()}s: {v}" for k, v in nonliving_counts.items() if v > 0])
    if not nonliving_counts:
        nonliving_text = "⬜ Non-living  →  None"
        
    # Draw semi-transparent background panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, h - 70), (w - 10, h - 10), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Draw text (red for living, gray for non-living)
    cv2.putText(frame, living_text, (20, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LIVING_COLOR, 1)
    cv2.putText(frame, nonliving_text, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, NONLIVING_COLOR, 1)
