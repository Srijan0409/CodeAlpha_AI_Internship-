from typing import Optional, Tuple
import cv2
import numpy as np

class DetectionHeatmap:
    """
    Accumulates object detection centroids into a 2D spatial heatmap.
    
    Each detection adds a Gaussian blob to the heatmap grid at the centroid position.
    The heatmap is normalized and rendered as a JET colormap overlay on the video frame.
    This visualizes where objects appear most frequently over the session \u2014 a form of 
    spatial frequency analysis useful in security and traffic monitoring.
    
    Attributes:
        h (int): The height of the video frame.
        w (int): The width of the video frame.
        decay (float): The temporal decay factor applied each frame to fade out older detections.
        grid (np.ndarray): The 2D float accumulator array maintaining the heatmap intensities.
        total_detections (int): The overall sum of detections registered to the heatmap.
        enabled (bool): State flag controlling whether the heatmap accumulates and renders.
    """
    
    def __init__(self, frame_height: int, frame_width: int, decay: float = 0.998):
        """
        Initializes the DetectionHeatmap.
        
        Args:
            frame_height (int): The height of the video frames being processed.
            frame_width (int): The width of the video frames being processed.
            decay (float): The multiplier applied per frame to naturally fade historical data.
        """
        self.h = frame_height
        self.w = frame_width
        self.decay = decay  # Multiply heatmap by this each frame \u2014 older detections fade
        
        # 2D float accumulator grid \u2014 same size as frame
        self.grid = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Gaussian kernel for spreading each detection into a blob
        self._kernel_size = 61  # must be odd
        self._sigma = 20.0
        self._kernel = self._build_gaussian_kernel()
        
        self.total_detections = 0
        self.enabled = True
        print(f"Heatmap initialized: {frame_width}x{frame_height}, decay={decay}")
    
    def _build_gaussian_kernel(self) -> np.ndarray:
        """
        Constructs a 2D Gaussian kernel used to distribute the 'heat' of a single detection.
        
        Returns:
            np.ndarray: A 2D normalized numpy array containing the Gaussian distribution.
        """
        k = self._kernel_size
        center = k // 2
        kernel = np.zeros((k, k), dtype=np.float32)
        for y in range(k):
            for x in range(k):
                dist_sq = (x - center)**2 + (y - center)**2
                kernel[y, x] = np.exp(-dist_sq / (2 * self._sigma**2))
        return kernel / kernel.max()  # normalize to [0, 1]
    
    def add_detection(self, cx: int, cy: int, confidence: float = 1.0) -> None:
        """
        Registers a new detection point to the heatmap grid.
        
        Adds a Gaussian blob centered at the coordinate (cx, cy), weighted by the confidence.
        
        Args:
            cx (int): The x-coordinate of the detection centroid.
            cy (int): The y-coordinate of the detection centroid.
            confidence (float): The confidence score of the detection, used as a multiplier.
        """
        if not self.enabled:
            return
        
        k = self._kernel_size
        half = k // 2
        
        # Compute crop region \u2014 handle edge of frame cases
        x1 = cx - half
        y1 = cy - half
        x2 = cx + half + 1
        y2 = cy + half + 1
        
        # Clamp to frame bounds
        kx1 = max(0, -x1)
        ky1 = max(0, -y1)
        kx2 = k - max(0, x2 - self.w)
        ky2 = k - max(0, y2 - self.h)
        
        fx1 = max(0, x1)
        fy1 = max(0, y1)
        fx2 = min(self.w, x2)
        fy2 = min(self.h, y2)
        
        if fx2 > fx1 and fy2 > fy1:
            self.grid[fy1:fy2, fx1:fx2] += self._kernel[ky1:ky2, kx1:kx2] * confidence
            self.total_detections += 1
    
    def decay_frame(self) -> None:
        """
        Applies a temporal decay multiplier to the entire heatmap grid.
        
        This forces older detection blobs to fade away gradually, ensuring the heatmap 
        represents recent spatial activity rather than an infinite accumulation.
        """
        self.grid *= self.decay
    
    def render(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        """
        Renders the accumulated heatmap as a semi-transparent color overlay on the video frame.
        
        Normalizes the intensity grid to [0, 255] and maps it through a JET colormap.
        
        Args:
            frame (np.ndarray): The original video frame (BGR).
            alpha (float): The blending factor for the heatmap overlay (0.0 to 1.0).
            
        Returns:
            np.ndarray: A new image array with the heatmap blended over the original frame.
        """
        if not self.enabled or self.total_detections == 0:
            return frame
        
        # Normalize grid to 0-255 range
        grid_max = self.grid.max()
        if grid_max < 0.001:
            return frame  # nothing accumulated yet
        
        normalized = (self.grid / grid_max * 255).astype(np.uint8)
        
        # Apply JET colormap \u2014 blue (rare) \u2192 green \u2192 yellow \u2192 red (hotspot)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
        
        # Mask: only show heatmap where grid value is above a threshold
        mask = normalized > 15  # threshold \u2014 ignore very faint areas
        mask_3ch = np.stack([mask, mask, mask], axis=2)
        
        # Blend: heatmap only where mask is True, original frame elsewhere
        result = frame.copy()
        result[mask_3ch] = cv2.addWeighted(colored, alpha, frame, 1 - alpha, 0)[mask_3ch]
        
        return result
    
    def reset(self) -> None:
        """
        Clears the heatmap accumulator array, destroying all historical spatial data.
        """
        self.grid = np.zeros((self.h, self.w), dtype=np.float32)
        self.total_detections = 0
        print("Heatmap reset")
    
    def toggle(self) -> None:
        """
        Toggles the enabled state of the heatmap generator and renderer.
        """
        self.enabled = not self.enabled
        print(f"Heatmap: {'ON' if self.enabled else 'OFF'}")
    
    def get_hotspot(self) -> Optional[Tuple[int, int]]:
        """
        Identifies the singular hottest coordinate point currently in the heatmap.
        
        Returns:
            Optional[Tuple[int, int]]: The (x, y) coordinates of the peak intensity, or None if empty.
        """
        if self.total_detections == 0:
            return None
        idx = np.unravel_index(np.argmax(self.grid), self.grid.shape)
        return (idx[1], idx[0])  # return as (x, y)
