import cv2
from typing import List, Dict

class CoordinateNormalizer:
    def __init__(self, image_path: str):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        self.height, self.width = self.image.shape[:2]
    
    def normalize_bbox(self, bbox: List[int], input_format: str = 'xywh') -> List[float]:
        if input_format == 'xywh':
            x, y, w, h = bbox
            x1, y1, x2, y2 = x, y, x + w, y + h
        elif input_format == 'xyxy':
            x1, y1, x2, y2 = bbox
        else:
            raise ValueError("input_format must be 'xywh' or 'xyxy'")
        
        return [x1/self.width, y1/self.height, x2/self.width, y2/self.height]
    
    def denormalize_bbox(self, norm_bbox: List[float], output_format: str = 'xywh') -> List[int]:
        norm_x1, norm_y1, norm_x2, norm_y2 = norm_bbox
        x1 = int(norm_x1 * self.width)
        y1 = int(norm_y1 * self.height)
        x2 = int(norm_x2 * self.width)
        y2 = int(norm_y2 * self.height)
        
        if output_format == 'xywh':
            return [x1, y1, x2 - x1, y2 - y1]
        else:
            return [x1, y1, x2, y2]
    
    def merge_layouts(self, layout_regions: List[str], table_structures: List) -> List[str]:
        all_regions = []
        
        # Process layout regions
        for region_str in layout_regions:
            parts = region_str.strip().split(',')
            if len(parts) >= 5:
                x, y, w, h, line = map(int, parts[:5])
                norm_bbox = self.normalize_bbox([x, y, w, h], 'xywh')
                all_regions.append({
                    'bbox_norm': norm_bbox,
                    'line': line,
                    'type': 'text'
                })
        
        # Process table structures
        for soup, table_bbox, cell_coords in table_structures:
            for cell in cell_coords:
                if len(cell) >= 4:
                    x1, y1, x2, y2 = cell[:4]
                    norm_bbox = self.normalize_bbox([x1, y1, x2, y2], 'xyxy')
                    all_regions.append({
                        'bbox_norm': norm_bbox,
                        'line': 0,
                        'type': 'table'
                    })
        
        # Sort by reading order
        all_regions.sort(key=lambda r: (r['bbox_norm'][1], r['bbox_norm'][0]))
        
        # Convert back to strings
        result = []
        for region in all_regions:
            x, y, w, h = self.denormalize_bbox(region['bbox_norm'], 'xywh')
            result.append(f"{x},{y},{w},{h},{region['line']}")
        
        return result