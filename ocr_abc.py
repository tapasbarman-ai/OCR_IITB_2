# import os
# import re
# import json
# import argparse
# from typing import List, Tuple, Dict, Set

# # ----------------------------
# # Geometry helpers
# # ----------------------------
# def bbox_intersection_area(b1: Tuple[int, int, int, int],
#                            b2: Tuple[int, int, int, int]) -> int:
#     x1 = max(b1[0], b2[0])
#     y1 = max(b1[1], b2[1])
#     x2 = min(b1[2], b2[2])
#     y2 = min(b1[3], b2[3])
#     if x2 <= x1 or y2 <= y1:
#         return 0
#     return (x2 - x1) * (y2 - y1)

# def bbox_area(b: Tuple[int, int, int, int]) -> int:
#     return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

# def bbox_overlap_ratio(word_bbox: Tuple[int, int, int, int], 
#                       cell_bbox: Tuple[int, int, int, int]) -> float:
#     """Calculate what percentage of the word overlaps with the cell"""
#     intersection = bbox_intersection_area(word_bbox, cell_bbox)
#     word_area = bbox_area(word_bbox)
#     return intersection / word_area if word_area > 0 else 0.0

# def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
#     """Get center point of bbox"""
#     return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

# _INT_RE = re.compile(r"-?\d+")
# _TD_WITH_BBOX_RE = re.compile(
#     r"<td\b[^>]*\bbbox\s*=\s*\"([^\"]+)\"[^>]*>.*?</td>",
#     re.IGNORECASE | re.DOTALL
# )

# def parse_bbox_from_string(s: str) -> Tuple[int, int, int, int]:
#     """Parse bbox string in x1,y1,x2,y2 format (for HTML td elements)"""
#     nums = _INT_RE.findall(s)
#     if len(nums) < 4:
#         raise ValueError(f"Could not parse 4 integers from bbox string: {s!r}")
#     return tuple(map(int, nums[:4]))

# def parse_layout_line(line: str) -> Tuple[Tuple[int, int, int, int], int]:
#     """
#     Parse layout.txt line in format: x y w h line_number
#     Returns: ((x1, y1, x2, y2), line_number)
#     """
#     nums = _INT_RE.findall(line.strip())
#     if len(nums) < 5:
#         raise ValueError(f"Could not parse 5 integers from layout line: {line!r}")
    
#     x, y, w, h, line_num = map(int, nums[:5])
#     # Convert from (x, y, w, h) to (x1, y1, x2, y2)
#     x1, y1 = x, y
#     x2, y2 = x + w, y + h
    
#     return ((x1, y1, x2, y2), line_num)

# def load_words_for_folder(folder_path: str) -> List[Tuple[Tuple[int,int,int,int], str, int]]:
#     """Load words with their bboxes and line numbers"""
#     layout_path = os.path.join(folder_path, "layout.txt")
#     ocr_path = os.path.join(folder_path, "ocr.txt")

#     if not os.path.isfile(layout_path) or not os.path.isfile(ocr_path):
#         raise FileNotFoundError(f"Missing layout.txt or ocr.txt in {folder_path}")

#     with open(layout_path, "r", encoding="utf-8") as f:
#         layout_lines = [ln.strip() for ln in f if ln.strip()]
#     with open(ocr_path, "r", encoding="utf-8") as f:
#         ocr_lines = [ln.rstrip("\n") for ln in f]

#     if len(layout_lines) != len(ocr_lines):
#         min_len = min(len(layout_lines), len(ocr_lines))
#         print(f"[WARN] {folder_path}: layout({len(layout_lines)}) != ocr({len(ocr_lines)}). Truncating to {min_len}.")
#         layout_lines = layout_lines[:min_len]
#         ocr_lines = ocr_lines[:min_len]

#     words = []
#     for i, (layout_line, word) in enumerate(zip(layout_lines, ocr_lines)):
#         try:
#             bbox, line_num = parse_layout_line(layout_line)
#             if word.strip():  # Only add non-empty words
#                 words.append((bbox, word.strip(), line_num))
#         except Exception as e:
#             print(f"[WARN] Skipping invalid layout line #{i} in {folder_path}: {layout_line!r} ({e})")
#             continue
    
#     return words

# def extract_table_structure(html: str) -> List[Dict]:
#     """Extract the table cell structure with bboxes, completely ignoring existing content"""
#     cells = []
    
#     # Find all td elements with bbox attributes
#     pattern = r'<td\b[^>]*\bbbox\s*=\s*"([^"]+)"[^>]*>'
    
#     for match in re.finditer(pattern, html, re.IGNORECASE):
#         bbox_str = match.group(1)
#         try:
#             bbox = parse_bbox_from_string(bbox_str)
#             cells.append({
#                 'bbox': bbox,
#                 'bbox_str': bbox_str,
#                 'start_pos': match.start(),
#                 'original_tag': match.group(0)
#             })
#         except Exception as e:
#             print(f"[WARN] Could not parse bbox {bbox_str}: {e}")
#             continue
    
#     # Sort cells by position (top-to-bottom, left-to-right)
#     cells.sort(key=lambda c: (c['bbox'][1], c['bbox'][0]))
    
#     return cells

# def assign_words_to_cells(words: List[Tuple[Tuple[int,int,int,int], str, int]], 
#                          cells: List[Dict],
#                          overlap_threshold: float = 0.5,
#                          debug: bool = True) -> Dict[int, List[str]]:
#     """
#     Assign words to table cells using intelligent matching.
#     Returns a dictionary mapping cell index to list of words.
#     """
#     cell_contents = {i: [] for i in range(len(cells))}
#     used_words = set()
    
#     if debug:
#         print(f"\n[DEBUG] Assigning {len(words)} words to {len(cells)} cells")
    
#     for word_idx, (word_bbox, word_text, line_num) in enumerate(words):
#         if word_idx in used_words:
#             continue
            
#         best_cell = None
#         best_score = 0
        
#         for cell_idx, cell in enumerate(cells):
#             cell_bbox = cell['bbox']
            
#             # Calculate overlap ratio
#             overlap_ratio = bbox_overlap_ratio(word_bbox, cell_bbox)
            
#             # Calculate center distance
#             word_center = bbox_center(word_bbox)
#             cell_center = bbox_center(cell_bbox)
#             distance = ((word_center[0] - cell_center[0])**2 + 
#                        (word_center[1] - cell_center[1])**2)**0.5
            
#             # Normalize distance by cell size
#             cell_diagonal = ((cell_bbox[2] - cell_bbox[0])**2 + 
#                             (cell_bbox[3] - cell_bbox[1])**2)**0.5
#             normalized_distance = distance / cell_diagonal if cell_diagonal > 0 else float('inf')
            
#             # Scoring
#             score = 0
            
#             # High weight for overlap
#             if overlap_ratio > overlap_threshold:
#                 score += overlap_ratio * 100
                
#             # Bonus for being close to center
#             if normalized_distance < 0.8:
#                 score += max(0, 50 - normalized_distance * 50)
                
#             # Check if word is mostly contained within cell
#             if (word_bbox[0] >= cell_bbox[0] - 10 and 
#                 word_bbox[1] >= cell_bbox[1] - 10 and
#                 word_bbox[2] <= cell_bbox[2] + 10 and
#                 word_bbox[3] <= cell_bbox[3] + 10):
#                 score += 30
                
#             if score > best_score:
#                 best_score = score
#                 best_cell = cell_idx
        
#         if best_cell is not None and best_score > 10:  # Minimum threshold
#             cell_contents[best_cell].append((word_text, word_bbox, line_num))
#             used_words.add(word_idx)
            
#             if debug:
#                 print(f"  '{word_text}' -> Cell {best_cell} (score: {best_score:.1f})")
#         elif debug:
#             print(f"  '{word_text}' -> UNMATCHED (best score: {best_score:.1f})")
    
#     # Sort words within each cell by reading order
#     for cell_idx in cell_contents:
#         cell_contents[cell_idx].sort(key=lambda item: (item[2], item[1][0]))  # Sort by line, then x
        
#     if debug:
#         print(f"\n[DEBUG] Final cell assignments:")
#         for cell_idx, words_list in cell_contents.items():
#             word_texts = [w[0] for w in words_list]
#             print(f"  Cell {cell_idx}: {word_texts}")
    
#     return {cell_idx: [w[0] for w in words_list] for cell_idx, words_list in cell_contents.items()}

# def reconstruct_table_html(html: str, cell_assignments: Dict[int, List[str]]) -> str:
#     """Reconstruct the HTML table with new content"""
    
#     # Extract all td tags with their positions
#     td_matches = []
#     pattern = r'<td\b[^>]*\bbbox\s*=\s*"([^"]+)"[^>]*>.*?</td>'
    
#     for match in re.finditer(pattern, html, re.IGNORECASE | re.DOTALL):
#         td_matches.append(match)
    
#     # Sort matches by their start position (reverse order for replacement)
#     td_matches.sort(key=lambda m: m.start(), reverse=True)
    
#     # Replace each td element with new content
#     result_html = html
    
#     for i, match in enumerate(td_matches):
#         cell_idx = len(td_matches) - 1 - i  # Reverse index due to reverse sorting
        
#         # Extract bbox and other attributes
#         bbox_match = re.search(r'\bbbox\s*=\s*"([^"]+)"', match.group(0))
#         if not bbox_match:
#             continue
            
#         bbox_str = bbox_match.group(1)
        
#         # Get the new content for this cell
#         new_content = " ".join(cell_assignments.get(cell_idx, []))
        
#         # Create new td tag
#         new_td = f'<td bbox="{bbox_str}" title="bbox {bbox_str}">{new_content}</td>'
        
#         # Replace in the HTML
#         result_html = result_html[:match.start()] + new_td + result_html[match.end():]
    
#     return result_html

# def pretty_print_words(words):
#     """Print word mapping for debugging"""
#     print("\nLoaded words:")
#     for i, (bbox, word, line_num) in enumerate(words):
#         print(f"  #{i:03d}: Line {line_num:2d} - '{word}' at {bbox}")
#     print(f"Total words: {len(words)}")

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--folder", required=True, help="Path to OCR folder")
#     parser.add_argument("--overlap", type=float, default=0.4, help="Overlap threshold (0.0-1.0)")
#     parser.add_argument("--html", help="HTML file to process (optional)")
#     parser.add_argument("--debug", action="store_true", help="Enable debug output")
#     args = parser.parse_args()

#     # Load OCR data
#     words = load_words_for_folder(args.folder)
    
#     if args.debug:
#         pretty_print_words(words)

#     # Load HTML
#     if args.html:
#         with open(args.html, 'r', encoding='utf-8') as f:
#             html = f.read()
#     else:
#         # Use the problematic HTML from your example
#         html = '''<html><table border=\"1\" class=\"ocr_tab\" title=\"\"><tbody><tr><td bbox=\"36 18 322 317\" title=\"bbox 36 18 322 317\">कर पूर्व लाभ (  करोड़</td><td bbox=\"321 18 588 317\" title=\"bbox 321 18 588 317\">कंपनियों की संख्या</td><td bbox=\"590 18 852 317\" title=\"bbox 590 18 852 317\">कर लाभ पूर्वें हिस्सेदारी</td><td bbox=\"852 18 1137 317\" title=\"bbox 852 18 1137 317\">कुल आय में हिस्सेदारी</td><td bbox=\"1160 18 1478 317\" title=\"bbox 1160 18 1478 317\">छुल कॉरपोरेट करदेयता में हिस्सेदारी</td><td bbox=\"1481 18 1766 317\" title=\"bbox 1481 18 1766 317\">कुल आय और कर पूर्व लाभ</td><td bbox=\"1768 18 2021 317\" title=\"bbox 1768 18 2021 317\">प्रभावी कॉरपोरेट दर टैक्स</td></tr><tr><td bbox=\"36 336 322 448\" title=\"bbox 36 336 322 448\">शून्य से कम</td><td bbox=\"321 336 588 448\" title=\"bbox 321 336 588 448\">2 ,54 ,79</td><td bbox=\"590 336 852 448\" title=\"bbox 590 336 852 448\"></td><td bbox=\"852 336 1137 448\" title=\"bbox 852 336 1137 448\">0.58</td><td bbox=\"1160 336 1478 448\" title=\"bbox 1160 336 1478 448\">0.47</td><td bbox=\"1481 336 1766 448\" title=\"bbox 1481 336 1766 448\"></td><td bbox=\"1768 336 2021 448\" title=\"bbox 1768 336 2021 448\"></td></tr><tr><td bbox=\"36 449 322 541\" title=\"bbox 36 449 322 541\">शून्य</td><td bbox=\"321 449 588 541\" title=\"bbox 321 449 588 541\">18 ,80</td><td bbox=\"590 449 852 541\" title=\"bbox 590 449 852 541\"></td><td bbox=\"852 449 1137 541\" title=\"bbox 852 449 1137 541\">6.54</td><td bbox=\"1160 449 1478 541\" title=\"bbox 1160 449 1478 541\">2.81</td><td bbox=\"1481 449 1766 541\" title=\"bbox 1481 449 1766 541\"></td><td bbox=\"1768 449 2021 541\" title=\"bbox 1768 449 2021 541\"></td></tr><tr><td bbox=\"36 544 322 633\" title=\"bbox 36 544 322 633\">0-1</td><td bbox=\"321 544 588 633\" title=\"bbox 321 544 588 633\">2 ,76 531</td><td bbox=\"590 544 852 633\" title=\"bbox 590 544 852 633\">2.73</td><td bbox=\"852 544 1137 633\" title=\"bbox 852 544 1137 633\">3.38</td><td bbox=\"1160 544 1478 633\" title=\"bbox 1160 544 1478 633\">3.25</td><td bbox=\"1481 544 1766 633\" title=\"bbox 1481 544 1766 633\">95.39</td><td bbox=\"1768 544 2021 633\" title=\"bbox 1768 544 2021 633\">29.37</td></tr><tr><td bbox=\"36 648 322 736\" title=\"bbox 36 648 322 736\">01-10</td><td bbox=\"321 648 588 736\" title=\"bbox 321 648 588 736\">२६ 983</td><td bbox=\"590 648 852 736\" title=\"bbox 590 648 852 736\">6.76</td><td bbox=\"852 648 1137 736\" title=\"bbox 852 648 1137 736\">7.54</td><td bbox=\"1160 648 1478 736\" title=\"bbox 1160 648 1478 736\">7.4</td><td bbox=\"1481 648 1766 736\" title=\"bbox 1481 648 1766 736\">85.44</td><td bbox=\"1768 648 2021 736\" title=\"bbox 1768 648 2021 736\">26.99</td></tr><tr><td bbox=\"36 737 322 828\" title=\"bbox 36 737 322 828\">10-50</td><td bbox=\"321 737 588 828\" title=\"bbox 321 737 588 828\">5 ,130</td><td bbox=\"590 737 852 828\" title=\"bbox 590 737 852 828\">9.17</td><td bbox=\"852 737 1137 828\" title=\"bbox 852 737 1137 828\">9.08</td><td bbox=\"1160 737 1478 828\" title=\"bbox 1160 737 1478 828\">9.48</td><td bbox=\"1481 737 1766 828\" title=\"bbox 1481 737 1766 828\">76.26</td><td bbox=\"1768 737 2021 828\" title=\"bbox 1768 737 2021 828\">25.52</td></tr><tr><td bbox=\"36 826 322 918\" title=\"bbox 36 826 322 918\">50-100</td><td bbox=\"321 826 588 918\" title=\"bbox 321 826 588 918\">894</td><td bbox=\"590 826 852 918\" title=\"bbox 590 826 852 918\">5.16</td><td bbox=\"852 826 1137 918\" title=\"bbox 852 826 1137 918\">5.01</td><td bbox=\"1160 826 1478 918\" title=\"bbox 1160 826 1478 918\">5.26</td><td bbox=\"1481 826 1766 918\" title=\"bbox 1481 826 1766 918\">74.83</td><td bbox=\"1768 826 2021 918\" title=\"bbox 1768 826 2021 918\">25.14</td></tr><tr><td bbox=\"36 918 322 1007\" title=\"bbox 36 918 322 1007\">100-500</td><td bbox=\"321 918 588 1007\" title=\"bbox 321 918 588 1007\">895</td><td bbox=\"590 918 852 1007\" title=\"bbox 590 918 852 1007\">15.55</td><td bbox=\"852 918 1137 1007\" title=\"bbox 852 918 1137 1007\">14.56</td><td bbox=\"1160 918 1478 1007\" title=\"bbox 1160 918 1478 1007\">15.12</td><td bbox=\"1481 918 1766 1007\" title=\"bbox 1481 918 1766 1007\">7२</td><td bbox=\"1768 918 2021 1007\" title=\"bbox 1768 918 2021 1007\">23.97</td></tr><tr><td bbox=\"36 1007 322 1165\" title=\"bbox 36 1007 322 1165\">५००  से अधिक</td><td bbox=\"321 1007 588 1165\" title=\"bbox 321 1007 588 1165\">297</td><td bbox=\"590 1007 852 1165\" title=\"bbox 590 1007 852 1165\">60.63</td><td bbox=\"852 1007 1137 1165\" title=\"bbox 852 1007 1137 1165\">53.31</td><td bbox=\"1160 1007 1478 1165\" title=\"bbox 1160 1007 1478 1165\">56.21</td><td bbox=\"1481 1007 1766 1165\" title=\"bbox 1481 1007 1766 1165\">67.66</td><td bbox=\"1768 1007 2021 1165\" title=\"bbox 1768 1007 2021 1165\">22.88</td></tr><tr><td bbox=\"36 1168 322 1249\" title=\"bbox 36 1168 322 1249\">सभी</td><td bbox=\"321 1168 588 1249\" title=\"bbox 321 1168 588 1249\">582889</td><td bbox=\"590 1168 852 1249\" title=\"bbox 590 1168 852 1249\">100</td><td bbox=\"852 1168 1137 1249\" title=\"bbox 852 1168 1137 1249\">100</td><td bbox=\"1160 1168 1478 1249\" title=\"bbox 1160 1168 1478 1249\">100</td><td bbox=\"1481 1168 1766 1249\" title=\"bbox 1481 1168 1766 1249\">76.94</td><td bbox=\"1768 1168 2021 1249\" title=\"bbox 1768 1168 2021 1249\">24.67</td></tr></tbody></table></html>'''

#     print("Processing table reconstruction...")
    
#     # Step 1: Extract table structure (ignoring existing content)
#     cells = extract_table_structure(html)
#     print(f"Found {len(cells)} table cells")
    
#     # Step 2: Assign words to cells
#     cell_assignments = assign_words_to_cells(words, cells, 
#                                            overlap_threshold=args.overlap,
#                                            debug=args.debug)
    
#     # Step 3: Reconstruct HTML
#     final_html = reconstruct_table_html(html, cell_assignments)
    
#     print("\n" + "="*80)
#     print("FINAL RECONSTRUCTED TABLE:")
#     print("="*80)
#     print(final_html)
    
#     # Save to file
#     with open("reconstructed_table.html", "w", encoding="utf-8") as f:
#         f.write(final_html)
#     print(f"\nSaved result to reconstructed_table.html")

# if __name__ == "__main__":
#     main()

#-------------------------------------------------------------------------------------x-----------------------------------------------

# import os
# import re
# import json
# import argparse
# from typing import List, Tuple, Dict, Set, Optional
# from collections import defaultdict
# import html

# # ----------------------------
# # Geometry helpers
# # ----------------------------
# def bbox_intersection_area(b1: Tuple[int, int, int, int],
#                            b2: Tuple[int, int, int, int]) -> int:
#     x1 = max(b1[0], b2[0])
#     y1 = max(b1[1], b2[1])
#     x2 = min(b1[2], b2[2])
#     y2 = min(b1[3], b2[3])
#     if x2 <= x1 or y2 <= y1:
#         return 0
#     return (x2 - x1) * (y2 - y1)

# def bbox_area(b: Tuple[int, int, int, int]) -> int:
#     return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

# def bbox_overlap_ratio(word_bbox, cell_bbox):
#     """Calculate what percentage of the word overlaps with the cell"""
#     intersection = bbox_intersection_area(word_bbox, cell_bbox)
#     word_area = bbox_area(word_bbox)
#     return intersection / word_area if word_area > 0 else 0.0

# def bbox_iou(bbox1, bbox2):
#     """Calculate Intersection over Union"""
#     intersection = bbox_intersection_area(bbox1, bbox2)
#     area1 = bbox_area(bbox1)
#     area2 = bbox_area(bbox2)
#     union = area1 + area2 - intersection
#     return intersection / union if union > 0 else 0.0

# def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
#     """Get center point of bbox"""
#     return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

# def point_in_bbox_with_margin(point, bbox, margin=10):
#     """Check if point is within bbox with margin"""
#     x, y = point
#     return (bbox[0] - margin <= x <= bbox[2] + margin and 
#             bbox[1] - margin <= y <= bbox[3] + margin)

# # ----------------------------
# # Parsing helpers
# # ----------------------------
# _INT_RE = re.compile(r"-?\d+")

# def parse_bbox_from_string(s: str) -> Tuple[int, int, int, int]:
#     """Parse bbox string in x1,y1,x2,y2 format"""
#     nums = _INT_RE.findall(s)
#     if len(nums) < 4:
#         raise ValueError(f"Could not parse 4 integers from bbox string: {s!r}")
#     return tuple(map(int, nums[:4]))

# def parse_layout_line(line: str) -> Tuple[Tuple[int, int, int, int], int]:
#     """Parse layout.txt line in format: x y w h line_number"""
#     nums = _INT_RE.findall(line.strip())
#     if len(nums) < 5:
#         raise ValueError(f"Could not parse 5 integers from layout line: {line!r}")
    
#     x, y, w, h, line_num = map(int, nums[:5])
#     x1, y1 = x, y
#     x2, y2 = x + w, y + h
    
#     return ((x1, y1, x2, y2), line_num)

# def load_words_for_folder(folder_path: str) -> List[Tuple[Tuple[int,int,int,int], str, int]]:
#     """Load words with their bboxes and line numbers"""
#     layout_path = os.path.join(folder_path, "layout.txt")
#     ocr_path = os.path.join(folder_path, "ocr.txt")

#     if not os.path.isfile(layout_path) or not os.path.isfile(ocr_path):
#         raise FileNotFoundError(f"Missing layout.txt or ocr.txt in {folder_path}")

#     with open(layout_path, "r", encoding="utf-8") as f:
#         layout_lines = [ln.strip() for ln in f if ln.strip()]
#     with open(ocr_path, "r", encoding="utf-8") as f:
#         ocr_lines = [ln.rstrip("\n") for ln in f]

#     if len(layout_lines) != len(ocr_lines):
#         min_len = min(len(layout_lines), len(ocr_lines))
#         print(f"[WARN] {folder_path}: layout({len(layout_lines)}) != ocr({len(ocr_lines)}). Truncating to {min_len}.")
#         layout_lines = layout_lines[:min_len]
#         ocr_lines = ocr_lines[:min_len]

#     words = []
#     for i, (layout_line, word) in enumerate(zip(layout_lines, ocr_lines)):
#         try:
#             bbox, line_num = parse_layout_line(layout_line)
#             if word.strip():
#                 words.append((bbox, word.strip(), line_num))
#         except Exception as e:
#             print(f"[WARN] Skipping invalid layout line #{i} in {folder_path}: {layout_line!r} ({e})")
#             continue
    
#     return words

# # ----------------------------
# # Improved table processing
# # ----------------------------
# class TableCell:
#     def __init__(self, bbox, bbox_str, row_idx, col_idx, html_match):
#         self.bbox = bbox
#         self.bbox_str = bbox_str
#         self.row_idx = row_idx
#         self.col_idx = col_idx
#         self.html_match = html_match
#         self.words = []
#         self.confidence = 0.0
    
#     def add_word(self, word_text, word_bbox, line_num, confidence):
#         self.words.append({
#             'text': word_text,
#             'bbox': word_bbox,
#             'line_num': line_num,
#             'confidence': confidence
#         })
#         self.confidence = max(self.confidence, confidence)
    
#     def get_content(self):
#         """Get sorted content for this cell"""
#         if not self.words:
#             return ""
        
#         # Sort by line number first, then by x-coordinate
#         sorted_words = sorted(self.words, key=lambda w: (w['line_num'], w['bbox'][0]))
#         return " ".join(w['text'] for w in sorted_words)

# def extract_table_structure_improved(html_content: str) -> List[TableCell]:
#     """Extract table structure with better cell tracking"""
#     cells = []
    
#     # Find all tr elements first to determine row structure
#     tr_pattern = r'<tr[^>]*>(.*?)</tr>'
#     tr_matches = list(re.finditer(tr_pattern, html_content, re.IGNORECASE | re.DOTALL))
    
#     cell_counter = 0
#     for row_idx, tr_match in enumerate(tr_matches):
#         tr_content = tr_match.group(1)
        
#         # Find all td elements in this row
#         td_pattern = r'<td\b[^>]*\bbbox\s*=\s*"([^"]+)"[^>]*>(.*?)</td>'
#         td_matches = list(re.finditer(td_pattern, tr_content, re.IGNORECASE | re.DOTALL))
        
#         for col_idx, td_match in enumerate(td_matches):
#             bbox_str = td_match.group(1)
#             try:
#                 bbox = parse_bbox_from_string(bbox_str)
                
#                 # Create full match object for later replacement
#                 full_start = tr_match.start() + td_match.start()
#                 full_end = tr_match.start() + td_match.end()
                
#                 cell = TableCell(
#                     bbox=bbox,
#                     bbox_str=bbox_str,
#                     row_idx=row_idx,
#                     col_idx=col_idx,
#                     html_match=(full_start, full_end, td_match.group(0))
#                 )
#                 cells.append(cell)
#                 cell_counter += 1
                
#             except Exception as e:
#                 print(f"[WARN] Could not parse bbox {bbox_str}: {e}")
#                 continue
    
#     print(f"[INFO] Extracted {len(cells)} cells from {len(tr_matches)} rows")
#     return cells

# def calculate_word_cell_score(word_bbox, word_text, cell_bbox, cell) -> float:
#     """Calculate comprehensive score for word-cell assignment"""
#     score = 0.0
    
#     # 1. Overlap ratio (most important)
#     overlap_ratio = bbox_overlap_ratio(word_bbox, cell_bbox)
#     score += overlap_ratio * 100
    
#     # 2. IoU score
#     iou = bbox_iou(word_bbox, cell_bbox)
#     score += iou * 50
    
#     # 3. Center distance penalty
#     word_center = bbox_center(word_bbox)
#     cell_center = bbox_center(cell_bbox)
    
#     cell_width = cell_bbox[2] - cell_bbox[0]
#     cell_height = cell_bbox[3] - cell_bbox[1]
    
#     if cell_width > 0 and cell_height > 0:
#         dx = abs(word_center[0] - cell_center[0]) / cell_width
#         dy = abs(word_center[1] - cell_center[1]) / cell_height
#         distance_penalty = (dx + dy) / 2
#         score += max(0, 30 - distance_penalty * 30)
    
#     # 4. Containment bonus
#     if (word_bbox[0] >= cell_bbox[0] - 5 and
#         word_bbox[1] >= cell_bbox[1] - 5 and
#         word_bbox[2] <= cell_bbox[2] + 5 and
#         word_bbox[3] <= cell_bbox[3] + 5):
#         score += 25
    
#     # 5. Size compatibility
#     word_area = bbox_area(word_bbox)
#     cell_area = bbox_area(cell_bbox)
#     if cell_area > 0:
#         area_ratio = word_area / cell_area
#         if 0.001 <= area_ratio <= 0.8:  # Reasonable word-to-cell size ratio
#             score += 10
    
#     return score

# def assign_words_to_cells_improved(words: List[Tuple], cells: List[TableCell], 
#                                  min_score_threshold: float = 15.0,
#                                  debug: bool = True) -> List[TableCell]:
#     """Improved word-to-cell assignment with better scoring"""
    
#     if debug:
#         print(f"\n[DEBUG] Assigning {len(words)} words to {len(cells)} cells")
#         print(f"[DEBUG] Using minimum score threshold: {min_score_threshold}")
    
#     # Track assignments
#     word_assignments = {}
    
#     for word_idx, (word_bbox, word_text, line_num) in enumerate(words):
#         best_cell = None
#         best_score = 0.0
#         scores = []
        
#         for cell_idx, cell in enumerate(cells):
#             score = calculate_word_cell_score(word_bbox, word_text, cell.bbox, cell)
#             scores.append((cell_idx, score))
            
#             if score > best_score:
#                 best_score = score
#                 best_cell = cell_idx
        
#         # Only assign if score meets threshold
#         if best_cell is not None and best_score >= min_score_threshold:
#             cells[best_cell].add_word(word_text, word_bbox, line_num, best_score)
#             word_assignments[word_idx] = best_cell
            
#             if debug:
#                 print(f"  '{word_text}' -> Cell[{cells[best_cell].row_idx},{cells[best_cell].col_idx}] (score: {best_score:.1f})")
#         else:
#             if debug:
#                 # Show top 3 candidates for debugging
#                 scores.sort(key=lambda x: x[1], reverse=True)
#                 top_scores = scores[:3]
#                 print(f"  '{word_text}' -> UNMATCHED (best: {best_score:.1f}, top3: {top_scores})")
    
#     # Statistics
#     assigned_words = len(word_assignments)
#     unassigned_words = len(words) - assigned_words
#     empty_cells = sum(1 for cell in cells if not cell.words)
    
#     print(f"\n[INFO] Assignment Summary:")
#     print(f"  - Assigned words: {assigned_words}/{len(words)} ({assigned_words/len(words)*100:.1f}%)")
#     print(f"  - Unassigned words: {unassigned_words}")
#     print(f"  - Empty cells: {empty_cells}/{len(cells)}")
    
#     return cells

# def reconstruct_table_html_improved(html_content: str, cells: List[TableCell]) -> str:
#     """Improved HTML reconstruction with proper cell tracking"""
    
#     # Sort cells by their position in HTML (reverse for safe replacement)
#     cells_with_pos = [(cell.html_match[0], cell.html_match[1], cell) for cell in cells]
#     cells_with_pos.sort(key=lambda x: x[0], reverse=True)
    
#     result_html = html_content
    
#     for start_pos, end_pos, cell in cells_with_pos:
#         # Get new content
#         new_content = cell.get_content()
        
#         # Escape HTML entities in content
#         new_content = html.escape(new_content) if new_content else ""
        
#         # Create new td tag with same attributes but new content
#         new_td = f'<td bbox="{cell.bbox_str}" title="bbox {cell.bbox_str}">{new_content}</td>'
        
#         # Replace in HTML
#         result_html = result_html[:start_pos] + new_td + result_html[end_pos:]
    
#     return result_html

# # ----------------------------
# # Alternative approaches
# # ----------------------------

# def cluster_words_by_proximity(words: List[Tuple], max_distance: int = 20) -> List[List[Tuple]]:
#     """Cluster nearby words together before assignment"""
#     if not words:
#         return []
    
#     clusters = []
#     used = set()
    
#     for i, (bbox1, text1, line1) in enumerate(words):
#         if i in used:
#             continue
            
#         cluster = [(bbox1, text1, line1)]
#         used.add(i)
#         center1 = bbox_center(bbox1)
        
#         # Find nearby words
#         for j, (bbox2, text2, line2) in enumerate(words[i+1:], i+1):
#             if j in used:
#                 continue
                
#             center2 = bbox_center(bbox2)
#             distance = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
            
#             if distance <= max_distance and abs(line1 - line2) <= 1:  # Same or adjacent lines
#                 cluster.append((bbox2, text2, line2))
#                 used.add(j)
        
#         clusters.append(cluster)
    
#     return clusters

# def reconstruct_with_clustering(html_content: str, words: List[Tuple], 
#                                cluster_distance: int = 15, debug: bool = True):
#     """Alternative approach using word clustering"""
    
#     print(f"\n[INFO] Using clustering approach with max distance: {cluster_distance}")
    
#     # Step 1: Cluster words
#     word_clusters = cluster_words_by_proximity(words, cluster_distance)
#     print(f"[INFO] Created {len(word_clusters)} word clusters from {len(words)} words")
    
#     # Step 2: Extract table structure
#     cells = extract_table_structure_improved(html_content)
    
#     # Step 3: Assign clusters to cells
#     for cluster_idx, cluster in enumerate(word_clusters):
#         if not cluster:
#             continue
            
#         # Use the first word's bbox as cluster representative
#         cluster_bbox = cluster[0][0]
#         cluster_text = " ".join(word[1] for word in cluster)
#         cluster_line = cluster[0][2]
        
#         best_cell = None
#         best_score = 0.0
        
#         for cell in cells:
#             score = calculate_word_cell_score(cluster_bbox, cluster_text, cell.bbox, cell)
#             if score > best_score:
#                 best_score = score
#                 best_cell = cell
        
#         if best_cell and best_score >= 10.0:
#             # Add all words from cluster to the best cell
#             for word_bbox, word_text, word_line in cluster:
#                 best_cell.add_word(word_text, word_bbox, word_line, best_score)
            
#             if debug:
#                 print(f"  Cluster {cluster_idx} ({len(cluster)} words): '{cluster_text[:50]}...' -> Cell[{best_cell.row_idx},{best_cell.col_idx}]")
    
#     # Step 4: Reconstruct HTML
#     return reconstruct_table_html_improved(html_content, cells)

# # ----------------------------
# # Main function with multiple approaches
# # ----------------------------

# def main():
#     parser = argparse.ArgumentParser(description="Improved table reconstruction with multiple approaches")
#     parser.add_argument("--folder", required=True, help="Path to OCR folder")
#     parser.add_argument("--html", help="HTML file to process (optional)")
#     parser.add_argument("--approach", choices=["improved", "clustering", "both"], 
#                        default="both", help="Which approach to use")
#     parser.add_argument("--overlap", type=float, default=15.0, help="Minimum score threshold")
#     parser.add_argument("--cluster-distance", type=int, default=15, help="Max distance for clustering")
#     parser.add_argument("--debug", action="store_true", help="Enable debug output")
#     args = parser.parse_args()

#     # Load OCR data
#     print(f"Loading words from {args.folder}...")
#     words = load_words_for_folder(args.folder)
#     print(f"Loaded {len(words)} words")

#     # Load HTML
#     if args.html:
#         with open(args.html, 'r', encoding='utf-8') as f:
#             html_content = f.read()
#         print(f"Loaded HTML from {args.html}")
#     else:
#         # Use the provided HTML
#         html_content = '''<html><table border=\"1\" class=\"ocr_tab\" title=\"\"><tbody><tr><td bbox=\"1535 22 2002 141\" title=\"bbox 1535 22 2002 141\">87.5</td></tr><tr><td bbox=\"1535 138 2002 264\" title=\"bbox 1535 138 2002 264\">64.</td></tr><tr><td bbox=\"1535 260 2002 390\" title=\"bbox 1535 260 2002 390\">78.2</td></tr><tr><td bbox=\"1535 477 2002 610\" title=\"bbox 1535 477 2002 610\">वाले 31.4</td></tr><tr><td bbox=\"1535 617 2002 758\" title=\"bbox 1535 617 2002 758\">76.1</td></tr><tr><td bbox=\"1535 758 2002 866\" title=\"bbox 1535 758 2002 866\">73.0</td></tr></tbody></table></html>'''
#         print("Using provided HTML sample")

#     if args.approach in ["improved", "both"]:
#         print("\n" + "="*80)
#         print("APPROACH 1: IMPROVED SCORING")
#         print("="*80)
        
#         cells = extract_table_structure_improved(html_content)
#         cells_with_words = assign_words_to_cells_improved(
#             words, cells, min_score_threshold=args.overlap, debug=args.debug)
        
#         final_html_improved = reconstruct_table_html_improved(html_content, cells_with_words)
        
#         with open("reconstructed_table_improved.html", "w", encoding="utf-8") as f:
#             f.write(final_html_improved)
#         print(f"\nSaved improved result to reconstructed_table_improved.html")

#     if args.approach in ["clustering", "both"]:
#         print("\n" + "="*80)
#         print("APPROACH 2: CLUSTERING")
#         print("="*80)
        
#         final_html_clustering = reconstruct_with_clustering(
#             html_content, words, cluster_distance=args.cluster_distance, debug=args.debug)
        
#         with open("reconstructed_table_clustering.html", "w", encoding="utf-8") as f:
#             f.write(final_html_clustering)
#         print(f"\nSaved clustering result to reconstructed_table_clustering.html")

# if __name__ == "__main__":
#     main()


#-------------------------------------------------------------------------------------x-----------------------------------------------
import bs4
from bs4 import BeautifulSoup

# Load your table HTML
with open("table.html", "r", encoding="utf-8") as f:
    soup = BeautifulSoup(f.read(), "html.parser")

# Load OCR text (list of words)
with open("ocr.txt", "r", encoding="utf-8") as f:
    ocr_words = f.read().splitlines()

# Load OCR layout (x, y, w, h, id)
layout = []
with open("layout.txt", "r", encoding="utf-8") as f:
    for line in f:
        x, y, w, h, idx = line.strip().split(",")
        layout.append((int(x), int(y), int(w), int(h), int(idx)))

# Map id -> text
id2text = {i+1: word for i, word in enumerate(ocr_words)}

# Function to check overlap
def overlaps(cell_bbox, word_bbox):
    x1, y1, x2, y2 = cell_bbox
    wx, wy, ww, wh, _ = word_bbox
    wx1, wy1, wx2, wy2 = wx, wy, wx+ww, wy+wh
    return not (x2 < wx1 or wx2 < x1 or y2 < wy1 or wy2 < y1)

# Fill table cells
for td in soup.find_all("td"):
    bbox_str = td["title"].replace("bbox", "").strip()
    x1, y1, x2, y2 = map(int, bbox_str.split())
    texts = []
    for word_bbox in layout:
        if overlaps((x1, y1, x2, y2), word_bbox):
            idx = word_bbox[4]
            texts.append(id2text.get(idx, ""))
    td.string = " ".join(texts)

# Save merged HTML
with open("merged_table.html", "w", encoding="utf-8") as f:
    f.write(str(soup.prettify()))
