# import bs4
# from bs4 import BeautifulSoup

# # Load your table HTML
# with open("output.html", "r", encoding="utf-8") as f:
#     soup = BeautifulSoup(f.read(), "html.parser")

# # Load OCR text (list of words)
# with open("ocr.txt", "r", encoding="utf-8") as f:
#     ocr_words = f.read().splitlines()

# # Load OCR layout (x, y, w, h, id)
# layout = []
# with open("layout.txt", "r", encoding="utf-8") as f:
#     for line in f:
#         x, y, w, h, idx = line.strip().split(",")
#         layout.append((int(x), int(y), int(w), int(h), int(idx)))

# # Map id -> text
# id2text = {i + 1: word for i, word in enumerate(ocr_words)}

# # Function to check overlap using intersection-over-area
# def overlaps(cell_bbox, word_bbox, threshold=0.3):
#     x1, y1, x2, y2 = cell_bbox
#     wx, wy, ww, wh, _ = word_bbox
#     wx1, wy1, wx2, wy2 = wx, wy, wx + ww, wy + wh

#     # Intersection
#     ix1, iy1 = max(x1, wx1), max(y1, wy1)
#     ix2, iy2 = min(x2, wx2), min(y2, wy2)
#     inter_w, inter_h = max(0, ix2 - ix1), max(0, iy2 - iy1)
#     inter_area = inter_w * inter_h

#     if inter_area == 0:
#         return False

#     word_area = ww * wh
#     # Require at least `threshold` overlap
#     return inter_area / word_area >= threshold

# # Fill table cells
# for td in soup.find_all("td"):
#     if "title" not in td.attrs:
#         continue
#     bbox_str = td["title"].replace("bbox", "").strip()
#     x1, y1, x2, y2 = map(int, bbox_str.split())
#     texts = []

#     for word_bbox in layout:
#         if overlaps((x1, y1, x2, y2), word_bbox):
#             idx = word_bbox[4]
#             texts.append((word_bbox[1], word_bbox[0], id2text.get(idx, "")))

#     # Sort by (y, x) so text reads top-to-bottom, left-to-right
#     texts.sort()
#     td.string = " ".join(t[-1] for t in texts if t[-1])

# # Save merged HTML
# with open("merged_table.html", "w", encoding="utf-8") as f:
#     f.write(str(soup.prettify()))

# print("✅ Merged table saved to merged_table.html")



#!/usr/bin/env python3
"""
Table Reconstructor
Reconstructs HTML tables by matching layout coordinates with OCR text
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int
    
    @property
    def x2(self) -> int:
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        return self.y + self.height
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def overlaps_with(self, other: 'BoundingBox') -> float:
        """Calculate overlap ratio with another bounding box"""
        x_overlap = max(0, min(self.x2, other.x2) - max(self.x, other.x))
        y_overlap = max(0, min(self.y2, other.y2) - max(self.y, other.y))
        
        if x_overlap == 0 or y_overlap == 0:
            return 0.0
        
        overlap_area = x_overlap * y_overlap
        return overlap_area / self.area


@dataclass
class LayoutItem:
    bbox: BoundingBox
    id: int
    text: str = ""


@dataclass
class TableCell:
    bbox: BoundingBox
    element: object
    row: int
    col: int
    
    def __hash__(self):
        return hash((self.row, self.col))
    
    def __eq__(self, other):
        if not isinstance(other, TableCell):
            return False
        return self.row == other.row and self.col == other.col


class TableReconstructor:
    def __init__(self):
        self.layout_items: List[LayoutItem] = []
        self.table_cells: List[TableCell] = []
        self.overlap_threshold = 0.3
    
    def load_layout_data(self, layout_file: str) -> None:
        """Load layout data from CSV-like file"""
        self.layout_items = []
        
        with open(layout_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                if len(parts) < 5:
                    print(f"Warning: Skipping malformed line {line_num}: {line}")
                    continue
                
                try:
                    x, y, width, height, item_id = map(int, parts[:5])
                    bbox = BoundingBox(x, y, width, height)
                    layout_item = LayoutItem(bbox=bbox, id=item_id)
                    self.layout_items.append(layout_item)
                except ValueError as e:
                    print(f"Warning: Error parsing line {line_num}: {e}")
        
        print(f"Loaded {len(self.layout_items)} layout items")
    
    def load_ocr_data(self, ocr_file: str) -> None:
        """Load OCR text data"""
        with open(ocr_file, 'r', encoding='utf-8') as f:
            ocr_texts = [line.strip() for line in f if line.strip()]
        
        # Assign OCR text to layout items based on order
        for i, text in enumerate(ocr_texts):
            if i < len(self.layout_items):
                self.layout_items[i].text = text
        
        print(f"Loaded {len(ocr_texts)} OCR text entries")
        
        if len(ocr_texts) != len(self.layout_items):
            print(f"Warning: Mismatch - {len(self.layout_items)} layout items but {len(ocr_texts)} OCR texts")
    
    def load_html_table(self, html_file: str) -> BeautifulSoup:
        """Load and parse HTML table"""
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        table = soup.find('table')
        
        if not table:
            raise ValueError("No table found in HTML file")
        
        self.table_cells = []
        
        # Extract table cells with their bounding boxes
        for row_idx, tr in enumerate(table.find_all('tr')):
            for col_idx, td in enumerate(tr.find_all('td')):
                title = td.get('title', '')
                
                # Extract bbox coordinates from title attribute
                bbox_match = re.search(r'bbox\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)', title)
                if bbox_match:
                    x1, y1, x2, y2 = map(int, bbox_match.groups())
                    bbox = BoundingBox(x1, y1, x2 - x1, y2 - y1)
                    
                    cell = TableCell(
                        bbox=bbox,
                        element=td,
                        row=row_idx,
                        col=col_idx
                    )
                    self.table_cells.append(cell)
        
        print(f"Found {len(self.table_cells)} table cells with bounding boxes")
        return soup
    
    def find_best_match(self, layout_item: LayoutItem) -> Optional[TableCell]:
        """Find the best matching table cell for a layout item"""
        best_match = None
        best_overlap = 0
        
        for cell in self.table_cells:
            overlap = layout_item.bbox.overlaps_with(cell.bbox)
            if overlap > best_overlap and overlap >= self.overlap_threshold:
                best_overlap = overlap
                best_match = cell
        
        return best_match
    
    def reconstruct_table(self) -> Tuple[BeautifulSoup, Dict[str, int]]:
        """Reconstruct the table by matching layout items to table cells"""
        if not hasattr(self, 'soup'):
            raise ValueError("HTML table not loaded")
        
        stats = {
            'total_items': len(self.layout_items),
            'matched_items': 0,
            'unmatched_items': 0
        }
        
        matched_cells = set()
        
        for layout_item in self.layout_items:
            best_cell = self.find_best_match(layout_item)
            
            if best_cell and best_cell not in matched_cells:
                # Insert text into the cell
                best_cell.element.string = layout_item.text
                
                # Add debugging info as data attribute
                best_cell.element['data-layout-id'] = str(layout_item.id)
                best_cell.element['data-confidence'] = f"{layout_item.bbox.overlaps_with(best_cell.bbox):.2f}"
                
                matched_cells.add(best_cell)
                stats['matched_items'] += 1
                
                print(f"Matched layout ID {layout_item.id} ('{layout_item.text}') to cell at row {best_cell.row}, col {best_cell.col}")
            else:
                stats['unmatched_items'] += 1
                print(f"Warning: No match found for layout ID {layout_item.id} ('{layout_item.text}')")
        
        return self.soup, stats
    
    def save_result(self, output_file: str, soup: BeautifulSoup) -> None:
        """Save the reconstructed table to file"""
        # Add some basic styling for better visualization
        html_template = """<!DOCTYPE html>
<html lang="hi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Reconstructed Table</title>
    <style>
        body {{
            font-family: 'Noto Sans Devanagari', Arial, sans-serif;
            margin: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: center;
            vertical-align: middle;
            font-size: 14px;
            min-height: 40px;
        }}
        td:empty {{
            background: #f9f9f9;
        }}
        .debug-info {{
            font-size: 10px;
            color: #666;
            margin-top: 5px;
        }}
        .stats {{
            background: #e8f4fd;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari&display=swap" rel="stylesheet">
</head>
<body>
    <div class="container">
        <h1>Reconstructed Table</h1>
        {content}
    </div>
</body>
</html>"""
        
        table_html = str(soup.find('table'))
        full_html = html_template.format(content=table_html)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
        
        print(f"Reconstructed table saved to: {output_file}")
    
    def process_files(self, layout_file: str, ocr_file: str, html_file: str, output_file: str) -> None:
        """Process all files and reconstruct the table"""
        print("Starting table reconstruction...")
        
        # Load all data
        self.load_layout_data(layout_file)
        self.load_ocr_data(ocr_file)
        self.soup = self.load_html_table(html_file)
        
        # Reconstruct table
        reconstructed_soup, stats = self.reconstruct_table()
        
        # Save result
        self.save_result(output_file, reconstructed_soup)
        
        # Print statistics
        print("\n" + "="*50)
        print("RECONSTRUCTION STATISTICS")
        print("="*50)
        print(f"Total layout items: {stats['total_items']}")
        print(f"Successfully matched: {stats['matched_items']}")
        print(f"Unmatched items: {stats['unmatched_items']}")
        print(f"Success rate: {(stats['matched_items']/stats['total_items']*100):.1f}%")
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='Reconstruct HTML table from layout and OCR data')
    parser.add_argument('--layout', required=True, help='Layout data file (layout.txt)')
    parser.add_argument('--ocr', required=True, help='OCR text data file (ocr.txt)')
    parser.add_argument('--html', required=True, help='HTML table file (output.html)')
    parser.add_argument('--output', default='reconstructed_table.html', help='Output file name')
    parser.add_argument('--threshold', type=float, default=0.3, help='Overlap threshold (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Validate input files
    for file_path in [args.layout, args.ocr, args.html]:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            return 1
    
    try:
        reconstructor = TableReconstructor()
        reconstructor.overlap_threshold = args.threshold
        reconstructor.process_files(args.layout, args.ocr, args.html, args.output)
        
        print(f"\n✅ Table reconstruction completed successfully!")
        print(f"📁 Output saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error during reconstruction: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())