# import os
# import cv2
# from PIL import Image
# import matplotlib.pyplot as plt
# from .td import TableDetector
# from .tsr import get_rows_from_yolo, get_cols_from_tatr, get_cells_from_rows_cols, get_rows_from_tatr
# from .utils import *
# from .sprint import get_logical_structure, align_otsl_from_rows_cols, convert_to_html
# from bs4 import BeautifulSoup
# import pathlib
# import torch
# import easyocr
# import numpy as np
# import re

# CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

# def get_cell_ocr(img, bbox, lang,reader):
#     try:
#         cell_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]

#         if cell_img.size == 0:
#             return ""

#         # Preprocess for better OCR
#         if len(cell_img.shape) == 3:
#             gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
#         else:
#             gray = cell_img

#         # Apply adaptive thresholding
#         thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#         cell_pil_img = Image.fromarray(thresh)
#         cell_array = np.array(cell_pil_img)
#         result = reader.readtext(cell_array, detail=0, paragraph=False)

#         if not result:
#             return ""

#         ocr_result = "\n".join(result).strip()
#         return ocr_result

#     except Exception as e:
#         print(f"OCR error for bbox {bbox}: {e}")
#         return ""


# def perform_td(image_path):
#     image = cv2.imread(image_path)
#     table_det = TableDetector()
#     return table_det.predict(image=image)


# def perform_tsr(img_file, x1, y1, struct_only, lang, reader):
#     print(f"Processing table image: {img_file}")

#     rows = get_rows_from_tatr(img_file)
#     cols = get_cols_from_tatr(img_file)
#     print('Physical TSR')
#     print(str(len(rows)) + ' rows detected')
#     print(str(len(cols)) + ' cols detected')

#     # Debug: Save visualization
#     if len(rows) > 0 or len(cols) > 0:
#         debug_img = cv2.imread(img_file)
#         for row in rows:
#             x1_r, y1_r, x2_r, y2_r = map(int, row)
#             cv2.rectangle(debug_img, (x1_r, y1_r), (x2_r, y2_r), (0, 0, 255), 2)
#         for col in cols:
#             x1_c, y1_c, x2_c, y2_c = map(int, col)
#             cv2.rectangle(debug_img, (x1_c, y1_c), (x2_c, y2_c), (255, 0, 0), 2)
#         cv2.imwrite("debug_rows_cols.jpg", debug_img)

#     if len(rows) == 0 or len(cols) == 0:
#         print("WARNING: No rows or columns detected!")

#     rows, cols = order_rows_cols(rows, cols)

#     # Extracting Grid Cells
#     cells = get_cells_from_rows_cols(rows, cols)

#     # Corner case if no cells detected
#     if len(cells) == 0 or len(rows) == 0 or len(cols) == 0:
#         print('No Physical Structure')
#         table_img = cv2.imread(img_file)
#         if table_img is None:
#             html_string = f'<table border=1><tr><td title="bbox {x1} {y1} {x1 + 100} {y1 + 50}">Error: Could not read image</td></tr></table>'
#             soup = BeautifulSoup(html_string, 'html.parser')
#             return soup, []
#         h, w, c = table_img.shape
#         bbox = [0, 0, w, h]
#         text = get_cell_ocr(table_img, bbox, lang, reader)
#         html_string = f'<table border=1><tr><td title="bbox {x1} {y1} {x1 + w} {y1 + h}">' + text + '</td></tr></table>'
#         soup = BeautifulSoup(html_string, 'html.parser')
#         return soup, []

#     print('Logical TSR')
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Use sprint's get_logical_structure function
#     otsl_string = get_logical_structure(img_file, device)

#     # Use sprint's align_otsl_from_rows_cols function
#     corrected_otsl = align_otsl_from_rows_cols(otsl_string, len(rows), len(cols))

#     # Correction - normalize all cell types to 'C'
#     corrected_otsl = corrected_otsl.replace("E", "C")
#     corrected_otsl = corrected_otsl.replace("F", "C")

#     print('OTSL => ' + otsl_string)
#     print("Corrected OTSL => " + corrected_otsl)

#     # Use sprint's convert_to_html function
#     html_string, struc_cells = convert_to_html(corrected_otsl, len(rows), len(cols), cells)

#     # Parse the HTML
#     soup = BeautifulSoup('<html>' + html_string + '</html>', 'html.parser')

#     # Do OCR if struct_only flag is FALSE
#     if not struct_only:
#         cropped_img = cv2.imread(img_file)
#         for bbox in soup.find_all('td'):
#             # Replace the content inside the td with OCR result
#             ocr_bbox = bbox['title'].split(' ')[1:]
#             ocr_bbox = list(map(int, ocr_bbox))
#             bbox.string = get_cell_ocr(cropped_img, ocr_bbox, lang, reader)

#             # Correct coordinates relative to table position
#             ocr_bbox[0] += x1
#             ocr_bbox[1] += y1
#             ocr_bbox[2] += x1
#             ocr_bbox[3] += y1
#             bbox['title'] = f'bbox {ocr_bbox[0]} {ocr_bbox[1]} {ocr_bbox[2]} {ocr_bbox[3]}'

#     return soup, struc_cells


# def get_full_page_hocr(img_file, lang, reader):
#     tabledata = get_table_hocrs(img_file, lang, reader)
#     finalimgtoocr = img_file

#     # Hide all tables from images before performing text recognition
#     if len(tabledata) > 0:
#         img = cv2.imread(img_file)
#         for entry in tabledata:
#             bbox = entry[1]
#             tab_x, tab_y, tab_x2, tab_y2 = bbox
#             img_x, img_y, img_x2, img_y2 = int(tab_x), int(tab_y), int(tab_x2), int(tab_y2)
#             cv2.rectangle(img, (img_x, img_y), (img_x2, img_y2), (255, 0, 255), -1)
#         finalimgfile = img_file[:-4] + '_filtered.jpg'
#         cv2.imwrite(finalimgfile, img)
#         finalimgtoocr = finalimgfile

#     # EasyOCR output with bounding boxes
#     result = reader.readtext(finalimgtoocr, detail=1, paragraph=False)

#     # Basic XHTML HOCR template
#     soup = BeautifulSoup("""
#         <html xmlns="http://www.w3.org/1999/xhtml">
#         <head>
#           <title>EasyOCR HOCR</title>
#           <meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>
#         </head>
#         <body>
#           <div class='ocr_page' id='page_1' title='bbox 0 0 1000 1000'>
#             <div class='ocr_carea' id='block_1'>
#               <p class='ocr_par' id='par_1'>
#               </p>
#             </div>
#           </div>
#         </body>
#         </html>
#         """, "html.parser")

#     # Add EasyOCR spans inside <p>
#     p_tag = soup.find('p', {'id': 'par_1'})

#     for i, (bbox, text, _) in enumerate(result):
#         (x1, y1), (x2, y2), (x3, y3), (x4, y4) = bbox
#         left = int(min(x1, x4))
#         top = int(min(y1, y2))
#         right = int(max(x2, x3))
#         bottom = int(max(y3, y4))

#         span = soup.new_tag('span', attrs={
#             'class': 'ocrx_word',
#             'id': f'word_{i + 1}',
#             'title': f'bbox {left} {top} {right} {bottom}'
#         })
#         span.string = text
#         p_tag.append(span)
#         p_tag.append(' ')  # Add space between words

#     # Adding table HOCR in final HOCR at proper position
#     if len(tabledata) > 0:
#         for entry in tabledata:
#             tab_element = entry[0]
#             tab_bbox = entry[1]
#             tab_position = tab_bbox[1]
#             for elem in soup.find_all('span', class_="ocr_line"):
#                 find_all_ele = elem.attrs["title"].split(" ")
#                 line_position = int(find_all_ele[2])
#                 if tab_position < line_position:
#                     elem.insert_before(tab_element)
#                     break

#     return soup


# def get_table_hocrs(image_file, lang, reader):
#     final_hocrs = []
#     try:
#         image = cv2.imread(image_file)
#         if image is None:
#             print(f"Error: Could not load image {image_file}")
#             return []

#         print(f"Processing image: {image_file} (shape: {image.shape})")

#         dets = perform_td(image_file)
#         print(str(len(dets)) + ' tables detected')

#         # Debug: Visualize detected tables
#         if len(dets) > 0:
#             debug_img = image.copy()
#             for i, det in enumerate(dets):
#                 x1, y1, x2, y2 = map(int, det)
#                 cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
#                 cv2.putText(debug_img, f"Table {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.imwrite("detected_tables.jpg", debug_img)

#         for i, det in enumerate(dets):
#             try:
#                 x1, y1, x2, y2 = map(int, det)
#                 print(f"Processing table {i + 1}: bbox({x1}, {y1}, {x2}, {y2})")

#                 # Validate bounding box
#                 if x2 <= x1 or y2 <= y1:
#                     print(f"Invalid bounding box for table {i + 1}")
#                     continue

#                 tab_box = [x1, y1, x2, y2]
#                 cropped_img = image[y1:y2, x1:x2]

#                 if cropped_img.size == 0:
#                     print(f"Empty crop for table {i + 1}")
#                     continue

#                 plt.imsave("temp.jpg", cropped_img)
#                 img_path = "temp.jpg"
#                 hocr_string, struct_cells = perform_tsr(img_path, x1, y1, False, lang, reader)
#                 final_hocrs.append([hocr_string, tab_box])

#             except Exception as e:
#                 print(f"Error processing table {i + 1}: {e}")
#                 continue

#     except Exception as e:
#         print(f"Error in get_table_hocrs: {e}")
#         return []

#     return final_hocrs


# if __name__ == "__main__":
#     print()
#     # Try TSR Call
#     img_path = r"C:\Users\91736\Downloads\hin-48.png"
#     hocr_string = perform_tsr(img_path, 0, 0, True, lang, reader)
#     print(hocr_string)

#--------------------------------------------------------------X----------------------------------------------------------------

# import os
# import cv2
# import matplotlib.pyplot as plt
# from PIL import Image
# from bs4 import BeautifulSoup
# import pathlib
# import torch

# from .td import TableDetector
# from .tsr import get_rows_from_tatr, get_cols_from_tatr, get_cells_from_rows_cols
# from .utils import *
# from .sprint import get_logical_structure, align_otsl_from_rows_cols, convert_to_html

# CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

# def perform_td(image_path):
#     """Run table detector (YOLO) and return bounding boxes [x1,y1,x2,y2]."""
#     image = cv2.imread(image_path)
#     table_det = TableDetector()
#     return table_det.predict(image=image)


# def perform_tsr(img_file, x1, y1):
#     """
#     Perform Table Structure Recognition (TSR).
#     Returns: (soup, struc_cells)
#         soup -> BeautifulSoup object with <table> HTML (without text inside cells)
#         struc_cells -> list of bounding boxes for each cell
#     """
#     print(f"Processing table image: {img_file}")

#     rows = get_rows_from_tatr(img_file)
#     cols = get_cols_from_tatr(img_file)
#     print('Physical TSR')
#     print(f"{len(rows)} rows detected")
#     print(f"{len(cols)} cols detected")

#     # Debug visualization
#     if len(rows) > 0 or len(cols) > 0:
#         debug_img = cv2.imread(img_file)
#         for row in rows:
#             x1_r, y1_r, x2_r, y2_r = map(int, row)
#             cv2.rectangle(debug_img, (x1_r, y1_r), (x2_r, y2_r), (0, 0, 255), 2)
#         for col in cols:
#             x1_c, y1_c, x2_c, y2_c = map(int, col)
#             cv2.rectangle(debug_img, (x1_c, y1_c), (x2_c, y2_c), (255, 0, 0), 2)
#         cv2.imwrite("debug_rows_cols.jpg", debug_img)

#     if len(rows) == 0 or len(cols) == 0:
#         print("WARNING: No rows or columns detected!")
#         table_img = cv2.imread(img_file)
#         if table_img is None:
#             html_string = f'<table border=1><tr><td title="bbox {x1} {y1} {x1+100} {y1+50}">Error: Could not read image</td></tr></table>'
#             soup = BeautifulSoup(html_string, 'html.parser')
#             return soup, []
#         h, w, _ = table_img.shape
#         bbox = [0, 0, w, h]
#         html_string = f'<table border=1><tr><td title="bbox {x1} {y1} {x1+w} {y1+h}"></td></tr></table>'
#         soup = BeautifulSoup(html_string, 'html.parser')
#         return soup, []

#     print('Logical TSR')
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#     # Get logical table structure (OTSL sequence)
#     otsl_string = get_logical_structure(img_file, device)

#     # Align with detected rows/cols
#     corrected_otsl = align_otsl_from_rows_cols(otsl_string, len(rows), len(cols))

#     # Normalize cell types to 'C'
#     corrected_otsl = corrected_otsl.replace("E", "C").replace("F", "C")

#     print('OTSL => ' + otsl_string)
#     print('Corrected OTSL => ' + corrected_otsl)

#     # Convert structure to HTML table
#     cells = get_cells_from_rows_cols(rows, cols)
#     html_string, struc_cells = convert_to_html(corrected_otsl, len(rows), len(cols), cells)

#     # Wrap in HTML
#     soup = BeautifulSoup('<html>' + html_string + '</html>', 'html.parser')

#     # IMPORTANT: Cells are empty here (no OCR text inserted)
#     return soup, struc_cells


# def get_table_structures(image_file):
#     """
#     Detect tables in a page and extract only table structures (no OCR).
#     Returns: list of (html_soup, table_bbox)
#     """
#     final_structs = []
#     try:
#         image = cv2.imread(image_file)
#         if image is None:
#             print(f"Error: Could not load image {image_file}")
#             return []

#         print(f"Processing image: {image_file} (shape: {image.shape})")

#         dets = perform_td(image_file)
#         print(f"{len(dets)} tables detected")

#         # Debug visualization
#         if len(dets) > 0:
#             debug_img = image.copy()
#             for i, det in enumerate(dets):
#                 x1, y1, x2, y2 = map(int, det)
#                 cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
#                 cv2.putText(debug_img, f"Table {i}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.imwrite("detected_tables.jpg", debug_img)

#         for i, det in enumerate(dets):
#             try:
#                 x1, y1, x2, y2 = map(int, det)
#                 print(f"Processing table {i+1}: bbox({x1}, {y1}, {x2}, {y2})")

#                 if x2 <= x1 or y2 <= y1:
#                     print(f"Invalid bounding box for table {i+1}")
#                     continue

#                 cropped_img = image[y1:y2, x1:x2]
#                 if cropped_img.size == 0:
#                     print(f"Empty crop for table {i+1}")
#                     continue

#                 tmp_path = "temp.jpg"
#                 plt.imsave(tmp_path, cropped_img)

#                 soup, struct_cells = perform_tsr(tmp_path, x1, y1)
#                 final_structs.append([soup, [x1, y1, x2, y2]])

#             except Exception as e:
#                 print(f"Error processing table {i+1}: {e}")
#                 continue

#     except Exception as e:
#         print(f"Error in get_table_structures: {e}")
#         return []

#     return final_structs


# if __name__ == "__main__":
#     img_path = r"C:\Users\91736\Downloads\hin-48.png"
#     tables = get_table_structures(img_path)
#     for soup, bbox in tables:
#         print("Detected table at:", bbox)
#         print(soup.prettify())


#--------------------------------------------------------------X----------------------------------------------------------------




import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from bs4 import BeautifulSoup
import pathlib
import torch

from .td import TableDetector
from .tsr import get_rows_from_tatr, get_cols_from_tatr, get_cells_from_rows_cols
from .utils import *
from .sprint import get_logical_structure, align_otsl_from_rows_cols, convert_to_html
from .coordinate_normalizer import CoordinateNormalizer

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

def perform_td(image_path):
    """Run table detector (YOLO) and return bounding boxes [x1,y1,x2,y2]."""
    image = cv2.imread(image_path)
    table_det = TableDetector()
    return table_det.predict(image=image)


def perform_tsr(img_file, x1, y1):
    """
    Perform Table Structure Recognition (TSR).
    Returns: (soup, struc_cells)
        soup -> BeautifulSoup object with <table> HTML (without text inside cells)
        struc_cells -> list of bounding boxes for each cell
    """
    print(f"Processing table image: {img_file}")

    rows = get_rows_from_tatr(img_file)
    cols = get_cols_from_tatr(img_file)
    print('Physical TSR')
    print(f"{len(rows)} rows detected")
    print(f"{len(cols)} cols detected")

    # Debug visualization
    if len(rows) > 0 or len(cols) > 0:
        debug_img = cv2.imread(img_file)
        if debug_img is not None:
            for row in rows:
                x1_r, y1_r, x2_r, y2_r = map(int, row)
                cv2.rectangle(debug_img, (x1_r, y1_r), (x2_r, y2_r), (0, 0, 255), 2)
            for col in cols:
                x1_c, y1_c, x2_c, y2_c = map(int, col)
                cv2.rectangle(debug_img, (x1_c, y1_c), (x2_c, y2_c), (255, 0, 0), 2)
            cv2.imwrite("debug_rows_cols.jpg", debug_img)

    if len(rows) == 0 or len(cols) == 0:
        print("WARNING: No rows or columns detected!")
        table_img = cv2.imread(img_file)
        if table_img is None:
            html_string = f'<table border="1"><tr><td title="bbox {x1} {y1} {x1+100} {y1+50}">Error: Could not read image</td></tr></table>'
            soup = BeautifulSoup(html_string, 'html.parser')
            return soup, []
        h, w = table_img.shape[:2]
        html_string = f'<table border="1"><tr><td title="bbox {x1} {y1} {x1+w} {y1+h}"></td></tr></table>'
        soup = BeautifulSoup(html_string, 'html.parser')
        return soup, []

    print('Logical TSR')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get logical table structure (OTSL sequence)
    otsl_string = get_logical_structure(img_file, device)

    # Align with detected rows/cols
    corrected_otsl = align_otsl_from_rows_cols(otsl_string, len(rows), len(cols))

    # Normalize cell types to 'C'
    corrected_otsl = corrected_otsl.replace("E", "C").replace("F", "C")

    print('OTSL => ' + otsl_string)
    print('Corrected OTSL => ' + corrected_otsl)

    # Convert structure to HTML table
    cells = get_cells_from_rows_cols(rows, cols)
    html_string, struc_cells = convert_to_html(corrected_otsl, len(rows), len(cols), cells)

    # Wrap in HTML
    soup = BeautifulSoup('<html>' + html_string + '</html>', 'html.parser')

    # IMPORTANT: Cells are empty here (no OCR text inserted)
    return soup, struc_cells


# def get_table_structures(image_file):
#     """
#     Detect tables in a page and extract only table structures (no OCR).
#     Returns: list of (html_soup, table_bbox)
#     """
#     final_structs = []
#     try:
#         image = cv2.imread(image_file)
#         if image is None:
#             print(f"Error: Could not load image {image_file}")
#             return []

#         print(f"Processing image: {image_file} (shape: {image.shape})")

#         dets = perform_td(image_file)
#         print(f"{len(dets)} tables detected")

#         # Debug visualization
#         if len(dets) > 0:
#             debug_img = image.copy()
#             for i, det in enumerate(dets):
#                 x1, y1, x2, y2 = map(int, det)
#                 cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
#                 cv2.putText(debug_img, f"Table {i}", (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#             cv2.imwrite("detected_tables.jpg", debug_img)

#         for i, det in enumerate(dets):
#             try:
#                 x1, y1, x2, y2 = map(int, det)
#                 print(f"Processing table {i+1}: bbox({x1}, {y1}, {x2}, {y2})")

#                 # Validate bounding box
#                 if x2 <= x1 or y2 <= y1:
#                     print(f"Invalid bounding box for table {i+1}")
#                     continue

#                 # Ensure coordinates are within image bounds
#                 h, w = image.shape[:2]
#                 x1 = max(0, min(x1, w-1))
#                 y1 = max(0, min(y1, h-1))
#                 x2 = max(x1+1, min(x2, w))
#                 y2 = max(y1+1, min(y2, h))

#                 cropped_img = image[y1:y2, x1:x2]
#                 if cropped_img.size == 0:
#                     print(f"Empty crop for table {i+1}")
#                     continue

#                 tmp_path = "temp.jpg"
#                 # Convert BGR to RGB for matplotlib
#                 cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
#                 plt.imsave(tmp_path, cropped_rgb)

#                 soup, struct_cells = perform_tsr(tmp_path, x1, y1)
                
#                 # Adjust cell coordinates to absolute image coordinates
#                 if soup and struct_cells:
#                     for td in soup.find_all('td'):
#                         if 'title' in td.attrs:
#                             bbox_str = td['title'].replace('bbox ', '')
#                             try:
#                                 coords = list(map(int, bbox_str.split()))
#                                 if len(coords) == 4:
#                                     # Adjust relative coordinates to absolute
#                                     coords[0] = coords[0] + x1
#                                     coords[1] = coords[1] + y1
#                                     coords[2] = coords[2] + x1
#                                     coords[3] = coords[3] + y1
#                                     td['title'] = f'bbox {coords[0]} {coords[1]} {coords[2]} {coords[3]}'
#                             except (ValueError, IndexError) as e:
#                                 print(f"Error adjusting coordinates: {e}")
                
#                 final_structs.append([soup, [x1, y1, x2, y2]])

#             except Exception as e:
#                 print(f"Error processing table {i+1}: {e}")
#                 continue

#     except Exception as e:
#         print(f"Error in get_table_structures: {e}")
#         return []

#     return final_structs



def get_table_structures(image_file):
    """Returns: list of [soup, table_bbox, cell_coordinates]"""
    final_structs = []
    try:
        image = cv2.imread(image_file)
        if image is None:
            return []

        dets = perform_td(image_file)
        
        for i, det in enumerate(dets):
            try:
                x1, y1, x2, y2 = map(int, det)
                
                # Validate coordinates
                h, w = image.shape[:2]
                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(x1+1, min(x2, w))
                y2 = max(y1+1, min(y2, h))

                cropped_img = image[y1:y2, x1:x2]
                if cropped_img.size == 0:
                    continue

                tmp_path = "temp.jpg"
                cropped_rgb = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
                plt.imsave(tmp_path, cropped_rgb)

                soup, struct_cells = perform_tsr(tmp_path, 0, 0)  # Use 0,0 since we'll adjust later
                
                # Convert cell coordinates to absolute image coordinates
                absolute_cells = []
                for cell in struct_cells:
                    if len(cell) >= 4:
                        # Add table offset to get absolute coordinates
                        abs_cell = [cell[0] + x1, cell[1] + y1, cell[2] + x1, cell[3] + y1]
                        absolute_cells.append(abs_cell)
                
                final_structs.append([soup, [x1, y1, x2, y2], absolute_cells])
                
            except Exception as e:
                print(f"Error processing table {i+1}: {e}")
                continue

    except Exception as e:
        print(f"Error in get_table_structures: {e}")
        return []

    return final_structs


# Add this function to match the import in __init__.py
def get_table_hocrs(image_file, lang=None, reader=None):
    """
    Wrapper function to maintain compatibility with existing imports.
    This version only returns table structures without OCR text.
    """
    print("Note: This version extracts table structures only (no OCR)")
    structures = get_table_structures(image_file)
    
    # Convert to format expected by existing code
    final_hocrs = []
    for soup, bbox in structures:
        final_hocrs.append([soup, bbox])
    
    return final_hocrs


if __name__ == "__main__":
    img_path = r"C:\Users\91736\Downloads\hin-2.png"
    if os.path.exists(img_path):
        # Extract table structures
        tables = get_table_structures(img_path)
        for soup, bbox in tables:
            print("Detected table at:", bbox)
            print(soup.prettify())
        
        # Export layout data in OCR-compatible format
        print(f"\nExtracted {len(layout_data)} table cells:")
        for i, cell_coords in enumerate(layout_data[:10]):  # Show first 10
            print(f"Cell {i+1}: {cell_coords}")
        if len(layout_data) > 10:
            print(f"... and {len(layout_data) - 10} more cells")
            
    else:
        print(f"Image file not found: {img_path}")
        print("Please provide a valid image path for testing.")
        print("\nUsage example:")
        print("from tables.main import export_table_layout")
        print("layout_data = export_table_layout('your_image.jpg', 'output_layout.txt')")