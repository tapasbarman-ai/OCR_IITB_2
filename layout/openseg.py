import os
from os.path import basename, join

import pytesseract
from PIL import Image


def perform_openseg(image_path, pretrained) -> list[str]:
    try:
        if os.path.exists(pretrained) and os.path.isdir(pretrained):
            os.environ['TESSDATA_PREFIX'] = pretrained
        else:
            raise ValueError(f'Please check the openseg pretrained folder path: {pretrained}')
        print(f'Processing OpenSeg for file: {basename(image_path)}')
        results = pytesseract.image_to_data(
            image_path,
            output_type=pytesseract.Output.DICT,
            # Using multiple layout traineddata files for processing.
            # If a particular layout file is not present, it will be ignored
            lang='layout1+layout2+layout3+layout4'
        )

        regions = []
        for i in range(0, len(results['text'])):
            if int(results['conf'][i]) <= 0:
                # Skipping the region as confidence is too low
                continue
            x = results['left'][i]
            y = results['top'][i]
            w = results['width'][i]
            h = results['height'][i]
            if h < 10:
                # Skipping box as height is too low.
                continue
            regions.append(','.join(list(map(str, [
                x, y, w, h,
                results['line_num'][i] + 1,
            ]))))
        return regions
    except Exception as e:
        print(f"Error processing file {image_path}: {e}")
        return []
    

def generate_hocr(image_path, pretrained, output_file):
    """Generate hOCR format output from the image"""
    try:
        if os.path.exists(pretrained) and os.path.isdir(pretrained):
            os.environ['TESSDATA_PREFIX'] = pretrained
        else:
            raise ValueError(f'Please check the openseg pretrained folder path: {pretrained}')
            
        print(f'Generating hOCR for file: {basename(image_path)}')
        
        # Generate hOCR output using pytesseract
        hocr_output = pytesseract.image_to_pdf_or_hocr(
            image_path,
            extension='hocr',
            lang='layout1+layout2+layout3+layout4'
        )
        
        # Write hOCR to file
        with open(output_file, 'wb') as f:
            f.write(hocr_output)
            
        print(f'hOCR saved to: {output_file}')
        return output_file
        
    except Exception as e:
        print(f"Error generating hOCR for file {image_path}: {e}")
        return None