# BhaashaOCR: Printed End-to-End Page Recognition for Indic Languages

**BhaashaOCR** is a powerful End-to-End Page Recognition system designed specifically for Indic languages. It seamlessly handles both text recognition and complex layout analysis, ensuring that document structure is preserved.

## 🌟 Key Updates 

Significant enhancements have been made to the project to support complex document structures, specifically through the addition of a **Table Detection & Construction** pipeline.

**Tapas** has integrated a multi-stage approach to handle tabular data:
1.  **Table Detection**: Utilizes **YOLO** to accurately identify table boundaries within a page.
2.  **Structure Analysis**: Deploys a **Transformer-based model (TATR)** to recognize internal structures like rows, columns, and spanning cells.
3.  **Holistic Reconstruction**: Intelligently merges OCR text with the detected structure to generate fully formatted **HTML** and **JSON** outputs.

*This work transforms the tool from a standard text recognizer into a comprehensive document digitization solution.*

## Setup

### Prerequisites
- **Python**: 3.10+
- **Tesseract OCR** (V5.4.1): Install from [Official Source](https://tesseract-ocr.github.io/tessdoc/Installation.html).

### Installation
1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Pretrained Models
Download various pretrained models for End-to-End Recognition from the [Assets](https://github.com/NLTM-OCR/BhaashaOCR/releases) page.

## Inference

The main entry point for the project is `infer.py`. It orchestrates the entire pipeline: text detection, table analysis, OCR, and output generation.

### Usage
```bash
python infer.py \
  --pretrained=/path/to/models/bengali \
  --image_path=/path/to/image.jpg \
  --out_dir=/path/to/output_folder
```

### Arguments
*   `--pretrained`: Path to the unzipped folder containing layout/ocr model files.
*   `--image_path`: Path to the input image.
*   `--out_dir`: Directory where outputs (`ocr.txt`, `layout.txt`, `result.json`, `reconstructed_table.html`) will be saved.

## Docker Usage

The project uses `uv` for package management within Docker.

### Build
```bash
docker build -t bhaashaocr .
```

### Run
```bash
docker run -d --rm --gpus all \
	--name bhaashaocr-container \
	-v <path_to_model_folder>:/model:ro \
	-v <path_to_data_folder>:/data \
	bhaashaocr \
	uv python infer.py \
	--pretrained /model \
	--image_path /data/image.jpg \
	--out_dir /data/output
```
View logs: `docker logs -f bhaashaocr-container`

## Contact & Credits

- **Table Detection & Construction**: Developed by [Tapas Barman](mailto:tb61946@gmail.com).
- **Project Maintainers**: [Ajoy Mondal](mailto:ajoy.mondal@iiit.ac.in), [Krishna Tulsyan](mailto:krishna.tulsyan@research.iiit.ac.in).

### Citation
```bibtex
@InProceedings{iiit_hw,
	author="Gongidi, Santhoshini and Jawahar, C. V.",
	editor="Llad{\'o}s, Josep and Lopresti, Daniel and Uchida, Seiichi",
	title="iiit-indic-hw-words: A Dataset for Indic Handwritten Text Recognition",
	booktitle="Document Analysis and Recognition -- ICDAR 2021",
	year="2021",
	pages="444--459"
}
```
