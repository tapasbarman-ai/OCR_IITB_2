from run_pipeline import run_ocr_pipeline

result = run_ocr_pipeline(
    image_path=r"C:\Users\91736\Downloads\hindi_text\sample.jpg",
    pretrained_dir=r"C:\Users\91736\Downloads\pretrained_models",
    out_dir=r"C:\Users\91736\Downloads\hindi_text"
)

print(result["text"])   # full recognized text
