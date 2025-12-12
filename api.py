# import os
# import shutil
# import json
# import uuid
# import subprocess
# from fastapi import FastAPI, File, Form, UploadFile
# from fastapi.responses import JSONResponse

# app = FastAPI()

# MODEL_DIR = "/models"   # Mounted model path (same as in docker run)
# DATA_BASE_DIR = "/tmp/bhaashaocr"  # Local tmp directory for inputs/outputs
# os.makedirs(DATA_BASE_DIR, exist_ok=True)  # Ensure base directory exists
# DOCKER_IMAGE = "bhaashaocr"  # Docker image name

# @app.post("/pageocr/api")
# async def pageocr_api(
#     language: str = Form(...),
#     version: str = Form(...),
#     modality: str = Form(...),
#     layout_model: str = Form(...),
#     image: UploadFile = File(...)
# ):
#     # Create a unique working directory for this request
#     request_id = str(uuid.uuid4())
#     work_dir = os.path.join(DATA_BASE_DIR, request_id)
#     os.makedirs(work_dir, exist_ok=True)

#     input_image_path = os.path.join(work_dir, image.filename)
#     output_dir = os.path.join(work_dir, "output")
#     os.makedirs(output_dir, exist_ok=True)

#     # Save uploaded image
#     with open(input_image_path, "wb") as f:
#         shutil.copyfileobj(image.file, f)

#     # Build docker run command
#     docker_cmd = [
#         # "python", "infer.py", 
#         # "--pretrained=/model",
#         # "--image_path={}".format(input_image_path),
#         # "--out_dir={}".format(output_dir),
#         # "docker", "run", "--rm", "--gpus", "all",
#         # "-v", f"{MODEL_DIR}:/model:ro",
#         # "-v", f"{work_dir}:/data",
#         # DOCKER_IMAGE,
#         "python", "infer.py",
#         "--pretrained", f"{MODEL_DIR}",
#         "--image_path", f"{input_image_path}",
#         "--out_dir", f"{output_dir}",
#     ]

#     # Run inference
#     try:
#         subprocess.run(docker_cmd, check=True)
#     except subprocess.CalledProcessError as e:
#         return JSONResponse(content={"error": "Model inference failed", "details": str(e)}, status_code=500)

#     # Read result JSON (assuming infer.py outputs a JSON file named result.json in output dir)
#     result_file = os.path.join(output_dir, "result.json")
#     if not os.path.exists(result_file):
#         return JSONResponse(content={"error": "No result generated"}, status_code=500)

#     with open(result_file, "r", encoding="utf-8") as f:
#         result_data = json.load(f)

#     # Cleanup temporary directory (optional, or schedule later)
#     shutil.rmtree(work_dir, ignore_errors=True)

#     return JSONResponse(content=result_data)



import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))

import layoutparser as lp

# Check available model catalogs
try:
    from layoutparser.models.detectron2 import catalog as d2_catalog
    print("Detectron2 models available:")
    print(d2_catalog.MODEL_CATALOG.keys())
except:
    print("Detectron2 catalog not available")

try:
    from layoutparser.models.paddledetection import catalog as pd_catalog
    print("PaddleDetection models available:")
    print(pd_catalog.MODEL_CATALOG.keys())
except:
    print("PaddleDetection catalog not available")