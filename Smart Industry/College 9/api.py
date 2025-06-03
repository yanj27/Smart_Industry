from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import io
from fastapi.responses import StreamingResponse
from motion_detection_utils import *
import tempfile
from io import BytesIO

app = FastAPI()

# CORS configuratie
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/create_motion_mask/")
async def create_motion_mask(file: UploadFile = File(...)):
    tmp_input_path = tempfile.mktemp(suffix=".mp4")
    tmp_output_path = tempfile.mktemp(suffix=".mp4")

    contents = await file.read()
    with open(tmp_input_path, "wb") as tmp_input:
        tmp_input.write(contents)

    cap = cv2.VideoCapture(tmp_input_path)
    if not cap.isOpened():
        return {"error": "Het was niet gelukt om de video te openen"}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(tmp_output_path, fourcc, fps, (width, height), isColor=True)

    ret, prev_frame = cap.read()
    if not ret:
        return {"error": "Video is ongeldig"}

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        mask = get_mask(prev_gray, gray)

        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        out.write(mask_bgr)

        prev_gray = gray

    cap.release()
    out.release()


    if not os.path.exists(tmp_output_path):
        return {"error": "Het was niet gelukt om een motion mask video te maken..."}
    
    file_size = os.path.getsize(tmp_output_path)
    if file_size == 0:
        return {"error": "Video werd niet gemaakt, probeer opnieuw..."}

    def iterfile():
        with open(tmp_output_path, "rb") as f:
            yield from f

    headers = {"Content-Disposition": f"attachment; filename=motion_mask_{file.filename}"}
    return StreamingResponse(iterfile(), media_type="video/mp4", headers=headers)


@app.post("/bgr_subtraction/")
async def create_motion_mask(file: UploadFile = File(...)):
    tmp_input_path = "temp_video.mp4"
    contents = await file.read()
    with open(tmp_input_path, "wb") as tmp_input:
        tmp_input.write(contents)

    cap = cv2.VideoCapture(tmp_input_path)
    if not cap.isOpened():
        return {"error": "Het was niet gelukt om de video te openen"}

    backSub = cv2.createBackgroundSubtractorKNN(dist2Threshold=1000, detectShadows=True)

    ret, frame = cap.read() # ret geeft True of False terug: geeft aan of het gelukt was om de frame te krijgen
    if not ret:
        cap.release()
        return {"error": "Video is ongeldig"}

    while ret: # Loop door alle frames van de gegeven video
        backSub.apply(frame)
        ret, frame = cap.read()

    cap.release()

    background = backSub.getBackgroundImage()

    if background is None or np.sum(background) == 0:
        return {"error": "Het was niet gelukt, probeer opnieuw"}

    _, img_bytes = cv2.imencode('.png', background)
    byte_io = BytesIO(img_bytes)

    return StreamingResponse(byte_io, media_type="image/png")