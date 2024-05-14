from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from PIL import Image
import io


def upload_data_to_front():
    pass

app = FastAPI()

# 上传图片文件并返回其内容
@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    return StreamingResponse(io.BytesIO(contents), media_type="image/png")
