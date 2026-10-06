import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException

router=APIRouter(prefix="/api/vi",tags=["File Upload"])
upload_dir="uploads"
os.makedirs(upload_dir,exist_ok=True)

@router.post("/upload")
async def upload_file(file: UploadFile=File(...)):
    file_path=os.path.join(upload_dir,file.filename)
    with open(file_path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
    return{"filename":file.filename,"content_type":file.content_type,"message":"file uploaded!"}