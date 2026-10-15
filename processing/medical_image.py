import io
import cv2
import numpy as np
import pydicom
from PIL import Image

class MedicalImageProcessor:

    EXTENSION={".jpg",".jpeg",".png",".dcm"}

    def load(self,file):
        filename=file.filename.lower()
        if filename.endswith(".dcm"):
            return self._load_dicom(file)
        return self._load_standard_image(file)

    def _load_dicom(self, file):
        file_bytes=file.file.read()
        dicom_file=pydicom.dcmread(io.BytesIO(file_bytes))
        image=(dicom_file.pixel_array.astype(np.float32))
        image-=image.min()
        if image.max()!=0:
            image/=image.max()
        image=(image*255).astype(np.uint8)
        image=cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
        return Image.fromarray(image)

    def _load_standard_image(self,file):
        image=(Image.open(file.file).convert("RGB"))
        image_np=np.array(image)
        image_np=cv2.resize(image_np,(224,224))
        return Image.fromarray(image_np)