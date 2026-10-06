import io

import cv2
import numpy as np
import pydicom

from PIL import Image


def load_medical_image(file):

    filename = file.filename.lower()

    # HANDLE DICOM FILES
    if filename.endswith(".dcm"):

        file_bytes = file.file.read()

        dicom_file = pydicom.dcmread(
            io.BytesIO(file_bytes)
        )

        image = dicom_file.pixel_array.astype(np.float32)


        # Normalize pixel values to 0-255
        image -= image.min()

        if image.max() != 0:
            image /= image.max()

        image = (image * 255).astype(np.uint8)


        # Convert grayscale → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2RGB
        )

        return Image.fromarray(image)

    # HANDLE PNG / JPG / JPEG
    else:

        image = Image.open(file.file).convert("RGB")

        image_np = np.array(image)


        # Basic OpenCV preprocessing

        image_np = cv2.resize(
            image_np,
            (224, 224)
        )


        return Image.fromarray(image_np)