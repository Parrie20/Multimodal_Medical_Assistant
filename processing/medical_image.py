import io
import cv2
import numpy as np
import pydicom
from PIL import Image

class MedicalImageProcessor:

    EXTENSION={".jpg",".jpeg",".png",".dcm"}

    def load(self,file):
        