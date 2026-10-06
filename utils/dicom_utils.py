import pydicom

def load_dicom(file_path):
    dicom_data=pydicom.dcmread(file_path)
    return dicom_data
