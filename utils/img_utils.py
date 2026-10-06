from PIL import Image

def load_image(file_path):
    image=Image.open(file_path)
    image=image.convert("RGB")
    return image