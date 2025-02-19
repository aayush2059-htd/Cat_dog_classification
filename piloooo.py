from PIL import Image
import os

def check_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # Check if the image is valid
            except (IOError, SyntaxError):
                print(f"Corrupt file removed: {file_path}")
                os.remove(file_path)  # Remove corrupt file

check_images("images/train")
check_images("images/test")
