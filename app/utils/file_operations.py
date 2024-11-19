import os
from app.config import Config

def save_image(image, student_id, image_index):
    folder_path = os.path.join(Config.STUDENT_IMAGES_FOLDER, student_id)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    file_path = os.path.join(folder_path, f'image_{image_index}.jpg')
    with open(file_path, 'wb') as f:
        f.write(image.read())
    return file_path