import argparse
import os
import shutil

import cv2
import numpy as np
import torch
from torchvision import transforms

from face_detection.scrfd.detector import SCRFD
# from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import read_features

# Check if CUDA is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the face detector (Choose one of the detectors)
# detector = Yolov5Face(model_file="face_detection/yolov5_face/weights/yolov5n-face.pt")
detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")

# Initialize the face recognizer
recognizer = iresnet_inference(
    model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=device
)


@torch.no_grad()
def get_feature(face_image):
    """
    Extract facial features from an image using the face recognition model.

    Args:
        face_image (numpy.ndarray): Input facial image.

    Returns:
        numpy.ndarray: Extracted facial features.
    """
    # Define a series of image preprocessing steps
    face_preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Convert the image to RGB format
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

    # Apply the defined preprocessing to the image
    face_image = face_preprocess(face_image).unsqueeze(0).to(device)

    # Use the model to obtain facial features
    emb_img_face = recognizer(face_image)[0].cpu().numpy()

    # Normalize the features
    images_emb = emb_img_face / np.linalg.norm(emb_img_face)
    return images_emb


def add_persons(backup_dir, add_persons_dir, faces_save_dir, features_path):
    """
    Add a new person to the face recognition database.

    Args:
        backup_dir (str): Directory to save backup data.
        add_persons_dir (str): Directory containing images of the new person.
        faces_save_dir (str): Directory to save the extracted faces.
        features_path (str): Path to save face features.
    """
    # Initialize lists to store names and features of added images
    images_name = []
    images_emb = []

    # Read the folder with images of the new person, extract faces, and save them
    for name_person in os.listdir(add_persons_dir):
        person_image_path = os.path.join(add_persons_dir, name_person)

        # Create a directory to save the faces of the person
        person_face_path = os.path.join(faces_save_dir, name_person)
        os.makedirs(person_face_path, exist_ok=True)

        for image_name in os.listdir(person_image_path):
            if image_name.endswith(("png", "jpg", "jpeg")):
                input_image = cv2.imread(os.path.join(person_image_path, image_name))

                # Detect faces and landmarks using the face detector
                bboxes, landmarks = detector.detect(image=input_image)

                # Check if no faces are detected
                if len(bboxes) == 0:
                    return f"Error: No face detected in image {image_name} for student {name_person}."

                # Check if more than one face is detected
                if len(bboxes) > 1:
                    return f"Error: More than one face detected in image {image_name} for student {name_person}."

                # Extract faces
                for i in range(len(bboxes)):
                    # Get the number of files in the person's path
                    number_files = len(os.listdir(person_face_path))

                    # Get the location of the face
                    x1, y1, x2, y2, score = bboxes[i]

                    # Extract the face from the image
                    face_image = input_image[y1:y2, x1:x2]

                    # Path to save the face
                    path_save_face = os.path.join(person_face_path, f"{number_files}.jpg")

                    # Save the face to the database
                    cv2.imwrite(path_save_face, face_image)

                    # Extract features from the face
                    images_emb.append(get_feature(face_image=face_image))
                    images_name.append(name_person)

    # Check if no new person is found
    if images_emb == [] and images_name == []:
        return "No new student found!"

    # Convert lists to arrays
    images_emb = np.array(images_emb)
    images_name = np.array(images_name)

    # Read existing features if available
    features = read_features(features_path)

    if features is not None:
        # Unpack existing features
        old_images_name, old_images_emb = features

        # Combine new features with existing features
        images_name = np.hstack((old_images_name, images_name))
        images_emb = np.vstack((old_images_emb, images_emb))

        print("Update features!")

    # Save the combined features
    np.savez_compressed(features_path, images_name=images_name, images_emb=images_emb)

    # Move the data of the new person to the backup data directory
    for sub_dir in os.listdir(add_persons_dir):
        dir_to_move = os.path.join(add_persons_dir, sub_dir)
        shutil.move(dir_to_move, backup_dir, copy_function=shutil.copytree)

    return "Successfully added new student!"

def delete_person(person_id, features_path, faces_save_dir, backup_dir):
    """
    Delete a person from the face recognition database.

    Args:
        person_id (str): The ID (name) of the person to be deleted.
        features_path (str): Path to the face features file (features.npz).
        faces_save_dir (str): Directory where faces are saved.
        backup_dir (str): Directory where backup data is stored.

    Returns:
        str: Success or error message.
    """
    try:
        # Read existing features
        features = read_features(features_path)
        if features is None:
            return "Error: No features found in the database."

        # Unpack existing features
        images_name, images_emb = features

        # Check if the person exists in the database
        if person_id not in images_name:
            return f"Error: Person with ID '{person_id}' not found in the database."

        # Filter out the features and names of the person to be deleted
        indices_to_keep = images_name != person_id
        updated_images_name = images_name[indices_to_keep]
        updated_images_emb = images_emb[indices_to_keep]

        # Save the updated features
        np.savez_compressed(features_path, images_name=updated_images_name, images_emb=updated_images_emb)
        print("Updated features!")

        # Remove the person's directory from the faces_save_dir
        person_face_path = os.path.join(faces_save_dir, person_id)
        if os.path.exists(person_face_path):
            shutil.rmtree(person_face_path)
            print(f"Removed person's faces from {person_face_path}")

        # Remove the person's directory from the backup_dir
        person_backup_path = os.path.join(backup_dir, person_id)
        if os.path.exists(person_backup_path):
            shutil.rmtree(person_backup_path)
            print(f"Removed person's backup from {person_backup_path}")

        return "Successfully deleted person!"

    except Exception as e:
        return f"Error: An unexpected error occurred - {str(e)}"

def delete(ID):
    return delete_person(person_id=ID, features_path="./datasets/face_features/feature", faces_save_dir="./datasets/data/", backup_dir="./datasets/backup")

def add():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="./datasets/backup",
        help="Directory to save person data.",
    )
    parser.add_argument(
        "--add-persons-dir",
        type=str,
        default="./datasets/new_persons",
        help="Directory to add new persons.",
    )
    parser.add_argument(
        "--faces-save-dir",
        type=str,
        default="./datasets/data/",
        help="Directory to save faces.",
    )
    parser.add_argument(
        "--features-path",
        type=str,
        default="./datasets/face_features/feature",
        help="Path to save face features.",
    )
    opt = parser.parse_args()

    # Run the main function
    return add_persons(**vars(opt))

if __name__ == "__main__":
    add()
