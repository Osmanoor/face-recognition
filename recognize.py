import threading
import time
import cv2
import numpy as np
import torch
import yaml
from torchvision import transforms

from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
# from face_detection.yolov5_face.detector import Yolov5Face
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.tracker.byte_tracker import BYTETracker
from face_tracking.tracker.visualize import plot_tracking
from app.services.attendance_management import AttendanceManager


class FaceTrackerRecognizer:
    def __init__(self,camera_ip, config_file="./face_tracking/config/config_tracking.yaml"):
        self.camera_ip = camera_ip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = self.load_config(config_file)
        self.detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
        self.recognizer = iresnet_inference(
            model_name="r100", path="face_recognition/arcface/weights/arcface_r100.pth", device=self.device
        )
        self.images_names, self.images_embs = read_features(feature_path="./datasets/face_features/feature")
        self.id_face_mapping = {}
        self.data_mapping = {
            "raw_image": [],
            "tracking_ids": [],
            "detection_bboxes": [],
            "detection_landmarks": [],
            "tracking_bboxes": [],
        }
        self.frame = None
        self.manager = AttendanceManager()
        self._stop_event = threading.Event()
        self.thread_track = None
        self.thread_recognize = None

    def load_config(self, file_name):
        """Load a YAML configuration file."""
        with open(file_name, "r") as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    def process_tracking(self, frame, detector, tracker, frame_id, fps):
        """Process tracking for a frame."""
        outputs, img_info, bboxes, landmarks = detector.detect_tracking(image=frame, input_size=(256, 256))

        tracking_tlwhs = []
        tracking_ids = []
        tracking_scores = []
        tracking_bboxes = []

        if outputs is not None:
            online_targets = tracker.update(
                outputs, [img_info["height"], img_info["width"]], (256, 256)
            )

            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > self.config["aspect_ratio_thresh"]
                if tlwh[2] * tlwh[3] > self.config["min_box_area"] and not vertical:
                    x1, y1, w, h = tlwh
                    tracking_bboxes.append([x1, y1, x1 + w, y1 + h])
                    tracking_tlwhs.append(tlwh)
                    tracking_ids.append(tid)
                    tracking_scores.append(t.score)

            tracking_image = plot_tracking(
                img_info["raw_img"],
                tracking_tlwhs,
                tracking_ids,
                names=self.id_face_mapping,
                frame_id=frame_id + 1,
                fps=fps,
            )
        else:
            tracking_image = img_info["raw_img"]

        self.data_mapping["raw_image"] = img_info["raw_img"]
        self.data_mapping["detection_bboxes"] = bboxes
        self.data_mapping["detection_landmarks"] = landmarks
        self.data_mapping["tracking_ids"] = tracking_ids
        self.data_mapping["tracking_bboxes"] = tracking_bboxes

        return tracking_image

    @torch.no_grad()
    def get_feature(self, face_image):
        """Extract features from a face image."""
        face_preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112)),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = face_preprocess(face_image).unsqueeze(0).to(self.device)
        emb_img_face = self.recognizer(face_image).cpu().numpy()
        images_emb = emb_img_face / np.linalg.norm(emb_img_face)

        return images_emb

    def recognition(self, face_image):
        """Recognize a face image."""
        query_emb = self.get_feature(face_image)
        score, id_min = compare_encodings(query_emb, self.images_embs)
        name = self.images_names[id_min]
        score = score[0]

        return score, name

    def mapping_bbox(self, box1, box2):
        """Calculate the Intersection over Union (IoU) between two bounding boxes."""
        x_min_inter = max(box1[0], box2[0])
        y_min_inter = max(box1[1], box2[1])
        x_max_inter = min(box1[2], box2[2])
        y_max_inter = min(box1[3], box2[3])

        intersection_area = max(0, x_max_inter - x_min_inter + 1) * max(
            0, y_max_inter - y_min_inter + 1
        )

        area_box1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        area_box2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        union_area = area_box1 + area_box2 - intersection_area

        iou = intersection_area / union_area

        return iou

    def tracking(self):
        """Face tracking in a separate thread."""
        start_time = time.time_ns()
        frame_count = 0
        fps = -1
        tracker = BYTETracker(args=self.config, frame_rate=30)
        frame_id = 0
        cap = cv2.VideoCapture(self.camera_ip)

        while not self._stop_event.is_set():
            _, img = cap.read()

            tracking_image = self.process_tracking(img, self.detector, tracker, frame_id, fps)

            frame_count += 1
            if frame_count >= 30:
                fps = 1e9 * frame_count / (time.time_ns() - start_time)
                frame_count = 0
                start_time = time.time_ns()

            cv2.imshow("Face Recognition", tracking_image)
            self.frame = tracking_image
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def recognize(self):
        """Face recognition in a separate thread."""
        while not self._stop_event.is_set():
            raw_image = self.data_mapping["raw_image"]
            detection_landmarks = self.data_mapping["detection_landmarks"]
            detection_bboxes = self.data_mapping["detection_bboxes"]
            tracking_ids = self.data_mapping["tracking_ids"]
            tracking_bboxes = self.data_mapping["tracking_bboxes"]

            for i in range(len(tracking_bboxes)):
                for j in range(len(detection_bboxes)):
                    mapping_score = self.mapping_bbox(box1=tracking_bboxes[i], box2=detection_bboxes[j])
                    if mapping_score > 0.9:
                        face_alignment = norm_crop(img=raw_image, landmark=detection_landmarks[j])

                        score, name = self.recognition(face_image=face_alignment)
                        if name is not None:
                            if score < 0.25:
                                caption = "UN_KNOWN"
                            else:
                                caption = f"{name}:{score:.2f}"
                                self.manager.add_entrance(name)

                        self.id_face_mapping[tracking_ids[i]] = caption

                        detection_bboxes = np.delete(detection_bboxes, j, axis=0)
                        detection_landmarks = np.delete(detection_landmarks, j, axis=0)

                        break

            if tracking_bboxes == []:
                print("Waiting for a person...")

    def start(self):
        """Start face tracking and recognition threads."""
        self._stop_event.clear()  
        self.thread_track = threading.Thread(target=self.tracking)
        self.thread_recognize = threading.Thread(target=self.recognize)

        self.thread_track.start()
        self.thread_recognize.start()
        
        
        # thread_track.join()
        # thread_recognize.join()

    def stop(self):
        """Stop face tracking and recognition threads safely."""
        self._stop_event.set()  # Signal the threads to stop

        # Wait for both threads to finish
        if self.thread_track and self.thread_track.is_alive():
            self.thread_track.join()
        if self.thread_recognize and self.thread_recognize.is_alive():
            self.thread_recognize.join()

    def get_frames(self):
        while True:
            # ret, frame = self.cap.read()
            if self.frame is None:
                continue
            # Encode frame as JPEG
            _, buffer = cv2.imencode('.jpg', self.frame)
            # Yield frame with headers
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# if __name__ == "__main__":
#     face_tracker_recognizer = FaceTrackerRecognizer(camera_ip="http://192.168.1.143:8080/video")
#     face_tracker_recognizer.start()

#     time.sleep(10)
#     face_tracker_recognizer.stop()
    