import cv2
import numpy as np

class FaceDetectorDNN:
    def __init__(self,
                 model_path="res10_300x300_ssd_iter_140000.caffemodel",
                 config_path="deploy.prototxt",
                 conf_threshold=0.5):

        self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Returns list of detected faces:
        [
            {
                "bbox": (x, y, w, h),
                "center": (cx, cy),
                "confidence": float
            },
            ...
        ]
        """

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w - 1, x2)
                y2 = min(h - 1, y2)

                bw = x2 - x1
                bh = y2 - y1

                cx = x1 + bw // 2
                cy = y1 + bh // 2

                faces.append({
                    "bbox": (x1, y1, bw, bh),
                    "center": (cx, cy),
                    "confidence": float(confidence)
                })

        return faces
