import cv2
import numpy as np


def preprocess_frame(frame):

    frame = cv2.resize(
        frame,
        (224, 224)
    )

    lab = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge(
        (l, a, b)
    )

    frame = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    frame = frame.astype(
        np.float32
    ) / 255.0

    frame = np.transpose(
        frame,
        (2, 0, 1)
    )

    return frame