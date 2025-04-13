import cv2
import math
import numpy as np
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image


def norm_coordinates(normalized_x, normalized_y, image_width, image_height):

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)

    return x_px, y_px


def get_box(fl, w, h):
    idx_to_coors = {}
    for idx, landmark in enumerate(fl.landmark):
        landmark_px = norm_coordinates(landmark.x, landmark.y, w, h)

        if landmark_px:
            idx_to_coors[idx] = landmark_px

    x_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 0])
    y_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 1])
    endX = np.max(np.asarray(list(idx_to_coors.values()))[:, 0])
    endY = np.max(np.asarray(list(idx_to_coors.values()))[:, 1])

    (startX, startY) = (max(0, x_min), max(0, y_min))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

    return startX, startY, endX, endY


def prepare(image):
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image)
        if not results.multi_face_landmarks:
            raise ValueError()
        for fl in results.multi_face_landmarks:
            startX, startY, endX, endY = get_box(fl, image.shape[1], image.shape[0])
            face = image[startY:endY, startX:endX]
            return cv2.resize(face, (250, 250), interpolation=cv2.INTER_LANCZOS4)


class PTHPipeLine:
    def __init__(self, model):
        self.model = model

    def pth_processing(self, fp):
        class PreprocessInput(torch.nn.Module):
            def init(self):
                super(PreprocessInput, self).init()

            def forward(self, x):
                x = x.to(torch.float32)
                x = torch.flip(x, dims=(0,))
                x[0, :, :] -= 91.4953
                x[1, :, :] -= 103.8827
                x[2, :, :] -= 131.0912
                return x

        def get_img_torch(img):
            img = img.resize((224, 224), Image.Resampling.NEAREST)
            ttransform = transforms.Compose(
                [transforms.PILToTensor(), PreprocessInput()]
            )
            img = ttransform(img)
            img = torch.unsqueeze(img, 0)
            return img

        return get_img_torch(fp)

    def norm_coordinates(self, normalized_x, normalized_y, image_width, image_height):
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)

        return x_px, y_px

    def get_box(self, fl, w, h):
        idx_to_coors = {}
        for idx, landmark in enumerate(fl.landmark):
            landmark_px = self.norm_coordinates(landmark.x, landmark.y, w, h)

            if landmark_px:
                idx_to_coors[idx] = landmark_px

        x_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 0])
        y_min = np.min(np.asarray(list(idx_to_coors.values()))[:, 1])
        endX = np.max(np.asarray(list(idx_to_coors.values()))[:, 0])
        endY = np.max(np.asarray(list(idx_to_coors.values()))[:, 1])

        (startX, startY) = (max(0, x_min), max(0, y_min))
        (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

        return startX, startY, endX, endY

    def __call__(self, image_path=None, image=None):
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            if image_path:
                image = np.array(Image.open(image_path))
            results = face_mesh.process(image)
            if not results.multi_face_landmarks:
                return 0
            for fl in results.multi_face_landmarks:
                startX, startY, endX, endY = self.get_box(
                    fl, image.shape[1], image.shape[0]
                )
                cur_face = image[startY:endY, startX:endX]
                cur_face = self.pth_processing(Image.fromarray(cur_face))

                output = (
                    torch.nn.functional.softmax(self.model(cur_face), dim=1)
                    .cpu()
                    .detach()
                    .numpy()
                )

                cl = np.argmax(output)
                return pth_idx_label_map[cl], output.max()


pth_idx_label_map = {
    0: 'Neutral',
    1: 'Happiness',
    2: 'Sadness',
    3: 'Surprise',
    4: 'Fear',
    5: 'Disgust',
    6: 'Anger',
}

tf_idx_label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
tf_label_idx_map = {'negative': 0, 'neutral': 1, 'positive': 2}
