import os
import cv2
import random
from pathlib import Path
from retinaface import RetinaFace
import numpy as np

# =========================================================
# CONFIG
# =========================================================

DATASET_ROOT = r"D:\dfd"

REAL_ROOT = os.path.join(
    DATASET_ROOT,
    "DFD_original sequences"
)

FAKE_ROOT = os.path.join(
    DATASET_ROOT,
    "DFD_manipulated_sequences",
    "DFD_manipulated_sequences"
)

OUTPUT_ROOT = r"D:\DFD_EXTRACTED"

NUM_REAL_VIDEOS = 50
NUM_FAKE_VIDEOS = 50

FRAMES_PER_VIDEO = 10

RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# =========================================================
# FACE ALIGNMENT
# =========================================================

def align_face(img, landmarks):

    try:
        right_eye = np.array(landmarks['right_eye'])
        left_eye = np.array(landmarks['left_eye'])

        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]

        angle = np.degrees(np.arctan2(dY, dX)) - 180

        if angle < -180:
            angle += 360

        eyes_center = (
            int((left_eye[0] + right_eye[0]) // 2),
            int((left_eye[1] + right_eye[1]) // 2)
        )

        M = cv2.getRotationMatrix2D(
            eyes_center,
            angle,
            1.0
        )

        aligned = cv2.warpAffine(
            img,
            M,
            (img.shape[1], img.shape[0]),
            flags=cv2.INTER_CUBIC
        )

        return aligned, M

    except:
        return img, None


def transform_bbox(bbox, M):

    if M is None:
        return bbox

    x1, y1, x2, y2 = bbox

    points = np.array([
        [x1, y1, 1],
        [x2, y1, 1],
        [x2, y2, 1],
        [x1, y2, 1]
    ])

    transformed = M.dot(points.T).T

    nx1 = np.min(transformed[:, 0])
    nx2 = np.max(transformed[:, 0])

    ny1 = np.min(transformed[:, 1])
    ny2 = np.max(transformed[:, 1])

    return [int(nx1), int(ny1), int(nx2), int(ny2)]


# =========================================================
# FIND VIDEOS
# =========================================================

def get_all_videos(root_dir):

    videos = []

    for ext in ["*.mp4", "*.avi", "*.mov"]:

        videos.extend(
            Path(root_dir).rglob(ext)
        )

    return [str(v) for v in videos]


# =========================================================
# FRAME EXTRACTION
# =========================================================

def process_video(
    video_path,
    output_dir,
    label_prefix,
    max_frames=10,
    margin=1.3
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Failed: {video_path}")
        return

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        max_frames,
        dtype=int
    )

    video_name = Path(video_path).stem

    saved = 0

    for idx in frame_indices:

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret, frame = cap.read()

        if not ret:
            continue

        faces = RetinaFace.detect_faces(frame)

        if type(faces) != dict or len(faces) == 0:
            continue

        largest_face = None
        max_area = 0

        for _, face in faces.items():

            bbox = face['facial_area']

            area = (
                (bbox[2] - bbox[0]) *
                (bbox[3] - bbox[1])
            )

            if area > max_area:
                max_area = area
                largest_face = face

        if largest_face is None:
            continue

        aligned, M = align_face(
            frame,
            largest_face['landmarks']
        )

        x1, y1, x2, y2 = transform_bbox(
            largest_face['facial_area'],
            M
        )

        w = x2 - x1
        h = y2 - y1

        cx = x1 + w // 2
        cy = y1 + h // 2

        nw = int(w * margin)
        nh = int(h * margin)

        nx1 = max(0, cx - nw // 2)
        ny1 = max(0, cy - nh // 2)

        nx2 = min(aligned.shape[1], cx + nw // 2)
        ny2 = min(aligned.shape[0], cy + nh // 2)

        crop = aligned[ny1:ny2, nx1:nx2]

        if crop.size == 0:
            continue

        filename = (
            f"{video_name}_frame_{saved:02d}.jpg"
        )

        save_path = os.path.join(
            output_dir,
            filename
        )

        cv2.imwrite(save_path, crop)

        saved += 1

    cap.release()


# =========================================================
# MAIN
# =========================================================

def main():

    real_out = os.path.join(
        OUTPUT_ROOT,
        "real"
    )

    fake_out = os.path.join(
        OUTPUT_ROOT,
        "fake"
    )

    os.makedirs(real_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)

    # -----------------------------------------------------
    # GET VIDEOS
    # -----------------------------------------------------

    real_videos = get_all_videos(REAL_ROOT)
    fake_videos = get_all_videos(FAKE_ROOT)

    print(f"Found {len(real_videos)} real videos")
    print(f"Found {len(fake_videos)} fake videos")

    # -----------------------------------------------------
    # SELECT FIRST 50
    # -----------------------------------------------------

    real_videos = sorted(real_videos)[:NUM_REAL_VIDEOS]
    fake_videos = sorted(fake_videos)[:NUM_FAKE_VIDEOS]

    # -----------------------------------------------------
    # PROCESS REAL
    # -----------------------------------------------------

    print("\nProcessing REAL videos...\n")

    for i, video in enumerate(real_videos):

        print(
            f"[REAL {i+1}/{len(real_videos)}] "
            f"{Path(video).name}"
        )

        process_video(
            video_path=video,
            output_dir=real_out,
            label_prefix="real",
            max_frames=FRAMES_PER_VIDEO
        )

    # -----------------------------------------------------
    # PROCESS FAKE
    # -----------------------------------------------------

    print("\nProcessing FAKE videos...\n")

    for i, video in enumerate(fake_videos):

        print(
            f"[FAKE {i+1}/{len(fake_videos)}] "
            f"{Path(video).name}"
        )

        process_video(
            video_path=video,
            output_dir=fake_out,
            label_prefix="fake",
            max_frames=FRAMES_PER_VIDEO
        )

    print("\nDONE")
    print(f"Saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()