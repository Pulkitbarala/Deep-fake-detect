import os
import cv2
import numpy as np

input_dir = "data_raw"
output_dir = "data_processed"
img_size = (224, 224)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

os.makedirs(output_dir, exist_ok=True)

def enhance_lighting(img_rgb):
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge((y_eq, cr, cb))
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2RGB)

def augment_images(img_rgb):
    aug_images = []

    flipped = cv2.flip(img_rgb, 1)
    aug_images.append(flipped)

    h, w, _ = img_rgb.shape
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, 10, 1.0)
    rotated = cv2.warpAffine(img_rgb, rot_matrix, (w, h))
    aug_images.append(rotated)

    return aug_images

for class_name in os.listdir(input_dir):
    class_input_path = os.path.join(input_dir, class_name)
    class_output_path = os.path.join(output_dir, class_name)

    if not os.path.isdir(class_input_path):
        continue

    os.makedirs(class_output_path, exist_ok=True)

    for img_name in os.listdir(class_input_path):
        img_path = os.path.join(class_input_path, img_name)

        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                img_rgb = img_rgb[y:y+h, x:x+w]

            img_rgb = cv2.resize(img_rgb, img_size)

            img_rgb = enhance_lighting(img_rgb)

            base_name, ext = os.path.splitext(img_name)
            save_path = os.path.join(class_output_path, base_name + "_refined.jpg")
            cv2.imwrite(save_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            augmented_imgs = augment_images(img_rgb)
            for i, aug_img in enumerate(augmented_imgs):
                aug_save_path = os.path.join(class_output_path, f"{base_name}_aug{i+1}.jpg")
                cv2.imwrite(aug_save_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

        except Exception:
            continue

print("All preprocessing steps completed successfully.")
