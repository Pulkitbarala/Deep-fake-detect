import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

class Preprocessor:
    def __init__(self, raw_dir, processed_dir, img_size=(224, 224)):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.img_size = img_size
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def preprocess_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            # Face Detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) == 0:
                # If no face detected, resize the whole image
                face_img = img
            else:
                # Use the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                face_img = img[y:y+h, x:x+w]
            
            # Resize
            face_img = cv2.resize(face_img, self.img_size)
            
            # Illumination Normalization (Y channel histogram equalization)
            img_yuv = cv2.cvtColor(face_img, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            face_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            
            return face_img
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None

    def augment_image(self, img):
        augmented_images = []
        # Original
        augmented_images.append(img)
        # Horizontal Flip
        augmented_images.append(cv2.flip(img, 1))
        # Small Rotation
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1) # 5 degrees
        rotated = cv2.warpAffine(img, M, (cols, rows))
        augmented_images.append(rotated)
        
        return augmented_images

    def run(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
            
        categories = ['real', 'fake']
        
        for category in categories:
            raw_path = os.path.join(self.raw_dir, category)
            processed_path = os.path.join(self.processed_dir, category)
            
            if not os.path.exists(processed_path):
                os.makedirs(processed_path)
                
            if not os.path.exists(raw_path):
                print(f"Warning: Directory {raw_path} does not exist.")
                continue
                
            print(f"Processing {category} images...")
            for img_name in tqdm(os.listdir(raw_path)):
                img_path = os.path.join(raw_path, img_name)
                processed_img = self.preprocess_image(img_path)
                
                if processed_img is not None:
                    aug_images = self.augment_image(processed_img)
                    base, ext = os.path.splitext(img_name)
                    
                    for i, aug_img in enumerate(aug_images):
                        save_name = f"{base}_aug{i}.jpg"
                        save_path = os.path.join(processed_path, save_name)
                        cv2.imwrite(save_path, aug_img)

if __name__ == "__main__":
    raw_dir = "data_raw"
    processed_dir = "data_processed"
    preprocessor = Preprocessor(raw_dir, processed_dir)
    preprocessor.run()
