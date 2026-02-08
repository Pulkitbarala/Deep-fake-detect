import torch
import cv2
import numpy as np
from torchvision import transforms
from model import HybridModel
import os

class Predictor:
    def __init__(self, model_path="best_model.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = HybridModel(num_classes=1).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            print(f"Warning: Model path {model_path} not found.")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def preprocess(self, img):
        if img is None:
            return None
        
        # Face Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) == 0:
            face_img = img
        else:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            face_img = img[y:y+h, x:x+w]
        
        # Resize
        face_img = cv2.resize(face_img, (224, 224))
        
        # Illumination Normalization
        img_yuv = cv2.cvtColor(face_img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
        face_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        return face_img

    def predict(self, img_path_or_array):
        if isinstance(img_path_or_array, str):
            img = cv2.imread(img_path_or_array)
        else:
            img = img_path_or_array
            
        processed_img = self.preprocess(img)
        if processed_img is None:
            return "Error", 0.0
            
        # Transform for model
        input_tensor = self.transform(processed_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.sigmoid(output).item()
            
        label = "Fake" if prob > 0.5 else "Real"
        confidence = prob if prob > 0.5 else 1 - prob
        
        return label, confidence

if __name__ == "__main__":
    predictor = Predictor()
    # Test on a dummy image if exists
    # print(predictor.predict("test_image.jpg"))
