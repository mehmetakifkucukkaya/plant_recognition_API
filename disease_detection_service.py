from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

class DiseaseDetectionService:
    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
        self.model = AutoModelForImageClassification.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
        self.model.eval()

    def predict(self, image: Image.Image):
        # Görseli 224x224'e yeniden boyutlandır
        image = image.resize((224, 224))
        inputs = self.processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            confidence = torch.softmax(logits, dim=1)[0, predicted_class_idx].item()
            predicted_class = self.model.config.id2label[predicted_class_idx]
        return predicted_class, confidence 