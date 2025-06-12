import tensorflow as tf
import numpy as np
from PIL import Image
import os
import logging

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_path="plant_recognition_model.h5", labels_path="labels.txt"):
        """
        Bitki tanıma servisini başlatır ve modeli yükler
        
        Args:
            model_path: .h5 model dosyasının yolu
            labels_path: Etiketlerin bulunduğu dosyanın yolu
        """
        # Model yolu kontrolü
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        # Etiketler dosyası kontrolü
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Etiketler dosyası bulunamadı: {labels_path}")
        
        # Modeli yükleyelm
        try:
            logger.info(f"Model yükleniyor: {model_path}")
            self.model = tf.keras.models.load_model(model_path)
            logger.info("Model başarıyla yüklendi")
        except Exception as e:
            logger.error(f"Model yüklenirken hata oluştu: {str(e)}")
            raise
        
        # Sınıf etiketlerini yükleyelim
        try:
            with open(labels_path, 'r', encoding='utf-8') as f:
                self.labels = [line.strip() for line in f.readlines() if line.strip()]
            logger.info(f"{len(self.labels)} sınıf etiketi yüklendi")
        except Exception as e:
            logger.error(f"Etiketler yüklenirken hata oluştu: {str(e)}")
            raise
    
    def preprocess_image(self, image):
        """
        Gelen görseli model için uygun formata dönüştürür
        
        Args:
            image: PIL Image nesnesi
            
        Returns:
            Ön işleme yapılmış tensor
        """
        # Görseli 224x224 boyutuna getir
        image = image.resize((224, 224))
        
        # Görseli array'e dönüştür
        img_array = np.array(image)
        
        # Batch boyutu ekle
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalizasyon (0-1 aralığına getir)
        img_array = img_array / 255.0
        
        return img_array
    
    def predict(self, image):
        """
        Görsel üzerinde tahmin yapar
        
        Args:
            image: PIL Image nesnesi
            
        Returns:
            tuple: (tahmin_edilen_sınıf, güven_oranı)
        """
        # Görseli önişleme sokar
        processed_img = self.preprocess_image(image)
        
        # Tahmini yapar
        predictions = self.model.predict(processed_img)
        
        # En yüksek olasılıklı sınıfı alır
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]
        
        # Sınıf adını bulur (lablesdaa)
        predicted_class = self.labels[predicted_class_index]
        
        logger.info(f"Tahmin: {predicted_class}, Güven: {confidence:.4f}")
        
        return predicted_class, float(confidence) 