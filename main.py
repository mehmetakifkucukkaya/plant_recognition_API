from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from PIL import Image
import io
import os
from prediction_service import PredictionService

app = FastAPI(
    title="Bitki Tanıma API",
    description="Görsellerden bitki türünü tanıyan REST API",
    version="1.0.0"
)

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tüm originlere izin ver (Üretimde değiştirilmeli)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = os.environ.get("MODEL_PATH", "plant_recognition_model.h5")
LABELS_PATH = os.environ.get("LABELS_PATH", "labels.txt")

# Servis başlatıldığında modeli yükleyelim
try:
    prediction_service = PredictionService(model_path=MODEL_PATH, labels_path=LABELS_PATH)
except FileNotFoundError as e:
    print(f"UYARI: {str(e)}")
    print("API çalışmaya devam edecek ancak tahmin endpoint'i çalışmayacak.")
    prediction_service = None

@app.get("/")
async def root():
    return {"message": "Bitki Tanıma API'sine Hoş Geldiniz! /docs adresinden dökümantasyona ulaşabilirsiniz."}

@app.post("/predict/", 
         summary="Bitki görseli yükleyerek tahmin yapma",
         description="Yüklenen görselden bitki türünü tahmin eder ve güven oranını döndürür.")
async def predict_plant(image: UploadFile = File(...)):
    # Model yüklü değilse hata veriyor
    if prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail="Model yüklenemedi. Lütfen sistem yöneticisiyle iletişime geçin."
        )
        
    # Dosya formatını kontrol ediyorz
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, 
            detail="Sadece JPEG veya PNG formatındaki görseller kabul edilmektedir."
        )
    
    try:
        # Görsel içeriğini oku
        contents = await image.read()
        
        # Bytes'tan PIL Image'a dönüştürme
        img = Image.open(io.BytesIO(contents))
        
        # Tahmin yap
        prediction, confidence = prediction_service.predict(img)
        
        # Sonuç döndür
        return {
            "predicted_class": prediction,
            "confidence": float(confidence)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin sırasında bir hata oluştu: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 