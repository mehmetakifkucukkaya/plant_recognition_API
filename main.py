from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import uvicorn
from PIL import Image
import io
import os
from prediction_service import PredictionService
from disease_detection_service import DiseaseDetectionService

app = FastAPI(
    title="Bitki Tanıma API",
    description="Görsellerden bitki türünü ve hastalığını tanıyan REST API",
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

# Servis başlatıldığında model yüklenir
try:
    prediction_service = PredictionService(model_path=MODEL_PATH, labels_path=LABELS_PATH)
except FileNotFoundError as e:
    print(f"UYARI: {str(e)}")
    print("API çalışmaya devam edecek ancak tahmin endpoint'i çalışmayacak.")
    prediction_service = None

# Bitki hastalığı tespiti servisi
try:
    disease_detection_service = DiseaseDetectionService()
except Exception as e:
    print(f"UYARI: Hastalık tespit modeli yüklenemedi: {str(e)}")
    disease_detection_service = None

@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang=\"tr\">
    <head>
        <meta charset=\"UTF-8\">
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
        <title>Bitki Tanıma API</title>
        <link href=\"https://fonts.googleapis.com/css2?family=Montserrat:wght@600;700&display=swap\" rel=\"stylesheet\">
        <style>
            body {
                background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%);
                min-height: 100vh;
                margin: 0;
                font-family: 'Montserrat', 'Segoe UI', Arial, sans-serif;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .card {
                background: rgba(255,255,255,0.95);
                border-radius: 22px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
                padding: 56px 44px 40px 44px;
                max-width: 440px;
                width: 100%;
                text-align: center;
                position: relative;
                overflow: hidden;
            }
            .card::before {
                content: '';
                position: absolute;
                top: -60px; left: -60px;
                width: 120px; height: 120px;
                background: linear-gradient(135deg, #2d7ff9 0%, #6ee7b7 100%);
                border-radius: 50%;
                opacity: 0.12;
            }
            .logo {
                font-size: 3.2rem;
                margin-bottom: 14px;
                animation: pop 1.2s cubic-bezier(.23,1.12,.72,1.11) 1;
            }
            @keyframes pop {
                0% { transform: scale(0.7); opacity: 0; }
                80% { transform: scale(1.15); opacity: 1; }
                100% { transform: scale(1); }
            }
            h1 {
                font-size: 2.2rem;
                margin: 0 0 14px 0;
                color: #2d7ff9;
                font-weight: 700;
                letter-spacing: 0.5px;
            }
            p {
                color: #333;
                font-size: 1.13rem;
                margin-bottom: 22px;
                line-height: 1.6;
            }
            .links {
                margin: 22px 0 0 0;
            }
            .links a {
                display: inline-block;
                margin: 0 12px;
                color: #fff;
                background: linear-gradient(90deg, #2d7ff9 0%, #6ee7b7 100%);
                text-decoration: none;
                font-weight: 700;
                padding: 10px 22px;
                border-radius: 8px;
                box-shadow: 0 2px 8px #2d7ff93a;
                transition: background 0.2s, box-shadow 0.2s, color 0.2s;
            }
            .links a:hover {
                background: linear-gradient(90deg, #174ea6 0%, #34d399 100%);
                color: #fff;
                box-shadow: 0 4px 16px #2d7ff94a;
            }
            .endpoint {
                background: #f3f6fa;
                border-radius: 8px;
                padding: 8px 14px;
                font-family: monospace;
                color: #2d7ff9;
                margin: 14px 0 0 0;
                display: inline-block;
                font-size: 1.08rem;
                font-weight: 600;
                letter-spacing: 0.2px;
            }
            .footer {
                margin-top: 32px;
                color: #aaa;
                font-size: 0.98rem;
            }
            @media (max-width: 600px) {
                .card { padding: 28px 6px 18px 6px; }
                h1 { font-size: 1.3rem; }
            }
        </style>
    </head>
    <body>
        <div class=\"card\">
            <div class=\"logo\">🌱</div>
            <h1>Bitki Tanıma API</h1>
            <p>Yapay zeka destekli bu API, yüklediğiniz görseldeki bitki türünü ve hastalığını otomatik olarak tanır.<br><br>
            <b>Kullanım için:</b> <span class=\"endpoint\">POST /predict/</span> veya <span class=\"endpoint\">POST /predict-disease/</span> endpointine bir görsel yükleyin.</p>
            <div class=\"links\">
                <a href=\"/docs\">Swagger Dokümantasyonu</a>
                <a href=\"https://github.com/mehmetakifkucukkaya/plant_recognition_API\" target=\"_blank\">GitHub</a>
            </div>
            <div class=\"footer\">&copy; 2025 Bitki Tanıma API &middot; <a href=\"https://github.com/mehmetakifkucukkaya/plant_recognition_API\" style=\"color:#2d7ff9;text-decoration:none;\" target=\"_blank\">mehmetakifkucukkaya</a></div>
        </div>
    </body>
    </html>
    """

@app.post("/predict/", 
         summary="Bitki görseli yükleyerek tür tahmini yapma",
         description="Yüklenen görselden bitki türünü tahmin eder ve güven oranını döndürür.")
async def predict_plant(image: UploadFile = File(...)):
    if prediction_service is None:
        raise HTTPException(
            status_code=503,
            detail="Model yüklenemedi. Lütfen sistem yöneticisiyle iletişime geçin."
        )
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, 
            detail="Sadece JPEG veya PNG formatındaki görseller kabul edilmektedir."
        )
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        prediction, confidence = prediction_service.predict(img)
        return {
            "predicted_class": prediction,
            "confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin sırasında bir hata oluştu: {str(e)}"
        )

@app.post("/predict-disease/", 
         summary="Bitki görseli yükleyerek hastalık tahmini yapma",
         description="Yüklenen görselden bitki hastalığını tahmin eder ve güven oranını döndürür.")
async def predict_disease(image: UploadFile = File(...)):
    if disease_detection_service is None:
        raise HTTPException(
            status_code=503,
            detail="Hastalık tespit modeli yüklenemedi. Lütfen sistem yöneticisiyle iletişime geçin."
        )
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400, 
            detail="Sadece JPEG veya PNG formatındaki görseller kabul edilmektedir."
        )
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents))
        prediction, confidence = disease_detection_service.predict(img)
        return {
            "predicted_disease": prediction,
            "confidence": float(confidence)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Tahmin sırasında bir hata oluştu: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True) 