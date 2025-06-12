# Bitki Tanıma API

Bu API, eğitilmiş bir MobileNetV2 modeli kullanarak yüklenen görsellerden bitki türünü tahmin eden bir REST servisi sunar.

## Kurulum

1. Gereksinimleri yükleyin:

```bash
pip install -r requirements.txt
```

2. API'yi çalıştırmak için model ve etiket dosyalarının ana dizinde olduğundan emin olun:
   - `best_mobilenetv2_finetuned_continue.h5` (model dosyamız)
   - `labels.txt` (her satırda bir sınıf adı bulunan etiketler dosyamız)

## Kullanım

API'yi başlatın:

```bash
python main.py
```

veya direkt uvicorn ile:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API aşağıdaki adreste çalışacaktır:
`http://localhost:8000`

## API Endpointleri

### Ana Sayfa

- **URL:** `/`
- **Metot:** `GET`
- **Açıklama:** API'nin çalıştığını doğrulayan bir mesaj döndürür.

### Bitki Tahmin Etme

- **URL:** `/predict/`
- **Metot:** `POST`
- **İstek Türü:** `multipart/form-data`
- **Parametreler:**
  - `image`: Bitki görseli (JPEG veya PNG formatında)
- **Yanıt:** JSON formatında tahmin edilen bitki sınıfı ve güven oranı.

Örnek yanıt:

```json
{
  "predicted_class": "Sümbül",
  "confidence": 0.95
}
```

## Dökümantasyon

API dökümantasyonuna aşağıdaki adresten ulaşabilirsiniz:

- Swagger UI: `http://localhost:8000/docs`
