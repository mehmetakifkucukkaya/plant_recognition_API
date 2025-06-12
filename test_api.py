import requests
import sys
from pathlib import Path
import json

def test_predict_endpoint(image_path, api_url="http://localhost:8000"):
    """
    API'nin çalışıp çalışmadığını test etmek için örnek bir görsel ile tahmin yapar
    
    Args:
        image_path: Test edilecek görsel dosyasının yolu
        api_url: API'nin URL'i
    """
    # Dosya kontrolü
    if not Path(image_path).exists():
        print(f"Hata: Belirtilen dosya bulunamadı: {image_path}")
        return
    
    # İstek URL'i
    url = f"{api_url}/predict/"
    
    # Multipart form data hazırla
    files = {
        'image': (Path(image_path).name, open(image_path, 'rb'), 'image/jpeg')
    }
    
    try:
        # İsteği gönderiyrouz
        print(f"İstek gönderiliyor: {url}")
        response = requests.post(url, files=files)
        
        # Yanıtı kontrol ediyoruz
        if response.status_code == 200:
            result = response.json()
            print("Tahmin başarılı!")
            print(f"Tahmin edilen sınıf: {result['predicted_class']}")
            print(f"Güven oranı: {result['confidence']:.4f}")
            return True
        else:
            print(f"Hata! Status code: {response.status_code}")
            print(f"Yanıt: {response.text}")
            return False
    
    except Exception as e:
        print(f"Test sırasında bir hata oluştu: {str(e)}")
        return False
    finally:
        
        files['image'][1].close()

if __name__ == "__main__":
    # Komut satırı argümanı olarak test edilecek görsel yolu alınabilir
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("Kullanım: python test_api.py <görsel_dosya_yolu>")
        sys.exit(1)
    
    test_predict_endpoint(image_path) 