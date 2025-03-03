import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Gagal membuka kamera")
else:
    print("Kamera berhasil dibuka")
    
cap.release()
