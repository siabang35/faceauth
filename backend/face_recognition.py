import numpy as np 
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
import logging
import io

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Cek device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model FaceNet
try:
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    logger.info("FaceNet model loaded successfully")
except Exception as e:
    logger.error(f"Error loading FaceNet model: {str(e)}")
    raise

# Load face detector
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Failed to load Haar cascade classifier")
    logger.info("Face detector loaded successfully")
except Exception as e:
    logger.error(f"Error loading face detector: {str(e)}")
    raise

def evaluate_face_position(frame, faces):
    """
    Evaluasi apakah posisi wajah sudah baik untuk recognition.
    
    Args:
        frame: Frame gambar dari kamera
        faces: Array hasil deteksi wajah dari cascade classifier
        
    Returns:
        tuple: (face_position_good, face_details)
    """
    if len(faces) == 0:
        return False, {}
    
    # Ambil wajah pertama saja jika ada banyak
    x, y, w, h = faces[0]
    
    # Ukuran frame
    frame_height, frame_width = frame.shape[:2]
    
    # Hitung metrik posisi wajah
    face_center_x, face_center_y = x + w//2, y + h//2
    frame_center_x, frame_center_y = frame_width//2, frame_height//2
    
    # Jarak dari pusat
    distance_from_center = np.sqrt((face_center_x - frame_center_x)**2 + 
                                   (face_center_y - frame_center_y)**2)
    center_threshold = min(frame_width, frame_height) * 0.15
    is_centered = distance_from_center < center_threshold
    
    # Ukuran wajah
    face_size_ratio = (w * h) / (frame_width * frame_height)
    min_size_ratio = 0.05  # Minimal 5% dari frame
    max_size_ratio = 0.5   # Maksimal 50% dari frame
    is_good_size = min_size_ratio < face_size_ratio < max_size_ratio
    
    # Cek apakah wajah terlalu dekat dengan tepi frame
    margin = 20  # piksel
    is_not_at_edge = (x > margin and y > margin and 
                      x + w < frame_width - margin and 
                      y + h < frame_height - margin)
    
    # Cek apakah wajah menghadap ke depan (estimasi sederhana)
    # Kita bisa menggunakan rasio lebar dan tinggi sebagai estimasi kasar
    face_ratio = w / h
    is_frontal = 0.7 < face_ratio < 1.3  # Wajah manusia biasanya memiliki rasio sekitar 0.8-1.2
    
    # Gabungkan semua kriteria
    face_position_good = is_centered and is_good_size and is_not_at_edge and is_frontal
    
    # Simpan detail untuk logging
    face_details = {
        "centered": is_centered,
        "good_size": is_good_size,
        "not_at_edge": is_not_at_edge,
        "frontal": is_frontal,
        "distance": distance_from_center,
        "size_ratio": face_size_ratio * 100,  # Konversi ke persentase
        "face_ratio": face_ratio
    }
    
    return face_position_good, face_details

def detect_and_align_face(image):
    """
    Deteksi wajah dan lakukan alignment sederhana.
    
    Args:
        image: PIL Image atau numpy array
        
    Returns:
        PIL Image wajah yang sudah di-crop dan align, atau None jika tidak ada wajah
    """
    # Konversi ke numpy array jika input adalah PIL Image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        # Konversi RGB ke BGR jika perlu (OpenCV menggunakan BGR)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_array = image
    
    # Deteksi wajah
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        logger.warning("No face detected in the image")
        return None
    
    # Ambil wajah terbesar (asumsi ini adalah wajah utama)
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Crop wajah dengan margin
    margin_percent = 0.2
    margin_x = int(w * margin_percent)
    margin_y = int(h * margin_percent)
    
    # Pastikan koordinat tidak keluar dari gambar
    height, width = img_array.shape[:2]
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(width, x + w + margin_x)
    y2 = min(height, y + h + margin_y)
    
    face_img = img_array[y1:y2, x1:x2]
    
    # Konversi kembali ke RGB untuk PIL
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    
    # Konversi ke PIL Image
    pil_face = Image.fromarray(face_img_rgb)
    
    return pil_face

def capture_face_with_detection():
    """
    Buka kamera, deteksi wajah, tampilkan kotak wajah, dan capture wajah yang terdeteksi.
    """
    logger.info("Starting face capture with detection")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Camera not available")
        raise Exception("Kamera tidak tersedia")

    # Atur resolusi kamera
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        logger.error("Failed to load Haar cascade classifier")
        cap.release()
        raise Exception("Gagal memuat classifier wajah")

    face_image = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

        # Evaluasi posisi wajah
        face_position_good, face_details = evaluate_face_position(frame, faces)
        
        # Gambar area target di tengah (area ideal untuk wajah)
        frame_height, frame_width = frame.shape[:2]
        target_size = min(frame_width, frame_height) * 0.5
        target_x = int(frame_width/2 - target_size/2)
        target_y = int(frame_height/2 - target_size/2)
        
        # Gambar kotak target dengan garis putus-putus
        for i in range(0, 360, 10):  # Setiap 10 derajat
            x1 = int(target_x + target_size/2 + target_size/2 * np.cos(np.radians(i)))
            y1 = int(target_y + target_size/2 + target_size/2 * np.sin(np.radians(i)))
            x2 = int(target_x + target_size/2 + target_size/2 * np.cos(np.radians(i+5)))
            y2 = int(target_y + target_size/2 + target_size/2 * np.sin(np.radians(i+5)))
            cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
        
        for (x, y, w, h) in faces:
            # Warna kotak: hijau jika posisi baik, merah jika tidak
            color = (0, 255, 0) if face_position_good else (0, 0, 255)
            
            # Gambar kotak di sekitar wajah
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
            
            # Tambahkan teks status
            status_text = "Posisi Baik" if face_position_good else "Posisi Wajah Belum Tepat"
            cv2.putText(frame, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Gambar titik tengah wajah
            face_center_x, face_center_y = x + w//2, y + h//2
            cv2.circle(frame, (face_center_x, face_center_y), 5, (255, 0, 255), -1)
            
            # Gambar garis dari tengah wajah ke tengah frame
            frame_center_x, frame_center_y = frame_width//2, frame_height//2
            cv2.line(frame, (face_center_x, face_center_y), 
                    (frame_center_x, frame_center_y), (255, 0, 255), 2)

        # Tambahkan instruksi
        cv2.putText(frame, "Tekan 'c' untuk Capture, 'q' untuk Keluar", 
                   (10, frame_height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Deteksi Wajah", frame)

        key = cv2.waitKey(1)

        if key == ord('c') and len(faces) > 0:
            if face_position_good:
                x, y, w, h = faces[0]
                # Tambahkan margin untuk hasil yang lebih baik
                margin = int(w * 0.2)  # 20% margin
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame_width, x + w + margin)
                y2 = min(frame_height, y + h + margin)
                
                face_image = frame[y1:y2, x1:x2]
                logger.info("Face captured successfully")
                break
            else:
                logger.warning("Attempted to capture face but position not good")
                # Tampilkan pesan bahwa posisi wajah belum baik
                info_frame = frame.copy()
                cv2.putText(info_frame, "Posisi wajah belum tepat! Coba lagi.", 
                           (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Deteksi Wajah", info_frame)
                cv2.waitKey(1000)  # Tampilkan pesan selama 1 detik
        elif key == ord('q'):
            logger.info("Face capture canceled by user")
            break

    cap.release()
    cv2.destroyAllWindows()

    if face_image is not None:
        return face_image
    else:
        logger.warning("No face was captured")
        raise Exception("Tidak ada wajah yang terdeteksi atau posisi wajah tidak tepat")

def preprocess_face_image(image):
    """
    Pra-proses gambar wajah sebelum ekstraksi fitur.
    
    Args:
        image: PIL Image wajah
        
    Returns:
        PIL Image yang sudah diproses
    """
    # Deteksi dan align wajah jika belum dilakukan
    aligned_face = detect_and_align_face(image)
    if aligned_face is not None:
        image = aligned_face
    
    # Pastikan gambar dalam mode RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize ke ukuran yang diharapkan model
    image = image.resize((160, 160))
    
    return image

def extract_embedding(image):
    """
    Ekstrak face embedding dari gambar wajah.
    
    Args:
        image: PIL Image wajah
        
    Returns:
        numpy array: Face embedding vector
    """
    logger.info("Extracting face embedding")
    
    try:
        # Pra-proses gambar
        image = preprocess_face_image(image)
        
        # Transformasi untuk model
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Konversi ke tensor dan normalisasi
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Ekstrak embedding
        with torch.no_grad():
            embedding = model(image_tensor).cpu().numpy().flatten()
        
        logger.info(f"Embedding extracted successfully, shape: {embedding.shape}")
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting face embedding: {str(e)}")
        raise Exception(f"Gagal mengekstrak embedding wajah: {str(e)}")

def compare_embeddings(embedding1, embedding2, threshold=0.7):
    """
    Bandingkan dua embedding wajah dan tentukan apakah dari orang yang sama.
    
    Args:
        embedding1: Embedding wajah pertama
        embedding2: Embedding wajah kedua
        threshold: Threshold untuk menentukan kecocokan (default: 0.7)
        
    Returns:
        tuple: (is_match, similarity_score)
    """
    # Hitung cosine similarity
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    
    # Konversi ke jarak (distance = 1 - similarity)
    distance = 1.0 - similarity
    
    # Tentukan apakah match berdasarkan threshold
    is_match = distance < threshold
    
    logger.info(f"Face comparison: distance={distance:.4f}, threshold={threshold}, is_match={is_match}")
    
    return is_match, similarity

def process_image_for_recognition(image_data):
    """
    Proses gambar untuk pengenalan wajah dari berbagai sumber.
    
    Args:
        image_data: Bisa berupa PIL Image, numpy array, atau bytes
        
    Returns:
        PIL Image yang sudah diproses
    """
    # Jika input adalah bytes, konversi ke PIL Image
    if isinstance(image_data, bytes):
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            logger.error(f"Error opening image from bytes: {str(e)}")
            raise
    # Jika input adalah numpy array, konversi ke PIL Image
    elif isinstance(image_data, np.ndarray):
        # Cek apakah array dalam format BGR (OpenCV)
        if len(image_data.shape) == 3 and image_data.shape[2] == 3:
            image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image_data)
    # Jika input sudah PIL Image, gunakan langsung
    elif isinstance(image_data, Image.Image):
        image = image_data
    else:
        logger.error(f"Unsupported image data type: {type(image_data)}")
        raise ValueError("Format gambar tidak didukung")
    
    # Deteksi dan crop wajah
    face_image = detect_and_align_face(image)
    if face_image is None:
        logger.warning("No face detected in the image")
        # Jika tidak ada wajah terdeteksi, gunakan gambar asli
        return image
    
    return face_image