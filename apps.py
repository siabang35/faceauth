from fastapi import FastAPI, Form, Depends, HTTPException
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session
import numpy as np
import bcrypt
from jose import JWTError, jwt
import cv2
import json
from PIL import Image
import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1

# Konfigurasi database PostgreSQL
DATABASE_URL = "postgresql://postgres:wildan123@localhost/facerecognitiondb"
app = FastAPI()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

SECRET_KEY = "supersecretkey"
ALGORITHM = "HS256"

# Model database untuk User
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    face_embedding = Column(String, nullable=False)  # Simpan sebagai JSON string

Base.metadata.create_all(bind=engine)

# Dependency untuk session database
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Inisialisasi model FaceNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# Fungsi menangkap gambar tanpa GUI
def capture_image():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="Kamera tidak tersedia")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise HTTPException(status_code=500, detail="Gagal menangkap gambar")

    return frame  # Langsung return frame tanpa menampilkan GUI

# Fungsi mendapatkan embedding wajah
def extract_embedding(image):
    try:
        transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model(image).cpu().numpy().flatten()

        return embedding.tolist()  # Pastikan dikonversi ke list sebelum disimpan

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error dalam ekstraksi embedding: {str(e)}")

# Endpoint registrasi
@app.post("/register/")
def register(name: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    try:
        if len(name) < 3 or len(password) < 6:
            raise HTTPException(status_code=400, detail="Nama minimal 3 karakter dan password minimal 6 karakter.")

        if db.query(User).filter(User.name == name).first():
            raise HTTPException(status_code=400, detail="Username sudah digunakan.")

        frame = capture_image()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        embedding = extract_embedding(image)

        password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        new_user = User(name=name, password_hash=password_hash, face_embedding=json.dumps(embedding))

        db.add(new_user)
        db.commit()
        return {"message": "Registrasi berhasil"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint login
@app.post("/login/")
def login(name: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    try:
        # 🔍 1. Cari user di database
        user = db.query(User).filter(User.name == name).first()
        if not user:
            raise HTTPException(status_code=401, detail="User tidak ditemukan")

        # 🔍 2. Verifikasi password
        if not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
            raise HTTPException(status_code=401, detail="Password salah")

        # 🔍 3. Ambil gambar wajah dari kamera
        frame = capture_image()
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_embedding = extract_embedding(image)
        input_encoding = np.array(input_embedding, dtype=np.float32)

        # 🔥 4. FIX: Perbaiki cara membaca embedding dari database
        try:
            stored_embedding_list = json.loads(user.face_embedding)  # Pastikan format JSON valid
            db_encoding = np.array(stored_embedding_list, dtype=np.float32)  # Konversi ke NumPy array
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error membaca embedding wajah: {e}")

        # 🔍 5. Bandingkan wajah dengan Euclidean Distance
        similarity = np.linalg.norm(db_encoding - input_encoding)

        print(f"Similarity Score: {similarity}")  # Debugging

        if similarity > 0.5:  # Jika di atas threshold, dianggap tidak cocok
            raise HTTPException(status_code=401, detail="Wajah tidak dikenali")

        # 🔥 6. Jika wajah cocok, buat token JWT
        token = jwt.encode({"sub": user.name}, SECRET_KEY, algorithm=ALGORITHM)
        return {"token": token}

    except Exception as e:
        print(f"Error saat login: {e}")  # Debugging
        raise HTTPException(status_code=500, detail=str(e))




# Endpoint mendapatkan info user berdasarkan token
@app.get("/me/")
def me(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"message": f"Selamat datang {payload['sub']}"}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token tidak valid")
