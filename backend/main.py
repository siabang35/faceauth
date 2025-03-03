from fastapi import FastAPI, UploadFile, Form, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from sqlalchemy.orm import Session
import cv2
import numpy as np
import io

from backend.auth import register_user, login_user, get_user_by_name
from backend.dependencies import get_db
from backend.models import User

app = FastAPI()

# Add CORS middleware to allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/register/")
async def register(
    name: str = Form(...),
    password: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar yang valid")

    return register_user(name, password, image, db)

@app.post("/login/")
async def login(
    name: str = Form(...),
    password: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar yang valid")

    return login_user(name, password, image, db)

@app.get("/users/")
async def get_all_users(db: Session = Depends(get_db)):
    """
    Mendapatkan daftar semua pengguna yang terdaftar.
    Hanya mengembalikan informasi dasar (id, nama) tanpa password atau embedding wajah.
    """
    users = db.query(User).all()
    return [{"id": user.id, "name": user.name} for user in users]

@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """
    Mendapatkan informasi pengguna berdasarkan ID.
    Hanya mengembalikan informasi dasar tanpa password atau embedding wajah.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Pengguna tidak ditemukan")
    
    return {"id": user.id, "name": user.name}

@app.delete("/users/{user_id}")
async def delete_user(user_id: int, db: Session = Depends(get_db)):
    """
    Menghapus pengguna berdasarkan ID.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Pengguna tidak ditemukan")
    
    db.delete(user)
    db.commit()
    
    return {"message": f"Pengguna {user.name} berhasil dihapus"}

@app.post("/verify-face/")
async def verify_face(
    name: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Endpoint untuk memverifikasi wajah tanpa login penuh.
    Berguna untuk testing kualitas pengenalan wajah.
    """
    try:
        image = Image.open(file.file).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="File yang diunggah bukan gambar yang valid")
    
    user = get_user_by_name(name, db)
    if not user:
        raise HTTPException(status_code=404, detail="Pengguna tidak ditemukan")
    
    # Import di sini untuk menghindari circular import
    from backend.face_recognition import extract_embedding, compare_embeddings
    import json
    import numpy as np
    
    try:
        # Ekstrak embedding dari gambar yang diunggah
        input_embedding = extract_embedding(image)
        
        # Ambil embedding yang tersimpan
        if isinstance(user.face_embedding, memoryview):
            face_embedding_str = user.face_embedding.tobytes().decode('utf-8')
        else:
            face_embedding_str = user.face_embedding
            
        stored_embedding = np.array(json.loads(face_embedding_str), dtype=np.float32)
        
        # Bandingkan embeddings
        is_match, similarity = compare_embeddings(stored_embedding, input_embedding, threshold=0.7)
        
        return {
            "match": is_match,
            "similarity": float(similarity),
            "distance": float(1.0 - similarity)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saat memverifikasi wajah: {str(e)}")