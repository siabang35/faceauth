from sqlalchemy.orm import Session
from jose import jwt
import bcrypt
import json
import numpy as np
from fastapi import HTTPException
from PIL import Image

from backend.models import User
from backend.face_recognition import extract_embedding, compare_embeddings
from backend.config import SECRET_KEY, ALGORITHM


def register_user(name: str, password: str, image: Image.Image, db: Session):
    if len(name) < 3 or len(password) < 6:
        raise HTTPException(status_code=400, detail="Nama minimal 3 karakter dan password minimal 6 karakter")

    # Cek apakah username sudah dipakai
    if db.query(User).filter(User.name == name).first():
        raise HTTPException(status_code=400, detail="Username sudah digunakan")

    try:
        # Ekstraksi fitur wajah
        embedding = extract_embedding(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengekstrak wajah: {str(e)}")

    # Hash password
    password_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    # Simpan embedding sebagai JSON string agar kompatibel dengan database
    embedding_json = json.dumps(embedding.tolist())

    new_user = User(
        name=name,
        password_hash=password_hash,
        face_embedding=embedding_json
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "Registrasi berhasil", "user_id": new_user.id}


def login_user(name: str, password: str, image: Image.Image, db: Session):
    user = db.query(User).filter(User.name == name).first()
    if not user:
        raise HTTPException(status_code=401, detail="User tidak ditemukan")

    # Verifikasi password
    if not bcrypt.checkpw(password.encode(), user.password_hash.encode()):
        raise HTTPException(status_code=401, detail="Password salah")

    try:
        # Ekstraksi fitur wajah dari gambar login
        input_embedding = extract_embedding(image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengekstrak wajah: {str(e)}")

    # Ambil face embedding yang disimpan di database
    try:
        # Pastikan kalau face_embedding disimpan dalam format string JSON, jadi langsung load.
        if isinstance(user.face_embedding, memoryview):
            face_embedding_str = user.face_embedding.tobytes().decode('utf-8')
        else:
            face_embedding_str = user.face_embedding

        stored_embedding = np.array(json.loads(face_embedding_str), dtype=np.float32)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error membaca data wajah: {str(e)}")

    # Bandingkan embeddings
    is_match, similarity = compare_embeddings(stored_embedding, input_embedding, threshold=0.7)
    
    # Jika wajah tidak cocok
    if not is_match:
        raise HTTPException(status_code=401, detail=f"Wajah tidak cocok (similarity: {similarity:.4f})")

    # Buat token JWT
    token = jwt.encode({"sub": user.name, "id": user.id}, SECRET_KEY, algorithm=ALGORITHM)

    return {"token": token, "user_id": user.id, "name": user.name}

def get_user_by_name(name: str, db: Session):
    """
    Mendapatkan user berdasarkan nama.
    """
    return db.query(User).filter(User.name == name).first()

def get_user_by_id(user_id: int, db: Session):
    """
    Mendapatkan user berdasarkan ID.
    """
    return db.query(User).filter(User.id == user_id).first()

def delete_user(user_id: int, db: Session):
    """
    Menghapus user berdasarkan ID.
    """
    user = get_user_by_id(user_id, db)
    if not user:
        raise HTTPException(status_code=404, detail="User tidak ditemukan")
    
    db.delete(user)
    db.commit()
    
    return {"message": f"User {user.name} berhasil dihapus"}

def get_all_users(db: Session):
    """
    Mendapatkan semua user.
    """
    users = db.query(User).all()
    return users