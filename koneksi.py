from sqlalchemy import create_engine, text

# Konfigurasi database PostgreSQL
DATABASE_URL = "postgresql://postgres:wildan123@localhost/facerecognitiondb"

# Buat engine
engine = create_engine(DATABASE_URL)

# Tes koneksi
try:
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1"))  # Tambahkan 'text()'
        print("✅ Koneksi ke database berhasil:", result.fetchone())
except Exception as e:
    print("❌ Gagal terhubung ke database:", e)
