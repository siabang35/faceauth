import streamlit as st
import cv2
import numpy as np
import requests
from PIL import Image
import tempfile
import os
import time
import io
import json

# URL Backend - sesuaikan
BACKEND_URL = "http://localhost:8000"

# Set page config
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🔒",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .status-box-success {
        background-color: rgba(0, 255, 0, 0.1);
        border: 1px solid rgba(0, 255, 0, 0.5);
    }
    .status-box-warning {
        background-color: rgba(255, 165, 0, 0.1);
        border: 1px solid rgba(255, 165, 0, 0.5);
    }
    .status-box-error {
        background-color: rgba(255, 0, 0, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.5);
    }
    .instruction-text {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: 600;
    }
    .user-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    .delete-button {
        color: white;
        background-color: #EF4444;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 0.25rem;
        cursor: pointer;
    }
    .delete-button:hover {
        background-color: #DC2626;
    }
    .tab-content {
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>Face Recognition System</h1>", unsafe_allow_html=True)
st.markdown("<p class='instruction-text'>Sistem pengenalan wajah untuk autentikasi dengan deteksi wajah real-time</p>", unsafe_allow_html=True)

# Initialize session state
if "token" not in st.session_state:
    st.session_state["token"] = None
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "user_name" not in st.session_state:
    st.session_state["user_name"] = None
if "captured_image" not in st.session_state:
    st.session_state["captured_image"] = None
if "detection_status" not in st.session_state:
    st.session_state["detection_status"] = {"message": "", "type": ""}
if "camera_is_open" not in st.session_state:
    st.session_state["camera_is_open"] = False
if "debug_info" not in st.session_state:
    st.session_state["debug_info"] = ""
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "auth"
if "faces" not in st.session_state:
    st.session_state["faces"] = []
if "uploaded_image" not in st.session_state:
    st.session_state["uploaded_image"] = None

# Sidebar for mode selection and user inputs
with st.sidebar:
    st.markdown("<h2 class='sub-header'>Navigasi</h2>", unsafe_allow_html=True)
    
    # Navigation tabs
    tabs = ["Autentikasi", "Data Pengguna", "Tentang"]
    tab_icons = ["🔐", "👥", "ℹ️"]
    
    selected_tab = st.radio(
        "Pilih Menu",
        [f"{icon} {tab}" for icon, tab in zip(tab_icons, tabs)]
    )
    
    # Extract the tab name without icon
    selected_tab_name = selected_tab.split(" ", 1)[1]
    
    # Set active tab in session state
    if selected_tab_name == "Autentikasi":
        st.session_state["active_tab"] = "auth"
    elif selected_tab_name == "Data Pengguna":
        st.session_state["active_tab"] = "users"
    else:
        st.session_state["active_tab"] = "about"
    
    # Show login/logout status
    st.markdown("---")
    if st.session_state["token"]:
        st.success(f"Logged in as: {st.session_state['user_name']}")
        if st.button("Logout"):
            st.session_state["token"] = None
            st.session_state["user_id"] = None
            st.session_state["user_name"] = None
            st.rerun()
    else:
        st.warning("Not logged in")
    
    # Informasi tambahan
    st.markdown("### Informasi")
    st.info("""
    - Register: Mendaftarkan wajah baru
    - Login: Masuk dengan wajah yang sudah terdaftar
    - Pastikan wajah terdeteksi dengan baik (kotak hijau)
    - Pencahayaan yang baik akan meningkatkan akurasi
    """)

# =========================
# FACE DETECTION FUNCTION
# =========================
def detect_and_draw_face(frame):
    # Buat salinan frame untuk dimodifikasi
    display_frame = frame.copy()
    
    # Deteksi wajah menggunakan Haar Cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
    
    # Store faces in session state for later use
    st.session_state["faces"] = faces
    
    face_detected = len(faces) > 0
    face_position_good = False
    face_details = {}
    
    # Ukuran frame
    frame_height, frame_width = frame.shape[:2]
    
    # Gambar area target di tengah (area ideal untuk wajah)
    target_size = min(frame_width, frame_height) * 0.5
    target_x = int(frame_width/2 - target_size/2)
    target_y = int(frame_height/2 - target_size/2)
    
    # Gambar kotak target dengan garis putus-putus
    for i in range(0, 360, 10):  # Setiap 10 derajat
        x1 = int(target_x + target_size/2 + target_size/2 * np.cos(np.radians(i)))
        y1 = int(target_y + target_size/2 + target_size/2 * np.sin(np.radians(i)))
        x2 = int(target_x + target_size/2 + target_size/2 * np.cos(np.radians(i+5)))
        y2 = int(target_y + target_size/2 + target_size/2 * np.sin(np.radians(i+5)))
        cv2.line(display_frame, (x1, y1), (x2, y2), (255, 255, 255), 1)
    
    # Tambahkan teks panduan
    if not face_detected:
        cv2.putText(display_frame, "Tidak ada wajah terdeteksi", 
                   (int(frame_width/2) - 150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    for (x, y, w, h) in faces:
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
        
        # Simpan detail untuk ditampilkan
        face_details = {
            "centered": is_centered,
            "good_size": is_good_size,
            "not_at_edge": is_not_at_edge,
            "frontal": is_frontal,
            "distance": distance_from_center,
            "size_ratio": face_size_ratio * 100,  # Konversi ke persentase
            "face_ratio": face_ratio
        }
        
        # Warna kotak: hijau jika posisi baik, merah jika tidak
        color = (0, 255, 0) if face_position_good else (0, 0, 255)
        
        # Gambar kotak di sekitar wajah dengan ketebalan yang lebih besar
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 3)
        
        # Tambahkan teks status di atas kotak wajah
        status_text = "✓ Posisi Baik" if face_position_good else "✗ Posisi Belum Tepat"
        cv2.putText(display_frame, status_text, 
                   (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Gambar titik tengah wajah
        cv2.circle(display_frame, (face_center_x, face_center_y), 5, (255, 0, 255), -1)
        
        # Gambar garis dari tengah wajah ke tengah frame
        cv2.line(display_frame, (face_center_x, face_center_y), 
                (frame_center_x, frame_center_y), (255, 0, 255), 2)
        
        # Tambahkan detail metrik di pojok kiri atas
        metrics_text = [
            f"Jarak dari pusat: {distance_from_center:.1f}px",
            f"Ukuran wajah: {face_size_ratio*100:.1f}% frame",
            f"Rasio wajah (w/h): {face_ratio:.2f}"
        ]
        
        for i, text in enumerate(metrics_text):
            cv2.putText(display_frame, text, 
                       (10, 60 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return display_frame, frame, faces, face_position_good, face_details


# =========================
# BUKA KAMERA STREAMING (Realtime Lancar)
# =========================
def open_camera():
    # Placeholder untuk video stream
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<h2 class='sub-header'>Kamera Live</h2>", unsafe_allow_html=True)
        video_placeholder = st.empty()
    
    with col2:
        st.markdown("<h2 class='sub-header'>Status Deteksi</h2>", unsafe_allow_html=True)
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        st.markdown("<h3 class='sub-header'>Kontrol</h3>", unsafe_allow_html=True)
        capture_button_placeholder = st.empty()
        close_button_placeholder = st.empty()
    
    cap = cv2.VideoCapture(0)
    
    # Atur resolusi agar lebih tajam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Simpan ukuran frame ke variabel
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    captured_frame = None
    stop_camera = False
    
    st.session_state["camera_is_open"] = True
    
    while not stop_camera and st.session_state.get("camera_is_open", False):
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("Kamera gagal dibuka.")
            break
        
        # Deteksi wajah & gambar kotaknya
        display_frame, original_frame, faces, face_position_good, face_details = detect_and_draw_face(frame)
        
        # Konversi ke RGB agar cocok dengan Streamlit
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        
        # Tampilkan frame realtime di Streamlit
        video_placeholder.image(display_frame_rgb, channels="RGB")
        
        # Update status
        if len(faces) == 0:
            status_html = """
            <div class='status-box status-box-error'>
                <h3>⚠️ Tidak ada wajah terdeteksi</h3>
                <p>Pastikan wajah Anda terlihat di kamera</p>
            </div>
            """
            status_placeholder.markdown(status_html, unsafe_allow_html=True)
            metrics_placeholder.empty()
        elif not face_position_good:
            status_html = """
            <div class='status-box status-box-warning'>
                <h3>⚠️ Posisi wajah belum tepat</h3>
                <p>Ikuti petunjuk di bawah untuk memperbaiki posisi</p>
            </div>
            """
            status_placeholder.markdown(status_html, unsafe_allow_html=True)
            
            # Tampilkan metrik dan saran
            if face_details:
                metrics_text = ""
                if not face_details["centered"]:
                    metrics_text += "- ❌ Wajah tidak di tengah\n"
                else:
                    metrics_text += "- ✅ Wajah di tengah\n"
                    
                if not face_details["good_size"]:
                    if face_details["size_ratio"] < 5:
                        metrics_text += "- ❌ Wajah terlalu kecil\n"
                    else:
                        metrics_text += "- ❌ Wajah terlalu besar\n"
                else:
                    metrics_text += "- ✅ Ukuran wajah baik\n"
                    
                if not face_details["not_at_edge"]:
                    metrics_text += "- ❌ Wajah terlalu dekat dengan tepi\n"
                else:
                    metrics_text += "- ✅ Posisi dalam frame baik\n"
                    
                if not face_details["frontal"]:
                    metrics_text += "- ❌ Wajah tidak menghadap ke depan\n"
                else:
                    metrics_text += "- ✅ Orientasi wajah baik\n"
                
                metrics_placeholder.markdown(metrics_text)
        else:
            status_html = """
            <div class='status-box status-box-success'>
                <h3>✅ Wajah terdeteksi dengan baik!</h3>
                <p>Posisi sudah tepat, siap untuk capture</p>
            </div>
            """
            status_placeholder.markdown(status_html, unsafe_allow_html=True)
            metrics_placeholder.empty()
        
        # Tombol Capture - hanya aktif jika wajah terdeteksi dengan baik
        capture_button = capture_button_placeholder.button(
            "📸 Capture Foto", 
            disabled=(len(faces) == 0 or not face_position_good),
            key=f"capture_{time.time()}"
        )
        
        # Tombol Tutup Kamera
        close_button = close_button_placeholder.button("❌ Tutup Kamera", key=f"close_{time.time()}")

        if capture_button and len(faces) > 0 and face_position_good:
            # Simpan frame asli (bukan hasil preview dengan anotasi)
            captured_frame = original_frame.copy()
            
            # Simpan juga wajah yang terdeteksi untuk ditampilkan
            x, y, w, h = faces[0]
            # Tambahkan margin untuk hasil yang lebih baik
            margin = int(w * 0.2)  # 20% margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame_width, x + w + margin)
            y2 = min(frame_height, y + h + margin)
            
            face_img = original_frame[y1:y2, x1:x2]
            
            st.session_state["detection_status"] = {
                "message": "Wajah berhasil di-capture dengan kualitas baik!",
                "type": "success"
            }
            stop_camera = True
        
        if close_button:
            st.session_state["detection_status"] = {
                "message": "Kamera ditutup tanpa capture wajah.",
                "type": "info"
            }
            stop_camera = True
        
        # Tambahkan delay kecil untuk mengurangi beban CPU
        time.sleep(0.03)
    
    cap.release()
    st.session_state["camera_is_open"] = False
    return captured_frame

# =========================
# KIRIM KE BACKEND
# =========================
def send_to_backend(image, mode, name, password):
    with st.spinner(f"Memproses {mode}..."):
        try:
            # Debug info
            debug_msg = f"Sending to {BACKEND_URL}/{'register' if mode == 'Register' else 'login'}/\n"
            debug_msg += f"Image shape: {image.shape}\n"
            debug_msg += f"Image type: {type(image)}\n"
            st.session_state["debug_info"] = debug_msg
            
            # Convert image to bytes
            is_success, buffer = cv2.imencode(".jpg", image)
            if not is_success:
                st.error("Gagal mengkonversi gambar ke format JPEG")
                st.session_state["debug_info"] += "Failed to encode image to JPEG\n"
                return False
            
            # Create file-like object
            io_buf = io.BytesIO(buffer)
            
            # Prepare multipart form data
            files = {"file": ("image.jpg", io_buf, "image/jpeg")}
            data = {"name": name, "password": password}
            
            # Send request
            if mode == "Register":
                response = requests.post(f"{BACKEND_URL}/register/", files=files, data=data)
            else:
                response = requests.post(f"{BACKEND_URL}/login/", files=files, data=data)
                
            # Process response
            if response.status_code == 200:
                result = response.json()
                if mode == "Login" and "token" in result:
                    st.session_state["token"] = result["token"]
                    st.session_state["user_id"] = result.get("user_id")
                    st.session_state["user_name"] = result.get("name")
                    
                    st.success(f"Login berhasil! Selamat datang, {name}.")
                    st.session_state["detection_status"] = {
                        "message": f"Login berhasil! Selamat datang, {name}.",
                        "type": "success"
                    }
                else:
                    st.success(f"Sukses: {result.get('message', 'Operasi berhasil')}")
                    st.session_state["detection_status"] = {
                        "message": f"Sukses: {result.get('message', 'Operasi berhasil')}",
                        "type": "success"
                    }
                return True
            else:
                try:
                    error_detail = response.json().get("detail", response.text)
                except:
                    error_detail = response.text
                    
                st.error(f"Gagal: {error_detail}")
                st.session_state["debug_info"] += f"\nError: {error_detail}"
                st.session_state["detection_status"] = {
                    "message": f"Gagal: {error_detail}",
                    "type": "error"
                }
                return False
                
        except requests.exceptions.ConnectionError:
            st.error(f"Tidak dapat terhubung ke server backend di {BACKEND_URL}. Pastikan server berjalan.")
            st.session_state["debug_info"] += f"\nConnection Error: Cannot connect to {BACKEND_URL}"
            return False
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            st.session_state["debug_info"] += f"\nException: {str(e)}"
            return False

# =========================
# GET USERS FROM BACKEND
# =========================
def get_users():
    try:
        response = requests.get(f"{BACKEND_URL}/users/")
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Gagal mengambil data pengguna: {response.text}")
            return []
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengambil data pengguna: {str(e)}")
        return []

# =========================
# DELETE USER
# =========================
def delete_user(user_id):
    try:
        response = requests.delete(f"{BACKEND_URL}/users/{user_id}")
        if response.status_code == 200:
            st.success(f"Pengguna berhasil dihapus")
            return True
        else:
            st.error(f"Gagal menghapus pengguna: {response.text}")
            return False
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menghapus pengguna: {str(e)}")
        return False

# =========================
# TAMPILKAN STATUS DETEKSI
# =========================
if st.session_state["detection_status"]["message"]:
    status_type = st.session_state["detection_status"]["type"]
    if status_type == "success":
        st.success(st.session_state["detection_status"]["message"])
    elif status_type == "warning":
        st.warning(st.session_state["detection_status"]["message"])
    elif status_type == "error":
        st.error(st.session_state["detection_status"]["message"])
    else:
        st.info(st.session_state["detection_status"]["message"])

# Debug info if available
if st.session_state["debug_info"] and st.checkbox("Show Debug Info"):
    with st.expander("Debug Information"):
        st.code(st.session_state["debug_info"])       

# =========================
# AUTHENTICATION TAB
# =========================
if st.session_state["active_tab"] == "auth":
    st.markdown("<h2 class='sub-header'>Autentikasi Wajah</h2>", unsafe_allow_html=True)
    
    # Pilih mode: Register atau Login
    mode = st.radio("Pilih Mode", ["Login", "Register"])
    
    # Input nama & password
    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Nama", placeholder="Masukkan nama")
    with col2:
        password = st.text_input("Password", type="password", placeholder="Masukkan password")
    
    
    # =========================
    # TOMBOL BUKA KAMERA 
    # =========================
    camera_col1, camera_col2 = st.columns([1, 1])

    with camera_col1:
        camera_button = st.button("🎥 Buka Kamera", disabled=st.session_state.get("camera_is_open", False))
        if camera_button:
            captured = open_camera()
            if captured is not None:
                st.session_state["captured_image"] = captured

    # =========================
    # TAMPILKAN HASIL CAPTURE (Kalau Ada)
    # =========================
    if st.session_state.get("captured_image") is not None:
        with camera_col2:
            st.markdown("<h3 class='sub-header'>Hasil Capture</h3>", unsafe_allow_html=True)
            st.image(cv2.cvtColor(st.session_state.get("captured_image"), cv2.COLOR_BGR2RGB), 
                    caption="Wajah yang akan digunakan untuk autentikasi", 
                    channels="RGB", 
                    use_container_width=True)
            
            if st.button("❌ Hapus Hasil Capture"):
                st.session_state["captured_image"] = None
                st.session_state["detection_status"] = {
                    "message": "Hasil capture dihapus, silakan ambil foto baru.",
                    "type": "info"
                }
                st.rerun()
                
    # =========================
    # UPLOAD IMAGE MANUAL
    # =========================
    st.markdown("<h3 class='sub-header'>Atau Upload Foto Manual</h3>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload foto wajah (jpg/png)", type=["jpg", "png"])

    if uploaded_file is not None:
        # Baca gambar upload
        uploaded_image = np.array(Image.open(uploaded_file).convert('RGB'))
        st.session_state["uploaded_image"] = uploaded_image
        
        # Deteksi wajah pada gambar upload
        display_frame, original_frame, faces, face_position_good, face_details = detect_and_draw_face(uploaded_image)
        
        # Tampilkan gambar dengan anotasi
        st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), 
                caption="Gambar Upload dengan Deteksi Wajah", 
                channels="RGB")
        
        # Tampilkan status deteksi
        if len(faces) == 0:
            st.error("Tidak ada wajah terdeteksi pada gambar yang diupload!")
        elif not face_position_good:
            st.warning("Wajah terdeteksi, tetapi posisi tidak ideal. Sebaiknya gunakan kamera untuk hasil lebih baik.")
        else:
            st.success("Wajah terdeteksi dengan baik pada gambar yang diupload!")

    # =========================
    # SUBMIT (Dari Capture atau Upload)
    # =========================
    st.markdown("<h3 class='sub-header'>Submit</h3>", unsafe_allow_html=True)

    submit_button = st.button(f"🚀 Submit {mode}")

    if submit_button:
        if not name or not password:
            st.error("Nama dan password harus diisi.")
            st.session_state["detection_status"] = {
                "message": "Nama dan password harus diisi.",
                "type": "error"
            }
        else:
            # Check if we have an uploaded image with faces
            if st.session_state.get("uploaded_image") is not None and len(st.session_state.get("faces", [])) > 0:
                # Use the uploaded image
                if send_to_backend(st.session_state["uploaded_image"], mode, name, password):
                    # If successful, refresh the page
                    st.rerun()
            # Check if we have a captured image
            elif st.session_state.get("captured_image") is not None:
                # Use the captured image
                captured_img = st.session_state["captured_image"]
                # Make sure the image is in BGR format (OpenCV default)
                if len(captured_img.shape) == 3 and captured_img.shape[2] == 3:
                    if send_to_backend(captured_img, mode, name, password):
                        # If successful, refresh the page
                        st.rerun()
                else:
                    st.error("Format gambar tidak valid. Silakan capture ulang.")
            else:
                st.error("Harap capture wajah atau upload foto terlebih dahulu.")
                st.session_state["detection_status"] = {
                    "message": "Harap capture wajah atau upload foto terlebih dahulu.",
                    "type": "error"
                }

# =========================
# USERS TAB
# =========================
elif st.session_state["active_tab"] == "users":
    st.markdown("<h2 class='sub-header'>Data Pengguna</h2>", unsafe_allow_html=True)
    
    # Cek apakah user sudah login
    if not st.session_state["token"]:
        st.warning("Anda harus login terlebih dahulu untuk melihat data pengguna.")
    else:
        # Tombol refresh
        if st.button("🔄 Refresh Data"):
            st.rerun()
        
        # Ambil data pengguna
        users = get_users()
        
        if users:
            st.markdown(f"<p>Total pengguna terdaftar: {len(users)}</p>", unsafe_allow_html=True)
            
            # Tampilkan daftar pengguna
            for user in users:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"""
                    <div class='user-card'>
                        <h3>ID: {user['id']}</h3>
                        <p>Nama: {user['name']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    # Tombol hapus hanya untuk user sendiri atau admin
                    if st.button("🗑️ Hapus", key=f"delete_{user['id']}"):
                        if delete_user(user['id']):
                            st.rerun()
        else:
            st.info("Tidak ada data pengguna yang tersedia.")

# =========================
# ABOUT TAB
# =========================
else:
    st.markdown("<h2 class='sub-header'>Tentang Aplikasi</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Face Recognition System
    
    Aplikasi ini menggunakan teknologi pengenalan wajah untuk autentikasi pengguna dengan fitur:
    
    - **Deteksi Wajah Real-time**: Mendeteksi wajah secara langsung melalui kamera
    - **Evaluasi Posisi Wajah**: Memastikan posisi wajah optimal untuk pengenalan
    - **Ekstraksi Fitur Wajah**: Menggunakan model FaceNet untuk mengekstrak embedding wajah
    - **Autentikasi Aman**: Membandingkan wajah dengan data tersimpan untuk verifikasi
    - **Manajemen Pengguna**: Mendaftarkan, melihat, dan menghapus data pengguna
    
    #### Teknologi yang Digunakan
    
    - **Backend**: FastAPI, SQLAlchemy, PostgreSQL
    - **Computer Vision**: OpenCV, FaceNet
    - **Frontend**: Streamlit
    - **Keamanan**: Bcrypt untuk hashing password, JWT untuk token autentikasi
    
    #### Cara Penggunaan
    
    1. **Register**: Daftarkan wajah dan data pengguna baru
    2. **Login**: Masuk dengan nama, password, dan verifikasi wajah
    3. **Kelola Data**: Lihat dan hapus data pengguna
    
    #### Tips Penggunaan
    
    - Pastikan pencahayaan baik saat melakukan capture wajah
    - Posisikan wajah di tengah frame dan ikuti petunjuk posisi
    - Gunakan password yang kuat untuk keamanan lebih baik
    """)
