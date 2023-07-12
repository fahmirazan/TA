import os
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas
import firebase_admin
from firebase_admin import credentials, storage
from datetime import datetime

firebase_app = None
if not firebase_admin._apps:
    cred = credentials.Certificate("tabangundatar-d2050-firebase-adminsdk-l0l6z-f91317de63.json")
    firebase_app = firebase_admin.initialize_app(cred, {'storageBucket': 'tabangundatar-d2050.appspot.com'})
else:
    firebase_app = firebase_admin.get_app()

bucket = storage.bucket(app=firebase_app)

# Daftar file model .h5
model_files = {
    'Model 1 Adam': 'model1_adam.h5',
    'Model 2 SGD': 'model2_sgd.h5',
    'Model 3 Rmsprop': 'model3_rmsprop.h5'
}

selected_model = 'Model 1 Adam'

def load_selected_model():
    # Memuat model yang dipilih
    model_file = model_files[selected_model]
    model = load_model(model_file)
    return model

class_names = ['jajargenjang', 'lingkaran', 'segiempat', 'segitiga', 'trapesium'] 

def preprocess_image(image):
    # Convert gambar ke RGB
    image_rgb = image.convert("RGB")

    # Konversi gambar menjadi array numpy
    img_array = np.array(image_rgb)

    # Preprocessing gambar
    processed_image = img_array / 255.0

    # Tambahkan dimensi batch ke gambar
    input_image = np.expand_dims(processed_image, axis=0)

    return input_image

def predict_image(image, model):
    # Preprocess gambar
    input_image = preprocess_image(image)

    # Prediksi kelas gambar menggunakan model
    predictions = model.predict(input_image)
    predicted_class = np.argmax(predictions[0])

    return class_names[predicted_class]

def save_to_firebase(image_pil, model_name, class_name):
    # Membuat direktori temp jika belum ada
    os.makedirs("temp", exist_ok=True)

    # Membuat nama file dengan format: model_bangundatar_datetime.png
    current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{model_name}_{class_name}_{current_datetime}.png"

    # Mendapatkan nama folder model berdasarkan model_name
    model_folder = model_name.replace(" ", "_")

    # Mendapatkan nama folder klasifikasi berdasarkan class_name
    class_folder = class_name.replace(" ", "_")

    # Menyimpan gambar sebagai file sementara
    temp_file_path = f"temp/{file_name}"
    image_pil.save(temp_file_path)

    # Menyimpan file ke Firebase Storage di folder yang sesuai
    blob = bucket.blob(f"{model_folder}/{class_folder}/{file_name}")
    blob.upload_from_filename(temp_file_path)
    blob.make_public()
    file_url = blob.public_url

    # Menghapus file sementara
    os.remove(temp_file_path)

    return file_name, file_url

# Fungsi untuk halaman Tentang Aplikasi
def show_about_app():
    st.title("Tentang Aplikasi")
    st.write("Aplikasi ini merupakan sebuah sistem klasifikasi gambar untuk mengenali jenis-jenis bangun datar dengan memanfaatkan model machine learning yang telah dilatih sebelumnya untuk melakukan klasifikasi gambar bangun datar. Hal ini melibatkan pemahaman tentang prinsip-prinsip dasar machine learning, pemrosesan gambar, dan pemilihan model yang tepat.")
    st.write("Pengguna dapat menggambar bangun datar di area yang disediakan dan kemudian aplikasi akan melakukan klasifikasi menggunakan model yang telah dilatih sebelumnya. Hasil klasifikasi akan ditampilkan kepada pengguna berserta dengan model yang digunakan dan jenis bangun datar yang dipilih. Selain itu, pengguna juga dapat menyimpan gambar yang digambar ke Firebase Storage untuk keperluan pengujian dan pengembangan lebih lanjut.")
    st.write("Aplikasi ini dibangun oleh Fahmi Razan Ramdani (1301194054) untuk memenuhi proyek Tugas Akhir yang di dampingi oleh dosen pembimbing yaitu Bapak Dr. PUTU HARRY GUNAWAN, S.Si., M.Si., M.Sc. dan Ibu Dra. INDWIARTI, M.Si")

# Fungsi untuk halaman Cara Penggunaan
def show_usage():
    st.title("Cara Penggunaan")
    st.write("Berikut adalah langkah-langkah penggunaan aplikasi ini:")

    st.write("1. Pilih model yang ingin digunakan dari pilihan yang tersedia.")
    st.write("2. Pilih jenis bangun datar yang akan diklasifikasikan.")
    st.write("3. Gambarlah bangun datar di area yang disediakan.")
    st.write("4. Klik tombol 'Klasifikasikan' untuk melihat hasil klasifikasi.")
    st.write("5. Hasil klasifikasi akan ditampilkan di bawahnya.")

# Fungsi untuk halaman Tentang Data
def show_about_data():
    st.title("Tentang Data")
    st.write("Data yang digunakan dalam aplikasi ini adalah dataset gambar bangun datar.")
    st.write("Jumlah dataset yang digunakan untuk membuat masing-masing model training berjumlah 1000 dataset dengan masing-masing kategori bangun datar 200 gambar")
    st.write("Dataset terdiri dari beberapa kategori bangun datar, yaitu jajargenjang, lingkaran, segiempat, segitiga, dan trapesium.")
    st.write("Setiap kategori memiliki beberapa sampel gambar untuk pelatihan dan pengujian model. Berikut beberapa sampel gambar yang telah diujikan sebelumnya terhadap model hasil training:")
    
    # Path ke folder "sampel"
    folder_path = "sampel"
    
    # Daftar file gambar di folder "sampel"
    image_files = os.listdir(folder_path)
    
    # Menghitung jumlah baris yang dibutuhkan
    num_rows = (len(image_files) + 1) // 2
    
    # Menampilkan gambar-gambar ke dalam halaman dengan layout grid
    for i in range(num_rows):
        row_images = image_files[i*2 : (i+1)*2]  # Mengambil 2 gambar untuk setiap baris
        
        # Membuat baris dengan layout grid
        cols = st.columns(2)
        
        # Menampilkan gambar dalam setiap kolom
        for j, col in enumerate(cols):
            if j < len(row_images):
                image_path = os.path.join(folder_path, row_images[j])
                image = Image.open(image_path)
                col.image(image, caption=row_images[j], use_column_width=True)

def main():
    
    st.title("Aplikasi Klasifikasi Gambar Bangun Datar")
    st.write("Silakan menggambar gambar di bawah ini.")

    # Memilih model
    selected_model = st.selectbox("Pilih Model:", ['Pilih Model'] + list(model_files.keys()))

    if selected_model != 'Pilih Model':
        # Memuat model yang dipilih
        model = load_selected_model()
        st.write(f"Model yang dipilih: {selected_model}") 
           
    class_name = st.selectbox("Pilih Jenis Bangun Datar:", class_names)
    st.write(f"Bangun Datar yang dipilih: {class_name}")

    # Membuat canvas untuk menggambar
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.8)",  # Warna latar belakang
        stroke_width=3,  # Ketebalan garis
        stroke_color="#000000",  # Warna garis
        background_color="#FFFFFF",  # Warna latar belakang canvas
        width=300,  # Lebar canvas
        height=300,  # Tinggi canvas
        drawing_mode="freedraw",  # Mode menggambar (freedraw: bebas)
        key="canvas"
    )

    if st.button("Klasifikasikan"):
        if selected_model == 'Pilih Model':
            st.write("Mohon pilih model yang valid.")
        elif canvas_result.image_data is not None and np.any(canvas_result.image_data != 255):
            # Ubah data gambar dari canvas menjadi objek PIL.Image
            image_pil = Image.fromarray(canvas_result.image_data)

            # Tampilkan gambar yang dihasilkan dari canvas
            st.image(image_pil, caption='Gambar yang Dihasilkan', use_column_width=True)

            # Prediksi kelas gambar
            predicted_class = predict_image(image_pil, model)

            # Tampilkan hasil prediksi
            st.write(f"Model yang Digunakan: {selected_model}")
            st.write(f"Jenis Bangun Datar yang Dipilih: {class_name}")
            st.write(f"Hasil Klasifikasi: {predicted_class}")

            # Cek apakah hasil klasifikasi sesuai dengan pilihan bangun datar
            if predicted_class != class_name:
                st.write(f"Gambar seharusnya termasuk dalam kategori {class_name}.")

            # Simpan gambar ke Firebase Storage
            file_name, file_url = save_to_firebase(image_pil, selected_model, class_name)
            st.write(f"Gambar berhasil disimpan di Firebase Storage dengan nama: {file_name}")
            st.write(f"URL Gambar: {file_url}")

        else:
            st.write("Canvas kosong, tidak ada gambar yang dihasilkan dari canvas.")

# Kontrol navigasi
pages = {
    "Beranda": main,
    "Tentang Aplikasi": show_about_app,
    "Cara Penggunaan": show_usage,
    "Tentang Data": show_about_data
}

# Pilihan navigasi
selected_page = st.sidebar.selectbox("Navigasi", list(pages.keys()))

# Memanggil fungsi yang sesuai berdasarkan pilihan navigasi
pages[selected_page]()