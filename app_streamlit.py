import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import os
import sys

# Добавляем корень проекта в пути, чтобы Streamlit видел нашу папку src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.img_processing import compute_ela
from utils.metrics import iou_metric, dice_coefficient, dice_focal_loss

# ========================================================
# 🔑 ЗАГРУЗКА МОДЕЛИ С КЭШИРОВАНИЕМ (чтобы приложение не тупило)
# ========================================================
@st.cache_resource # Твой стримлитовский кэш
def load_forensics_model():
    model_path = "F:/course_u-net/models/best_unet_forensics.keras"
    
    model = tf.keras.models.load_model(
    "F:/course_u-net/models/best_unet_forensics.keras", 
    compile=False
)
    return model

# ========================================================
# 🎨 НАСТРОЙКА ИНТЕРФЕЙСА STREAMLIT
# ========================================================
st.set_page_config(
    page_title="Photo Forensics Neural Engine",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Photo Forensics Neural Engine (U-Net + ELA)")
st.write(
    "Данный модуль предназначен для попиксельного обнаружения областей локальных манипуляций "
    "(сплайсинг, ретушь, копирование-вставка) на цифровых изображениях с использованием анализа уровня ошибок сжатия."
)

st.markdown("---")

# Загружаем модель
with st.spinner("🧠 Загрузка нейросетевого движка U-Net..."):
    model = load_forensics_model()

if model is None:
    st.error("🚨 Файл модели `best_unet_forensics.keras` не найден в папке `models/`!")
    st.info("Пожалуйста, перенесите обученные веса с Google Диска в локальную папку проекта `models/` рядом с этим скриптом.")
    st.stop()

# Создаем интерфейс в два столбца: загрузка и настройки
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📥 Исходные данные")
    uploaded_file = st.file_uploader(
        "Выберите изображение для криминалистического анализа...", 
        type=["jpg", "jpeg", "png"]
    )
    
    st.subheader("⚙️ Параметры детектора")
    ela_quality = st.slider("Качество повторного сжатия (JPEG Quality)", 50, 100, 90)
    ela_scale = st.slider("Коэффициент усиления аномалий (Scale)", 5, 50, 15)
    detection_threshold = st.slider("Порог уверенности сети (Threshold)", 0.1, 0.9, 0.5, step=0.05)

# Если файл загружен — запускаем анализ форензики
if uploaded_file is not None:
    # Сохраняем загруженный файл временно на диск, чтобы передать в compute_ela
    temp_input_path = "temp_user_input.jpg"
    image = Image.open(uploaded_file)

    # Если у картинки есть альфа-канал (RGBA), конвертируем в RGB
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    image.save(temp_input_path, "JPEG", quality=100)
    
    with col2:
        st.subheader("🔍 Результаты криминалистического анализа")
        
        # Индикатор прогресса
        with st.spinner("🕒 Вычисление ELA и инференс нейросети..."):
            try:
                # 1. Считаем ELA-снимок по нашей исправленной логике
                ela_res = compute_ela(temp_input_path, quality=ela_quality, scale=ela_scale, target_size=(256, 256))
                
                # Подготавливаем тензор для модели (1, 256, 256, 3)
                input_tensor = np.expand_dims(ela_res, axis=0)
                
                # 2. Нейросеть рисует маску
                pred_raw = model.predict(input_tensor, verbose=0)[0]
                
                # Применяем порог уверенности
                pred_mask = (pred_raw.squeeze() > detection_threshold).astype(np.uint8) * 255
                
                # Очищаем за собой временный файл
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
                    
                    # ========================================================
                # 🛡️ УМНАЯ ПОСТОБРАБОТКА МАСКИ И ВЫНЕСЕНИЕ ВЕРДИКТА
                # ========================================================
                
                # Применяем порог уверенности сети
                pred_mask = (pred_raw.squeeze() > detection_threshold).astype(np.uint8) * 255
                
                # Морфологическая фильтрация (убираем мелкую "сыпь" и связываем крупные объекты)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                # Сначала opening (убирает изолированные точки), потом closing (затягивает дыры внутри маски)
                processed_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
                processed_mask = cv2.morphologyEx(processed_mask, cv2.MORPH_CLOSE, kernel)
                
                # Считаем площади
                total_forged_pixels = np.sum(processed_mask > 0)
                total_pixels = processed_mask.size
                forgery_percentage = (total_forged_pixels / total_pixels) * 100
                
                # Очищаем за собой временный файл
                if os.path.exists(temp_input_path):
                    os.remove(temp_input_path)
            
            except Exception as e:
                st.error(f"Ошибка при обработке файла: {e}")
                st.stop()

        # Отображаем результаты в три колонки для наглядного сравнения
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.image(image, caption="Оригинал (Загружен пользователем)", use_container_width=True)
            
        with res_col2:
            ela_vis = (ela_res - ela_res.min()) / (ela_res.max() - ela_res.min() + 1e-6)
            st.image(ela_vis, caption="Вычисленный ELA-профиль", use_container_width=True)
            
        with res_col3:
            # Выводим уже отфильтрованную, чистую маску
            st.image(processed_mask, caption="Локализованный след (Фильтрация шума)", use_container_width=True, channels="GRAY")
            
        st.markdown("---")
        st.subheader("📊 Экспертное заключение системы:")

        # Выносим вердикт на основе порога площади (задаем, например, 2.0%)
        AREA_THRESHOLD_PCT = 15.0

        if forgery_percentage >= AREA_THRESHOLD_PCT:
            st.error(
                f"🚨 **ОБНАРУЖЕНА ЛОКАЛЬНАЯ МОДИФИКАЦИЯ ИЗОБРАЖЕНИЯ!** "
                f"Найдено аномальное изменение структуры пикселей на площади {forgery_percentage:.2f}% от всего кадра. "
                f"Выделенная область выходит за рамки погрешности естественного сжатия JPEG."
            )
        else:
            st.success(
                f"✅ **ПРИЗНАКОВ НАПРАВЛЕННОГО ФОТОМОНТАЖА НЕ ОБНАРУЖЕНО.** "
                f"Обнаруженные микроаномалии составляют всего {forgery_percentage:.2f}% кадра, "
                f"что укладывается в допустимый порог естественного шума контрастности ({AREA_THRESHOLD_PCT}%). "
                f"Изображение признано целостным."
            )