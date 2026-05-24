import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- 1. УКАЖИ СВОИ ПУТИ К ФАЙЛАМ ---
# Папка, где лежат только ВАЛИДАЦИОННЫЕ ELA-картинки (X)
VAL_IMAGES_DIR = "F:/course_u-net/data/validation/images/" 
# Папка, где лежат только ВАЛИДАЦИОННЫЕ черно-белые маски (Y)
VAL_MASKS_DIR = "F:/course_u-net/data/validation/masks/"   
# Путь к твоей сохраненной модели
MODEL_PATH = "F:/course_u-net/models/best_unet_forensics.keras"

IMG_SIZE = 256  # Размер, который принимает твоя сеть U-Net

# --- 2. ЗАГРУЗКА ДАННЫХ ИЗ ПАПОК НА ЛЕТУ ---
print("Загрузка валидационных данных...")
x_val_list = []
y_val_list = []

# Сортируем списки файлов, чтобы картинки и маски строго соответствовали друг другу
image_files = sorted(os.listdir(VAL_IMAGES_DIR))
mask_files = sorted(os.listdir(VAL_MASKS_DIR))

for img_name, mask_name in zip(image_files, mask_files):
    # Читаем ELA-картинку (RGB)
    img_path = os.path.join(VAL_IMAGES_DIR, img_name)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    x_val_list.append(img)
    
    # Читаем маску (в градациях серого)
    mask_path = os.path.join(VAL_MASKS_DIR, mask_name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    y_val_list.append(mask)

# Переводим в массивы NumPy и нормализуем
X_val = np.array(x_val_list, dtype=np.float32) / 255.0
y_val = np.array(y_val_list, dtype=np.float32) / 255.0

# Добавляем канальную размерность для масок, чтобы стало (461, 256, 256, 1)
y_val = np.expand_dims(y_val, axis=-1)

# Принудительно делаем маски бинарными (0 или 1)
y_true_bin = (y_val > 0.5).astype(np.uint8)

print(f"Успешно загружено {len(X_val)} валидационных пар.")
print(f"Размерность X_val: {X_val.shape}, Размерность y_val: {y_val.shape}")

# --- 3. ЗАГРУЗКА МОДЕЛИ И ПРЕДСКАЗАНИЕ ---
print("\nЗагрузка обученной модели U-Net...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

print("Модель генерирует предсказания масок...")
y_pred_probs = model.predict(X_val, batch_size=16)

# Переводим вероятности Сигмоиды в жесткие 0 и 1 (порог 0.5)
y_pred_bin = (y_pred_probs > 0.5).astype(np.uint8)

# --- 4. ВЫПРЯМЛЕНИЕ ПИКСЕЛЕЙ И РАСЧЕТ МАТРИЦЫ ---
print("\nВыпрямляем матрицы пикселей для попиксельного анализа...")
y_true_flat = y_true_bin.flatten()
y_pred_flat = y_pred_bin.flatten()

print("Вычисляем элементы матрицы ошибок...")
cm = confusion_matrix(y_true_flat, y_pred_flat)

# --- 5. ОТРИСОВКА МАТРИЦЫ ОШИБОК ---
classes = ['Оригинал', 'Манипуляция']
plt.figure(figsize=(6, 5.5))

# Оформляем точь-в-точь как на твоем референсе
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=classes, yticklabels=classes,
            linewidths=0.5, linecolor='#111111',
            annot_kws={"size": 12, "weight": "bold"})

plt.title('Confusion Matrix (Попиксельный анализ ELA)', fontsize=12, pad=10)
plt.ylabel('Actual (Истинные маски)', fontsize=11)
plt.xlabel('Predicted (Предсказания U-Net)', fontsize=11)

plt.tight_layout()

# Сохраняем готовую картинку, которую сразу вставишь в курсовую
output_filename = "confusion_matrix_final.png"
plt.savefig(output_filename, dpi=300)
plt.show()

print(f"\nВеликолепно! Матрица ошибок сохранена в файл: {output_filename}")