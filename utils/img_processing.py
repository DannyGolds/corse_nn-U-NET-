import cv2
import numpy as np
import os
from PIL import Image

def compute_ela(image_path, quality=90, scale=15, target_size=(256, 256)):
    """
    Вычисляет Error Level Analysis (ELA) для изображения.
    """
    # 1. Надежно читаем оригинальное изображение через OpenCV
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Не удалось открыть изображение по пути: {image_path}")
        
    # Приводим к нужному размеру ДО сохранения, чтобы не искажать сетку JPEG
    orig = cv2.resize(orig, target_size)
    orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)

    # 2. Переводим в PIL, чтобы сохранить с точным качеством JPEG
    pil_img = Image.fromarray(orig)
    
    # Используем временный файл в локальной памяти Colab (это быстро)
    tmp_path = "temp_ela.jpg"
    pil_img.save(tmp_path, 'JPEG', quality=quality)

    # 3. Читаем временный JPEG обратно
    compressed = cv2.imread(tmp_path)
    compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
    
    # Чистим за собой временный файл
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    # 4. Считаем абсолютную разницу между оригиналом и сжатой копией
    # Переводим в float32, чтобы избежать переполнения типов при вычитании
    diff = np.abs(orig.astype(np.float32) - compressed.astype(np.float32))
    
    # 5. Умножаем на scale, чтобы подсветить микро-аномалии форензики
    ela_img = diff * scale
    
    # 6. Жестко нормируем в диапазон [0.0, 1.0]
    # Используем клиппинг, чтобы значения не улетали выше 255
    ela_img = np.clip(ela_img, 0, 255) / 255.0
    
    return ela_img

def load_gt_mask(mask_path, target_size=(256, 256)):
    """
    Загружает истинную маску фотомонтажа и приводит её к бинарному виду (256, 256, 1).
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Не удалось прочитать маску: {mask_path}")
        
    # Ресайзим маску
    mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    
    # Бинаризуем: всё, что не черный фон -> белый объект (1.0)
    _, binary_mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    
    # Нормализуем к [0.0, 1.0] и добавляем канал для U-Net (256, 256, 1)
    binary_mask = binary_mask.astype(np.float32) / 255.0
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    
    return binary_mask
