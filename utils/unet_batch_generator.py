from utils.img_processing import load_gt_mask
from utils.img_processing import compute_ela
import numpy as np

def unet_batch_generator(pairs, batch_size=8, target_size=(256, 256), is_training=True):
    """
    Бесконечный Python-генератор батчей для Keras model.fit
    """
    num_samples = len(pairs)
    while True:
        if is_training:
            # Перемешиваем пары каждую эпоху, чтобы сеть не зазубривала порядок
            np.random.shuffle(pairs)
            
        for offset in range(0, num_samples, batch_size):
            batch_pairs = pairs[offset:offset+batch_size]
            
            X_batch = []
            Y_batch = []
            
            for pair in batch_pairs:
                try:
                    # На лету считаем ELA-снимок для картинки
                    ela_img = compute_ela(pair['tp_path'], target_size=target_size)
                    # Загружаем бинарную маску
                    mask = load_gt_mask(pair['gt_path'], target_size=target_size)
                    
                    X_batch.append(ela_img)
                    Y_batch.append(mask)
                except Exception as e:
                    # Если файл поврежден, просто пропускаем его, чтобы обучение не падало
                    continue
            
            # Keras ждет на вход numpy-массивы тензоров
            yield np.array(X_batch), np.array(Y_batch)
