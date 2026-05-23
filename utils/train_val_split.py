from utils.build_pairs import build_data_pairs
from utils.unet_batch_generator import unet_batch_generator
import numpy as np
from sklearn.model_selection import train_test_split

def get_train_val_generators(tp_dir, gt_dir, batch_size=8, target_size=(256, 256), split_ratio=0.15):
    """
    Собирает пары, делит их на Train/Val и возвращает готовые генераторы вместе со steps_per_epoch
    """
    all_pairs = build_data_pairs(tp_dir, gt_dir)
    
    if len(all_pairs) == 0:
        raise RuntimeError("Не найдено ни одной валидной пары изображений и масок!")
        
    train_pairs, val_pairs = train_test_split(all_pairs, test_size=split_ratio, random_state=42)
    
    print(f"Распределение: {len(train_pairs)} пар на обучение, {len(val_pairs)} пар на валидацию.")
    
    train_gen = unet_batch_generator(train_pairs, batch_size=batch_size, target_size=target_size, is_training=True)
    val_gen = unet_batch_generator(val_pairs, batch_size=batch_size, target_size=target_size, is_training=False)
    
    # Считаем количество шагов для одной эпохи
    train_steps = int(np.ceil(len(train_pairs) / batch_size))
    val_steps = int(np.ceil(len(val_pairs) / batch_size))
    
    return train_gen, val_gen, train_steps, val_steps