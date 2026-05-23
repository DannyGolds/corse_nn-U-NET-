import pickle

# Путь к твоему файлу истории
pickle_path = "./models/history_unet.pkl"

try:
    with open(pickle_path, "rb") as f:
        history = pickle.load(f)
except FileNotFoundError:
    print(f"Ошибка: Не удалось найти файл по пути {pickle_path}")
    exit()

total_epochs = len(history['loss'])

# Проверяем, есть ли вообще 3 эпохи в истории
start_epoch = max(0, total_epochs - 3)

keras_full_log = ""

# Цикл по последним 3 эпохам
for i in range(start_epoch, total_epochs):
    epoch_num = i + 1
    
    # Строка заголовка эпохи: Epoch X/Y
    keras_full_log += f"Epoch {epoch_num}/{total_epochs}\n"
    
    # Имитируем строку прогресс-бара (для батча 32 и 1844 элементов это 58 шагов)
    keras_full_log += "58/58 ────────────────────────────── 0s 42ms/step"
    
    metrics_list = []
    
    # Тренировочные метрики для текущей эпохи i
    train_keys = ['loss', 'accuracy', 'dice_coefficient', 'dice_loss', 'iou_metric']
    for key in train_keys:
        if key in history:
            metrics_list.append(f"{key}: {history[key][i]:.4f}")
            
    # Валидационные метрики для текущей эпохи i
    val_keys = ['val_loss', 'val_accuracy', 'val_dice_coefficient', 'val_dice_loss', 'val_iou_metric']
    for key in val_keys:
        if key in history:
            metrics_list.append(f"{key}: {history[key][i]:.4f}")
            
    # Шаг обучения для текущей эпохи i
    if 'learning_rate' in history:
        metrics_list.append(f"learning_rate: {history['learning_rate'][i]:.2e}")
        
    # Собираем строку эпохи воедино через разделители " - "
    keras_full_log += " - " + " - ".join(metrics_list) + "\n"

print("--- Сгенерированный лог последних 3-х эпох в формате Keras ---")
print(keras_full_log)

# Сохраняем в текстовый файл для удобного копирования
with open("keras_last_3_epochs_log.txt", "w", encoding="utf-8") as f:
    f.write(keras_full_log)
print("Лог успешно сохранен в файл 'keras_last_3_epochs_log.txt'!")