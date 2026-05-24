import pickle

pickle_path = r"F:\course_u-net\models\history_unet.pkl"

try:
    with open(pickle_path, "rb") as f:
        history = pickle.load(f)
except FileNotFoundError:
    print(f"Ошибка: Не удалось найти файл по пути {pickle_path}")
    exit()

total_epochs = len(history['val_loss'])

# Проверяем, есть ли вообще 3 эпохи в истории
start_epoch = max(0, total_epochs - 3)

keras_full_log = ""

# Цикл по последним 3 эпохам
for i in range(start_epoch, total_epochs):
    keras_full_log += "-" * 100 + "\n"
    keras_full_log += f"Эпоха {i + 1}/{total_epochs}\n"
    keras_full_log += f"loss: {history['loss'][i]:.4f} - dice_coefficient: {history['dice_coefficient'][i]:.4f} - iou_metric: {history['iou_metric'][i]:.4f} - accuracy: {history['accuracy'][i]:.4f}\n"
    keras_full_log += f"val_loss: {history['val_loss'][i]:.4f} - val_dice_coefficient: {history['val_dice_coefficient'][i]:.4f} - val_iou_metric: {history['val_iou_metric'][i]:.4f} - val_accuracy: {history['val_accuracy'][i]:.4f}\n"

with open("./static/last_3_epochs_log.txt", "w", encoding="utf-8") as log_file:
    log_file.write(keras_full_log)