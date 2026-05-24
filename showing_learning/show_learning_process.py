import pickle
import matplotlib.pyplot as plt
import os

pickle_path = r"F:\course_u-net\models\history_unet.pkl"

with open(pickle_path, 'rb') as f:
    history = pickle.load(f)

epochs = range(1, len(history['loss']) + 1)

# Устанавливаем аккуратный академический стиль
plt.style.use('seaborn-v0_8-whitegrid')

# Окно 1: Коллаж 2х2 из основных метрик обучения и валидации
fig, axs = plt.subplots(2, 2, figsize=(16, 11))

# 1. Функция потерь (Total Loss)
axs[0, 0].plot(epochs, history['loss'], 'b-o', label='Train Loss', linewidth=2, markersize=3)
axs[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2, markersize=3)
axs[0, 0].set_title('Общие потери (Dice Focal Loss)', fontsize=12, fontweight='bold')
axs[0, 0].set_xlabel('Эпоха')
axs[0, 0].set_ylabel('Значение')
axs[0, 0].legend()

# 2. Коэффициент Дайса (Dice Coefficient / F1-мера)
axs[0, 1].plot(epochs, history['dice_coefficient'], 'b-o', label='Train Dice', linewidth=2, markersize=3)
axs[0, 1].plot(epochs, history['val_dice_coefficient'], 'r-s', label='Val Dice', linewidth=2, markersize=3)
axs[0, 1].set_title('Коэффициент Дайса (Пиксельная F1-мера)', fontsize=12, fontweight='bold')
axs[0, 1].set_xlabel('Эпоха')
axs[0, 1].set_ylabel('Точность [0..1]')
axs[0, 1].legend()

# 3. Метрика IoU (Jaccard Index)
axs[1, 0].plot(epochs, history['iou_metric'], 'b-o', label='Train IoU', linewidth=2, markersize=3)
axs[1, 0].plot(epochs, history['val_iou_metric'], 'r-s', label='Val IoU', linewidth=2, markersize=3)
axs[1, 0].set_title('Метрика IoU (Пересечение над объединением)', fontsize=12, fontweight='bold')
axs[1, 0].set_xlabel('Эпоха')
axs[1, 0].set_ylabel('Точность [0..1]')
axs[1, 0].legend()

# 4. Пиксельная точность (Accuracy)
axs[1, 1].plot(epochs, history['accuracy'], 'b-o', label='Train Acc', linewidth=2, markersize=3)
axs[1, 1].plot(epochs, history['val_accuracy'], 'r-s', label='Val Acc', linewidth=2, markersize=3)
axs[1, 1].set_title('Общая пиксельная точность (Accuracy)', fontsize=12, fontweight='bold')
axs[1, 1].set_xlabel('Эпоха')
axs[1, 1].set_ylabel('Доля верных пикселей')
axs[1, 1].legend()

plt.tight_layout()
main_plot_path = "./static/unet_training_metrics_grid.png"
plt.savefig(main_plot_path, dpi=300)
print(f"Основной коллаж графиков сохранен как '{main_plot_path}'")


# Окно 2: График изменения Скорости Обучения (Learning Rate)
if 'learning_rate' in history:
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history['learning_rate'], 'g-p', label='Learning Rate', linewidth=2, markersize=4)