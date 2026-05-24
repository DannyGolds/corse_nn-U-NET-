import pickle
import matplotlib.pyplot as plt

with open("F:/course_u-net/models/history_unet.pkl", "rb") as f:
    history = pickle.load(f)

total_epochs = len(history['val_loss'])

print(history["loss"][-1])
print(history["val_loss"][-1])
print(history["dice_coefficient"][-1])
print(history["val_dice_coefficient"][-1])
print(history["iou_metric"][-1])
print(history["val_iou_metric"][-1])
print(history["accuracy"][-1])
print(history["val_accuracy"][-1])


plt.figure(figsize=(12, 8))
plt.plot(range(1, total_epochs + 1), history['loss'], label='Train Loss', color='blue')
plt.plot(range(1, total_epochs + 1), history['val_loss'], label='Val Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Ошибки на обучающем и валидационном наборах')
plt.legend()
plt.grid()
plt.savefig("F:/course_u-net/showing_learning/static/loss_plot.png", dpi=300)
print("График потерь сохранен как 'F:/course_u-net/showing_learning/static/loss_plot.png'") 
