import tensorflow as tf

# Загружаем модель (с compile=False, чтобы не мучиться с кастомными лоссами)
model = tf.keras.models.load_model(
    "F:\\course_u-net\\models\\best_unet_forensics.keras", 
    compile=False
)

# Выводим структуру модели в консоль
model.summary()

# Если захочешь сохранить эту таблицу в текстовый файл для курсовой:
with open('model_summary.txt', 'w', encoding='utf-8') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
print("Структура модели также сохранена в файл model_summary.txt!")