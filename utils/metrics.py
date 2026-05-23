import tensorflow as tf

K = tf.keras.backend

### Кастомные метрики для оценки сегментации
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Вычисляет коэффициент Дайса (схожесть масок).
    F1-мера на уровне отдельных пикселей.
    """
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


# Кастомные функции для борьбы с дисбалансом классов
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


def dice_focal_loss(gamma=2., alpha=0.25):
    """
    Фабрика функции потерь. Возвращает кастомный лосс с зафиксированными гиперпараметрами.
    Focal Loss выравнивает дисбаланс классов (фоновые пиксели vs пиксели склейки),
    а Dice Loss заставляет маску держать монолитную структуру.
    """
    def loss_fn(y_true, y_pred):
        # Принудительно приводим маску к float32, чтобы Keras не ругался на типы данных
        y_true_dt = tf.cast(y_true, tf.float32)
        y_pred_dt = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6) # Защита от log(0)
        
        # Вычисляем Focal Loss компоненту
        pt_1 = tf.where(tf.equal(y_true_dt, 1.0), y_pred_dt, tf.ones_like(y_pred_dt))
        pt_0 = tf.where(tf.equal(y_true_dt, 0.0), y_pred_dt, tf.zeros_like(y_pred_dt))
        
        f_loss = -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
                 - K.mean((1. - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
                 
        # Вычисляем Dice Loss компоненту
        d_loss = dice_loss(y_true_dt, y_pred_dt)
        
        # Суммируем обе потери
        return f_loss + d_loss
        
    return loss_fn

def iou_metric(y_true, y_pred, smooth=1e-6):
    """
    Вычисляет метрику IoU (Intersection over Union).
    Оценивает перекрытие между предсказанной и истинной маской.
    """
    y_true_f = K.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = K.flatten(y_pred)
    total = K.sum(y_true_f) + K.sum(y_pred_f)
    intersection = K.sum(y_true_f * y_pred_f)
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

