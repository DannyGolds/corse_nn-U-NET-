import tensorflow as tf
from utils.metrics import dice_coefficient, dice_focal_loss, iou_metric, dice_loss

layers = tf.keras.layers
models = tf.keras.models
K = tf.keras.backend

def build_unet(input_shape=(256, 256, 3)):
    """
    Улучшенная архитектура U-Net с регуляризацией для задач форензики.
    """
    inputs = layers.Input(input_shape, name="ela_input")

    # --- ЭНКОДЕР (Путь сжатия) ---
    
    # Блок 1 (256x256)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    # Блок 2 (128x128)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Dropout(0.1)(c2) # Легкий дропаут, чтобы не заучивала мелкий шум
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    # Блок 3 (64x64)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Dropout(0.2)(c3) # Увеличиваем регуляризацию глубже
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    # --- БОТТЛНЕК (32x32) ---
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Dropout(0.3)(c4) # Защищаем самое узкое горлышко от зубрежки
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)

    # --- ДЕКОДЕР (Путь расширения) ---
    
    # Блок 5 (Разжимаем до 64x64)
    u5 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c4)
    merge5 = layers.Concatenate()([u5, c3])
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(merge5)
    c5 = layers.BatchNormalization()(c5) 
    c5 = layers.Dropout(0.2)(c5)         
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)
    c5 = layers.BatchNormalization()(c5) 

    # Блок 6 (Разжимаем до 128x128)
    u6 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    merge6 = layers.Concatenate()([u6, c2])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(merge6)
    c6 = layers.BatchNormalization()(c6) 
    c6 = layers.Dropout(0.1)(c6)         
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)

    # Блок 7 (Разжимаем до 256x256)
    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    merge7 = layers.Concatenate()([u7, c1])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(merge7)
    c7 = layers.BatchNormalization()(c7) 
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7) 

    # --- ВЫХОДНОЙ СЛОЙ ---
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid', name="mask_output")(c7)

    model = models.Model(inputs, outputs, name="U-Net_Forensics_Engine")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss=dice_focal_loss(gamma=2.0, alpha=0.25), 
        metrics=['accuracy', dice_coefficient, iou_metric, dice_loss]
    )
    
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()