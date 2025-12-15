from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 불러오기
train_dir = '../emotion_dataset/train'
test_dir = '../emotion_dataset/test'

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(48,48), color_mode='grayscale',
    batch_size=64, class_mode='categorical'
)
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(48,48), color_mode='grayscale',
    batch_size=64, class_mode='categorical'
)

# CNN 모델 설계
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 모델 훈련
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=30
)

# 모델 저장
model.save('../models/emotion_cnn_model.h5')
print("✅ 모델 저장 완료: models/emotion_cnn_model.h5")