import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 모델 로드
model = load_model('../models/emotion_cnn_model.h5')
emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# 얼굴 탐지기
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 🔹 동영상 파일 경로 지정
video_path = '../videos/angry.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ 동영상을 열 수 없습니다. 경로를 확인하세요.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 동영상 끝나면 종료

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi = roi_gray.astype('float') / 255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        preds = model.predict(roi, verbose=0)
        label = emotion_labels[np.argmax(preds)]

        # 초록색 얼굴 박스 + 빨간색 감정 텍스트
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()