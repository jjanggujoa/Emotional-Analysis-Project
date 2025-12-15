# 🚀 감정 인식 CNN 모델 개발 프로젝트 (Emotion Recognition CNN Model)

이 프로젝트는 딥러닝(CNN)을 사용하여 얼굴 이미지에서 7가지 감정을 인식하고, 이를 동영상 스트림에 적용하여 실시간으로 감정을 탐지하고 표시하는 시스템입니다.

## 1. 프로젝트 개요

| 구분 | 내용 |
| :--- | :--- |
| **목적** | 얼굴 표정을 기반으로 7가지 주요 감정(Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)을 분류하는 CNN 모델 개발. |
| **핵심 기술** | Convolutional Neural Network (CNN), Keras/TensorFlow, OpenCV |
| **결과물** | 훈련된 모델 파일 (`emotion_cnn_model.h5`), 동영상 기반 감정 인식 시스템 |

## 2. 프로젝트 파일 구성

| 파일명 | 역할 |
| :--- | :--- |
| `data_preparation.py` | 데이터셋 로드, 정규화, 및 데이터 증강(Data Augmentation) 설정 |
| `cnn_model_training.py` | CNN 모델 설계, 컴파일, 훈련 및 모델 저장 |
| `emotion_recognition_video.py` | 훈련된 모델 로드 및 동영상 파일에서 감정 인식 및 시각화 |

## 3. 데이터 준비 및 전처리 (`data_preparation.py`)

훈련 데이터의 일반화 성능 향상을 위해 `ImageDataGenerator`를 사용한 데이터 증강을 적용했습니다.

### 3.1. 데이터셋 설정

* **입력 크기:** $48 \times 48$ 픽셀
* **색상 모드:** 흑백 (grayscale)
* **배치 크기:** 64
* **클래스 모드:** Categorical (7개 감정)

### 3.2. 데이터 증강 옵션 (훈련 데이터)

| 옵션 | 설정 값 | 목적 |
| :--- | :--- | :--- |
| `rescale` | `1./255` | 픽셀 값 정규화 |
| `rotation_range` | `30` | 이미지 회전 |
| `width_shift_range` | `0.2` | 가로 이동 |
| `height_shift_range` | `0.2` | 세로 이동 |
| `horizontal_flip` | `True` | 수평 뒤집기 |

## 4. CNN 모델 아키텍처 및 훈련 (`cnn_model_training.py`)

### 4.1. 모델 구조

본 모델은 3개의 Convolutional 블록과 Dropout이 적용된 완전 연결층(Dense Layer)으로 구성됩니다.

| 계층 (Layer) | 필터/유닛 수 | 커널 크기 | 활성화 함수 |
| :--- | :--- | :--- | :--- |
| **Input** | - | - | $48 \times 48 \times 1$ |
| **Conv2D + MaxPooling2D** | 32 | $(3,3)$ | ReLU |
| **Conv2D + MaxPooling2D** | 64 | $(3,3)$ | ReLU |
| **Conv2D + MaxPooling2D** | 128 | $(3,3)$ | ReLU |
| **Flatten** | - | - | - |
| **Dense** | 128 | - | ReLU |
| **Dropout** | 0.5 | - | - |
| **Dense (Output)** | 7 | - | Softmax |

### 4.2. 훈련 설정

* **옵티마이저:** `adam`
* **손실 함수:** `categorical_crossentropy`
* **평가 지표:** `accuracy`
* **에폭 (Epochs):** 30
* **모델 저장 경로:** `../models/emotion_cnn_model.h5`

## 5. 동영상 감정 인식 실행 (`emotion_recognition_video.py`)

훈련된 모델을 사용하여 비디오 파일에 감정 인식을 적용합니다.

### 5.1. 주요 실행 단계

1.  **모델 로드:** 저장된 `emotion_cnn_model.h5` 파일을 불러옵니다.
2.  **얼굴 탐지:** OpenCV의 `haarcascade_frontalface_default.xml`를 사용하여 프레임에서 얼굴을 탐지합니다.
3.  **전처리 및 예측:** 탐지된 얼굴 영역을 $48 \times 48$로 리사이즈 및 정규화한 후, 모델에 입력하여 7가지 감정 중 하나를 예측합니다.
4.  **시각화:** 인식된 감정 레이블을 얼굴 영역 위에 표시합니다 (박스: 녹색, 텍스트: 빨간색).
5.  **입력 비디오 경로:** `../videos/angry.mp4`

## 6. youtube 영상
https://youtu.be/11mVrU_ipw4
(실시간 감정분석이기 때문에 따로 사이트에서 실행 x)
