# 02. Multi-Layer Perceptron (MLP) with MNIST

## 📌 실습 개요

이 실습에서는 **MNIST 손글씨 숫자 데이터셋**을 이용하여  
TensorFlow/Keras 기반의 **다층 퍼셉트론(Multi-Layer Perceptron, MLP)** 을 구현합니다.

완성된 코드를 한 번에 실행하는 것이 아니라 다음 과정을 단계별로 확인합니다.

**MNIST 데이터 → 데이터 확인 → 정규화 → Flatten → MLP 구성 → 학습 → 평가 → 예측 → 오답 분석**

---

## 1. MNIST Dataset

MNIST는 0부터 9까지의 손글씨 숫자 이미지로 구성된 대표적인 이미지 분류 데이터셋입니다.

| 구분 | 데이터 수 |
|---|---:|
| Training Data | 60,000 |
| Test Data | 10,000 |

각 이미지는 다음과 같은 구조를 가집니다.

```text
Image Size : 28 × 28
Channel    : 1 (Gray Scale)
Class      : 10개
Label      : 0 ~ 9
```

하나의 이미지는 총 **784개(28 × 28)의 픽셀**로 구성됩니다.

```text
28 × 28 Image

[0   0   0  ...]
[0  25  89  ...]
[0 120 255  ...]
...
```

각 픽셀 값은 `0 ~ 255` 범위를 가집니다.

---

## 2. 라이브러리 불러오기

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
```

- `tensorflow` : 딥러닝 모델 구성 및 학습
- `numpy` : 배열 및 수치 계산
- `matplotlib` : 이미지와 학습 결과 시각화
- `Sequential` : Layer를 순서대로 쌓아 모델을 구성
- `Flatten` : 2차원 이미지를 1차원 벡터로 변환
- `Dense` : 모든 입력과 출력 뉴런이 연결된 완전연결층

---

## 3. MNIST 데이터 불러오기

TensorFlow/Keras에서 제공하는 `mnist.load_data()`를 이용합니다.

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
```

데이터의 크기를 확인합니다.

```python
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
```

실행 결과:

```text
(60000, 28, 28)
(60000,)
(10000, 28, 28)
(10000,)
```

이를 해석하면 다음과 같습니다.

| 변수 | Shape | 의미 |
|---|---|---|
| `x_train` | `(60000, 28, 28)` | 60,000개의 학습 이미지 |
| `y_train` | `(60000,)` | 학습 이미지의 정답 |
| `x_test` | `(10000, 28, 28)` | 10,000개의 테스트 이미지 |
| `y_test` | `(10000,)` | 테스트 이미지의 정답 |

---

## 4. 데이터 확인

첫 번째 이미지와 정답을 확인합니다.

```python
print("Label:", y_train[0])

plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()
```

이미지는 사람이 보기에는 숫자 그림이지만  
컴퓨터에서는 **28 × 28 크기의 숫자 배열**로 처리됩니다.

### 4.1 숫자별 데이터 분포 확인

MNIST 학습 데이터에 숫자 `0 ~ 9`가 각각 몇 개씩 포함되어 있는지 확인합니다.

```python
labels, counts = np.unique(y_train, return_counts=True)

plt.figure(figsize=(8, 5))
plt.bar(labels, counts)

plt.xticks(labels)
plt.xlabel("Digit")
plt.ylabel("Count")
plt.title("MNIST Training Data Distribution")
plt.grid(axis="y", alpha=0.3)

plt.show()
```

`np.unique()`를 이용하면 각 Label별 데이터 개수를 계산할 수 있습니다.

```python
labels, counts = np.unique(y_train, return_counts=True)
```

예를 들어,

```text
Digit
0  → 약 5,900개
1  → 약 6,700개
2  → 약 6,000개
...
9  → 약 5,900개
```

와 같이 각 숫자의 데이터 개수를 확인할 수 있습니다.

그래프를 통해 MNIST 데이터셋이 특정 숫자에 지나치게 편중되지 않고  
**각 클래스가 비교적 비슷한 수의 데이터를 가지고 있음**을 확인할 수 있습니다.

```text
Count
 ↑
 |       █
 | █ █ █ █ █ █ █ █ █
 | █ █ █ █ █ █ █ █ █
 +----------------------→ Digit
   0 1 2 3 4 5 6 7 8 9
```

---

### 4.2 픽셀 값 확인

픽셀 값의 범위를 확인합니다.

```python
print(x_train[0].min())
print(x_train[0].max())
```

실행 결과:

```text
0
255
```

---

## 5. 데이터 정규화

MNIST의 픽셀 값은 `0 ~ 255` 범위입니다.

딥러닝 학습을 안정적으로 수행하기 위해 다음과 같이 `0 ~ 1` 범위로 변환합니다.

```python
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
```

정규화는 다음과 같은 계산입니다.

```text
                 x
정규화된 값 = ───────
                255
```

예를 들어,

```text
0   → 0.0
128 → 약 0.502
255 → 1.0
```

정규화 이후 값의 범위를 확인합니다.

```python
print(x_train.min())
print(x_train.max())
```

결과:

```text
0.0
1.0
```

---

## 6. Flatten

MNIST 이미지는 `28 × 28` 형태의 2차원 데이터입니다.

하지만 Dense Layer는 입력을 1차원 벡터 형태로 사용합니다.

따라서 다음과 같이 변환합니다.

```text
28 × 28 Image

        ↓ Flatten

784 Vector
```

TensorFlow에서는 다음과 같이 작성합니다.

```python
Flatten(input_shape=(28, 28))
```

즉,

```text
(28, 28) → (784,)
```

로 변환됩니다.

---

## 7. MLP 구조

이번 실습에서는 다음 구조의 MLP를 사용합니다.

```text
Input Image
   28 × 28
      │
      ▼
   Flatten
      │
      ▼
     784
      │
      ▼
Dense(128, ReLU)
      │
      ▼
Dense(64, ReLU)
      │
      ▼
Dense(10, Softmax)
      │
      ▼
0 ~ 9 Classification
```

코드는 다음과 같습니다.

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])
```

---

## 8. Dense Layer

Dense Layer는 이전 층의 모든 입력이 다음 층의 모든 뉴런과 연결되는 구조입니다.

하나의 뉴런은 기본적으로 다음 계산을 수행합니다.

```text
z = x₁w₁ + x₂w₂ + ... + xₙwₙ + b
```

그리고 활성화 함수(Activation Function)를 적용합니다.

```text
출력 = Activation(z)
```

이번 실습에서는 은닉층에 `ReLU`를 사용합니다.

```python
Dense(128, activation="relu")
```

ReLU는 다음과 같이 동작합니다.

```text
z < 0  → 0
z ≥ 0  → z
```

수식:

```text
ReLU(z) = max(0, z)
```

---

## 9. 출력층과 Softmax

MNIST는 0부터 9까지 총 10개의 클래스를 분류합니다.

따라서 출력층의 뉴런 수는 10개입니다.

```python
Dense(10, activation="softmax")
```

Softmax는 각 클래스의 값을 **확률 형태**로 변환합니다.

예:

```text
[0.00, 0.01, 0.02, 0.90, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01]
```

가장 큰 확률을 가진 위치를 선택하면 예측 클래스가 됩니다.

```python
pred = np.argmax(predictions[0])
```

---

## 10. 모델 구조 확인

```python
model.summary()
```

모델의 Layer, Output Shape, Parameter 수를 확인할 수 있습니다.

---

## 11. 모델 컴파일

모델 학습 전에 다음 세 가지를 지정합니다.

```python
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
```

각 항목의 의미는 다음과 같습니다.

| 항목 | 설정 | 의미 |
|---|---|---|
| Optimizer | `adam` | Weight를 업데이트하는 방법 |
| Loss | `sparse_categorical_crossentropy` | 다중 클래스 분류 오차 계산 |
| Metric | `accuracy` | 분류 정확도 |

`y_train`이 다음처럼 정수 Label이므로

```text
5
0
4
1
9
...
```

`sparse_categorical_crossentropy`를 사용합니다.

---

## 12. 모델 학습

```python
history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)
```

각 설정의 의미는 다음과 같습니다.

| 설정 | 의미 |
|---|---|
| `epochs=10` | 전체 학습 데이터를 10번 반복 |
| `batch_size=128` | 128개 데이터마다 Weight 업데이트 |
| `validation_split=0.2` | 학습 데이터의 20%를 검증용으로 사용 |

학습의 기본 흐름은 다음과 같습니다.

```text
Input
  ↓
Feed-Forward
  ↓
Prediction
  ↓
Loss 계산
  ↓
Backpropagation
  ↓
Weight Update
  ↓
다음 Batch
```

---

## 13. Accuracy 확인

학습 과정에서 정확도를 확인할 수 있습니다.

```python
print(history.history["accuracy"])
print(history.history["val_accuracy"])
```

- `accuracy` : Training Accuracy
- `val_accuracy` : Validation Accuracy

두 값의 차이가 너무 크다면 **Overfitting**을 의심할 수 있습니다.

---

## 14. Accuracy 그래프

```python
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
```

일반적으로 학습이 진행되면 정확도는 증가합니다.

```text
Accuracy
   ↑
1.0|                 ─────
   |             ───
   |         ───
   |      ──
   |   ──
0.0+------------------------→ Epoch
```

---

## 15. Loss 그래프

```python
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()
```

학습이 잘 진행된다면 일반적으로 Loss는 감소합니다.

---

## 16. 테스트 데이터 평가

학습에 사용하지 않은 Test Dataset으로 모델을 평가합니다.

```python
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
```

Training Data가 아니라 별도의 Test Data를 사용하는 이유는  
모델이 **처음 보는 데이터에도 잘 동작하는지** 확인하기 위해서입니다.

---

## 17. 실제 숫자 예측

테스트 이미지 10개를 예측합니다.

```python
predictions = model.predict(x_test[:10], verbose=0)
```

첫 번째 데이터의 예측 결과를 확인합니다.

```python
print(predictions[0])
```

가장 높은 확률의 위치를 구합니다.

```python
pred = np.argmax(predictions[0])

print("예측:", pred)
print("정답:", y_test[0])
```

---

## 18. 예측 결과 시각화

```python
plt.figure(figsize=(12, 6))

predictions = model.predict(x_test[:15], verbose=0)

for i in range(15):
    pred = np.argmax(predictions[i])
    actual = y_test[i]

    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i], cmap="gray")
    plt.title(f"Pred: {pred}\nTrue: {actual}")
    plt.axis("off")

plt.tight_layout()
plt.show()
```

---

## 19. 오답 데이터 찾기

전체 Test Dataset을 예측합니다.

```python
predictions = model.predict(x_test, verbose=0)

pred_labels = np.argmax(predictions, axis=1)
```

예측과 실제 정답이 다른 위치를 찾습니다.

```python
wrong_indices = np.where(pred_labels != y_test)[0]
```

오답 수를 확인합니다.

```python
print("전체 테스트 데이터:", len(y_test))
print("오답 개수:", len(wrong_indices))
```

---

## 20. 오답 이미지 확인

```python
plt.figure(figsize=(12, 6))

for i, idx in enumerate(wrong_indices[:15]):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[idx], cmap="gray")
    plt.title(
        f"Pred: {pred_labels[idx]}\n"
        f"True: {y_test[idx]}"
    )
    plt.axis("off")

plt.tight_layout()
plt.show()
```

오답을 직접 확인하면 모델이 어떤 형태의 숫자를 어려워하는지 관찰할 수 있습니다.

---

## 💡 핵심 정리

```text
MNIST Dataset
60,000 Train / 10,000 Test

        │
        ▼

Normalization
0 ~ 255
   ↓
0 ~ 1

        │
        ▼

Flatten
28 × 28
   ↓
784

        │
        ▼

Dense(128, ReLU)

        │
        ▼

Dense(64, ReLU)

        │
        ▼

Dense(10, Softmax)

        │
        ▼

0 ~ 9 Classification
```

---

## 실습 문제

### 실습 1. 은닉층 뉴런 수 변경

첫 번째 Dense Layer의 뉴런 수를 변경하고 정확도를 비교합니다.

```text
32
64
128
256
```

---

### 실습 2. Layer 추가

다음 구조로 변경합니다.

```text
Flatten
Dense(256, ReLU)
Dense(128, ReLU)
Dense(64, ReLU)
Dense(10, Softmax)
```

기존 모델과 Test Accuracy를 비교합니다.

---

### 실습 3. Epoch 변경

다음 값을 각각 적용합니다.

```python
epochs=5
epochs=10
epochs=20
```

Train Accuracy와 Validation Accuracy의 차이를 확인합니다.

---

### 실습 4. Activation Function 변경

`relu`를 `sigmoid`로 변경합니다.

```python
Dense(128, activation="sigmoid")
```

학습 속도와 정확도를 비교합니다.

---

### 실습 5. 모델 구조 직접 설계

자신만의 MLP를 만들어보세요.

조건:

```text
은닉층 2개 이상
출력층 10개
Softmax 사용
Test Accuracy 출력
```

---
