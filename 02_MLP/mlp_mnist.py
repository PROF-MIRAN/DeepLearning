import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


# ============================================================
# 1. MNIST Dataset
# ============================================================

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape :", x_test.shape)
print("y_test shape :", y_test.shape)


# ============================================================
# 2. 데이터 확인
# ============================================================

print("\n첫 번째 이미지 Label:", y_train[0])
print("Pixel Min:", x_train[0].min())
print("Pixel Max:", x_train[0].max())

plt.imshow(x_train[0], cmap="gray")
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()


# ============================================================
# 2-1. 숫자별 데이터 분포 확인
# ============================================================

labels, counts = np.unique(y_train, return_counts=True)

print("\n숫자별 학습 데이터 개수")
for label, count in zip(labels, counts):
    print(f"{label}: {count}")

plt.figure(figsize=(8, 5))
plt.bar(labels, counts)

plt.xticks(labels)
plt.xlabel("Digit")
plt.ylabel("Count")
plt.title("MNIST Training Data Distribution")
plt.grid(axis="y", alpha=0.3)

plt.show()


# ============================================================
# 3. 데이터 정규화
# ============================================================

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

print("\n정규화 후")
print("Min:", x_train.min())
print("Max:", x_train.max())


# ============================================================
# 4. MLP 모델 생성
# ============================================================

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])

model.summary()


# ============================================================
# 5. 모델 컴파일
# ============================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ============================================================
# 6. 모델 학습
# ============================================================

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.2
)


# ============================================================
# 7. Accuracy 그래프
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training / Validation Accuracy")
plt.legend()
plt.grid()
plt.show()


# ============================================================
# 8. Loss 그래프
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training / Validation Loss")
plt.legend()
plt.grid()
plt.show()


# ============================================================
# 9. Test Dataset 평가
# ============================================================

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


# ============================================================
# 10. 실제 예측
# ============================================================

predictions = model.predict(x_test[:10], verbose=0)

print("\n첫 번째 이미지의 클래스별 확률:")
print(predictions[0])

print("예측:", np.argmax(predictions[0]))
print("정답:", y_test[0])


# ============================================================
# 11. 예측 결과 시각화
# ============================================================

plt.figure(figsize=(12, 6))

predictions = model.predict(x_test[:15], verbose=0)

for i in range(15):

    pred = np.argmax(predictions[i])
    actual = y_test[i]

    plt.subplot(3, 5, i + 1)

    plt.imshow(x_test[i], cmap="gray")

    plt.title(
        f"Pred: {pred}\n"
        f"True: {actual}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# 12. 오답 분석
# ============================================================

predictions = model.predict(x_test, verbose=0)

pred_labels = np.argmax(
    predictions,
    axis=1
)

wrong_indices = np.where(
    pred_labels != y_test
)[0]

print("\n전체 테스트 데이터:", len(y_test))
print("오답 개수:", len(wrong_indices))


# ============================================================
# 13. 오답 이미지 확인
# ============================================================

plt.figure(figsize=(12, 6))

for i, idx in enumerate(wrong_indices[:15]):

    plt.subplot(3, 5, i + 1)

    plt.imshow(
        x_test[idx],
        cmap="gray"
    )

    plt.title(
        f"Pred: {pred_labels[idx]}\n"
        f"True: {y_test[idx]}"
    )

    plt.axis("off")

plt.tight_layout()
plt.show()
