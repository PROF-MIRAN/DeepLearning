import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. Iris 데이터 불러오기
# ============================================================

iris = datasets.load_iris()

X_all = iris.data
Y_all = iris.target


print("===== Iris Dataset =====")

print("데이터 크기 :", X_all.shape)
print("레이블 크기 :", Y_all.shape)

print("\nFeature:")
print(iris.feature_names)

print("\nClass:")
print(iris.target_names)


# ============================================================
# 2. Setosa(0), Versicolor(1) 데이터만 추출
# ============================================================

mask = (Y_all == 0) | (Y_all == 1)

X = X_all[mask]
Y = Y_all[mask]


print("\n===== Binary Classification Dataset =====")

print("X shape :", X.shape)
print("Y shape :", Y.shape)

print("\nOriginal Label:")
print(Y)


# ============================================================
# 3. Label 변환
#
# Setosa      : 0 → +1
# Versicolor  : 1 → -1
# ============================================================

Y = np.where(Y == 0, 1, -1)


print("\nConverted Label:")
print(Y)


# ============================================================
# 4. 데이터 표준화
# ============================================================

scaler = StandardScaler()

X = scaler.fit_transform(X)


print("\n===== Standardization =====")

print("Standardized X:")
print(X[:10])


# ============================================================
# 5. Weight / Bias 초기화
# ============================================================

w = np.array([
    0.5,
    0.5,
    0.5,
    0.5
])

b = 0.1


print("\n===== Initial Parameters =====")

print("Weight :", w)
print("Bias   :", b)


# ============================================================
# 6. Feed-Forward
#
# z = Xw + b
# ============================================================

z = X @ w + b


print("\n===== Weighted Sum =====")

print("z shape :", z.shape)
print("z:")
print(z)


# ============================================================
# 7. Step Function
#
# z >= 0 → +1
# z <  0 → -1
# ============================================================

y_pred = np.where(
    z >= 0,
    1,
    -1
)


print("\n===== Prediction =====")

print(y_pred)


# ============================================================
# 8. 정확도 계산
# ============================================================

classified = np.sum(
    y_pred == Y
)

accuracy = np.mean(
    y_pred == Y
)


print("\n===== Classification Result =====")

print(
    f"정분류 개수: "
    f"{classified} / {len(Y)}"
)

print(
    f"Accuracy: "
    f"{accuracy:.3f}"
)


# ============================================================
# 9. Loss 계산
# ============================================================

# Mean Squared Error
mse = np.mean(
    (Y - y_pred) ** 2
)

# Mean Absolute Error
mae = np.mean(
    np.abs(Y - y_pred)
)


print("\n===== Loss =====")

print(
    f"MSE = {mse:.3f}"
)

print(
    f"MAE = {mae:.3f}"
)