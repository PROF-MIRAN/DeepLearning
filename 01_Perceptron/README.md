# 01. Perceptron

## 📌 실습 개요

이 실습에서는 **Iris(붓꽃) 데이터셋**을 이용하여 퍼셉트론(Perceptron)의 기본적인 연산 과정을 직접 구현합니다.

Scikit-learn의 완성된 Perceptron 모델을 바로 사용하는 것이 아니라 NumPy를 이용하여 다음 과정을 단계별로 확인합니다.

**Iris 데이터 → 데이터 추출 → 표준화 → 가중합 → 활성화 함수 → 예측 → 성능 확인**

---

## 1. Iris Dataset

Iris 데이터셋은 3종의 붓꽃으로 구성되어 있습니다.

![Iris Dataset](./images/iris.png)

| Class | 이름 |
|---|---|
| 0 | Setosa |
| 1 | Versicolor |
| 2 | Virginica |

각 데이터는 다음 4개의 특징(Feature)을 가지고 있습니다.

| Feature | 의미 |
|---|---|
| Sepal Length | 꽃받침 길이 |
| Sepal Width | 꽃받침 너비 |
| Petal Length | 꽃잎 길이 |
| Petal Width | 꽃잎 너비 |

![Iris Dataset Example](./images/iris_vector_ex.png)

각 품종별 50개씩 총 **150개의 데이터**가 존재합니다.

---

## 2. 라이브러리 불러오기

```python
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
```

- `numpy` : 배열 연산, 선형대수, 난수 생성, 수학 함수 등 수치 계산에 특화된 라이브러리

   **공식 문서 링크:** [NumPy Documentation](https://numpy.org/doc/stable/)
- `datasets` : Scikit-learn에서 머신러닝 학습 및 실습을 위해 제공하는 데이터셋 모듈

   **공식 문서 링크:** [Scikit-learn/dataset](https://scikit-learn.org/stable/datasets.html)

   **Scikit-learn Github:**[Scikit-learn/github](https://github.com/scikit-learn/scikit-learn)

- `StandardScaler` : Scikit-learn의 데이터 전처리(Preprocessing) 기능 중 하나이며, 
각 Feature의 평균과 데이터 범위를 일정한 기준으로 맞추기 위해 표준화(Standardization)를 수행함. 

   **공식 문서 링크:** [Scikit-learn/StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)
---

## 3. Iris 데이터 불러오기
Scikit-learn에서 제공하는 `load_iris()` 함수를 이용하여 **Iris Dataset**을 불러옵니다.

```python
iris = datasets.load_iris()

X_all = iris.data
Y_all = iris.target
```
- `datasets.load_iris()` : Scikit-learn에 내장된 Iris Dataset을 불러오는 함수
- `iris.data` : 붓꽃의 4가지 특징(Feature) 값을 저장한 입력 데이터
- `iris.target` : 각 붓꽃이 어떤 품종인지 나타내는 정답(Label) 데이터



데이터셋의 구조는 다음 명령으로 확인할 수 있습니다.

불러온 `iris`에는 입력 데이터뿐만 아니라 클래스 이름, Feature 이름, 데이터셋 설명 등 다양한 정보가 함께 저장되어 있습니다.

```python
```python
print(iris.keys())
```

실행 결과:

```text
dict_keys([
    'data',
    'target',
    'frame',
    'target_names',
    'DESCR',
    'feature_names',
    'filename',
    'data_module'
])
```

주요 항목의 의미는 다음과 같습니다.

| Key | 의미 |
|---|---|
| `data` | 4가지 Feature로 구성된 입력 데이터 |
| `target` | 각 데이터의 정답 Label (0, 1, 2) |
| `target_names` | Label에 해당하는 붓꽃 품종 이름 |
| `feature_names` | 4가지 Feature의 이름 |
| `DESCR` | Iris Dataset에 대한 상세 설명 |

```python 
    print(iris.keys())
    print(iris.target_names)
    print(iris.feature_names)
```
---

---

## 4. 이진 분류 데이터 생성

Iris 데이터에는 3개의 클래스가 있지만,
이번 퍼셉트론 실습에서는 다음 두 품종만 사용합니다.

- Setosa : 0
- Versicolor : 1

### 4.1 필요한 데이터 선택하기

전체 Label `Y_all`에서 값이 `0` 또는 `1`인 데이터의 위치를 찾습니다.

```python
mask = (Y_all == 0) | (Y_all == 1)
```

여기서 각 조건의 의미는 다음과 같습니다.

| 코드 | 의미 |
|---|---|
| `Y_all == 0` | Label이 Setosa인지 확인 |
| `Y_all == 1` | Label이 Versicolor인지 확인 |
| `\|` | 두 조건 중 하나라도 참이면 `True` |
| `mask` | 조건을 만족하는 데이터 위치를 저장 |

---

### 4.2 Boolean Mask란?

`Y_all == 0`과 같은 조건을 실행하면 각 데이터가 조건을 만족하는지에 따라  
`True` 또는 `False`로 구성된 배열이 생성됩니다.

예를 들어,

```python
Y_all
```

```text
[0, 0, 0, ..., 1, 1, 1, ..., 2, 2, 2, ...]
```

위 데이터에 다음 조건을 적용하면,

```python
mask = (Y_all == 0) | (Y_all == 1)
```

개념적으로 다음과 같은 결과가 만들어집니다.

```text
Y_all : [0,    0,    ...  1,    1,    ...  2,     2,   ...]
         ↓     ↓          ↓     ↓          ↓      ↓

mask  : [True, True, ... True, True, ... False, False, ...]
```

즉,

- Setosa (`0`) → `True`
- Versicolor (`1`) → `True`
- Virginica (`2`) → `False`

가 됩니다.

> **Boolean Masking**  
> `True`와 `False`로 이루어진 배열을 이용하여 원하는 데이터만 선택하는 방법입니다.

---
### 4.3 Setosa와 Versicolor 데이터 추출

앞에서 만든 `mask`를 입력 데이터 `X_all`과 정답 데이터 `Y_all`에 동일하게 적용합니다.

```python
X = X_all[mask]
Y = Y_all[mask]
```

- `X_all[mask]` : 조건을 만족하는 입력 데이터(Feature)만 선택
- `Y_all[mask]` : 해당 데이터의 정답(Label)만 선택

따라서 `Virginica` 데이터는 제외되고 **Setosa와 Versicolor 데이터만 남게 됩니다.**

---
### 4.4 추출된 데이터 확인

```python
print(X.shape)
print(Y.shape)
```

실행 결과:

```text
(100, 4)
(100,)
```

이를 해석하면 다음과 같습니다.

| 변수 | Shape | 의미 |
|---|---|---|
| `X` | `(100, 4)` | 100개 데이터 × 4개 Feature |
| `Y` | `(100,)` | 100개 데이터의 정답 Label |

Setosa 50개와 Versicolor 50개를 사용하므로,

```text
Setosa       50개
              +
Versicolor   50개
────────────────
총           100개
```

의 데이터가 만들어집니다.

---

### 💡 핵심 정리

```text
전체 Iris Dataset
150개 / 3 Classes

        │
        │ Boolean Mask
        │
        ├── Setosa (0)       → 선택 ✓ 50개
        │
        ├── Versicolor (1)   → 선택 ✓ 50개
        │
        └── Virginica (2)    → 제외 ✗ 50개
        │
        ▼

이진 분류 Dataset
100개 / 2 Classes

X.shape = (100, 4)
Y.shape = (100,)
```

> **왜 2개의 Class만 사용할까?**  
> 이번 실습의 목적은 퍼셉트론의 가장 기본적인 **두 클래스 간 분류 원리**를 이해하는 것입니다.  
> 따라서 3개의 Iris 품종 중 Setosa와 Versicolor만 선택하여 이진 분류 문제로 구성합니다.

---

### 다음 단계

현재 정답 Label은 다음과 같이 구성되어 있습니다.

```text
Setosa       → 0
Versicolor   → 1
```

다음 단계에서는 퍼셉트론의 출력 표현에 맞추어 Label을 **`+1`, `-1`** 형태로 변환합니다.

---

---

## 5. Label 변환

앞에서 추출한 데이터의 Label은 다음과 같이 구성되어 있습니다.

```text
Setosa       → 0
Versicolor   → 1
```

이번 퍼셉트론 실습에서는 두 클래스를 `+1`과 `-1`로 표현하여 이진 분류를 수행합니다.

따라서 기존 Label을 다음과 같이 변환합니다.

```text
Setosa       : 0 → +1
Versicolor   : 1 → -1
```

---

### 5.1 Label 변환하기

NumPy의 `np.where()` 함수를 이용하여 Label을 변환합니다.

```python
Y = np.where(Y == 0, 1, -1)
```

`np.where()`는 **조건에 따라 서로 다른 값을 선택**하는 함수입니다.

기본 형태는 다음과 같습니다.

```python
np.where(조건, 조건이 True일 때 값, 조건이 False일 때 값)
```

따라서 이번 코드

```python
np.where(Y == 0, 1, -1)
```

는 다음과 같이 동작합니다.

| 기존 Label | 조건 `Y == 0` | 변환 Label |
|---:|:---:|---:|
| `0` (Setosa) | `True` | `+1` |
| `1` (Versicolor) | `False` | `-1` |

즉,

```text
기존 Y

[0, 0, 0, ..., 0, 1, 1, 1, ..., 1]

                    ↓
              np.where()
                    ↓

변환된 Y

[1, 1, 1, ..., 1, -1, -1, -1, ..., -1]
```

이 됩니다.

---

### 5.2 변환 결과 확인

```python
print(Y)
```

실행 결과는 다음과 같은 형태입니다.

```text
[ 1  1  1  1  1 ...  1
 -1 -1 -1 -1 -1 ... -1]
```

총 100개의 데이터 중

```text
Setosa       50개 → +1
Versicolor   50개 → -1
```

로 변환됩니다.

---

### 5.3 왜 `+1`, `-1`로 변환할까?

이번 실습에서 퍼셉트론은 입력 데이터의 가중합(Weighted Sum)을 계산한 후 Step Function을 이용하여 클래스를 결정합니다.

퍼셉트론의 가중합은 다음과 같습니다.

```text
z = x₁w₁ + x₂w₂ + x₃w₃ + x₄w₄ + b
```

그리고 활성화 함수에서 `z = 0`을 기준으로 두 클래스를 구분합니다.

```text
             Step Function

                   z
                   │
        z < 0      │      z ≥ 0
                   │
          ↓        │        ↓
         -1        │       +1
                   │
     Versicolor    │      Setosa
```

코드로 표현하면 다음과 같습니다.

```python
y_pred = np.where(z >= 0, 1, -1)
```

따라서 **실제 정답 `Y`와 퍼셉트론의 예측값 `y_pred`를 동일한 `+1`, `-1` 형식으로 맞추기 위해** Label을 변환합니다.

> 💡 `0`과 `1`을 반드시 `-1`과 `+1`로 바꿔야 모든 이진 분류가 가능한 것은 아닙니다.  
> 이번 실습에서는 **퍼셉트론의 Step Function과 출력 구조를 쉽게 이해하기 위해** `{-1, +1}` 형태를 사용합니다.

---

### 💡 핵심 정리

```text
Iris 원본 Label

Setosa       → 0
Versicolor   → 1

        │
        │ np.where()
        ▼

Perceptron Label

Setosa       → +1
Versicolor   → -1

        │
        ▼

Step Function의 출력과 비교

z ≥ 0 → +1
z < 0 → -1
```

---
---
## 6. 데이터 표준화

퍼셉트론에 데이터를 입력하기 전에 `StandardScaler`를 이용하여 각 Feature의 값을 **표준화(Standardization)** 합니다.

Iris Dataset의 Feature들은 서로 다른 값과 분포를 가지고 있습니다.

| Feature | 의미 | 단위 |
|---|---|---|
| Sepal Length | 꽃받침 길이 | cm |
| Sepal Width | 꽃받침 너비 | cm |
| Petal Length | 꽃잎 길이 | cm |
| Petal Width | 꽃잎 너비 | cm |

Feature마다 값의 분포가 다르면 퍼셉트론의 가중합 계산에서  
특정 Feature가 상대적으로 큰 영향을 줄 수 있습니다.

따라서 각 Feature를 **비슷한 Scale**로 변환합니다.

---

### 6.1 표준화(Standardization)란?

표준화는 각 데이터에서 해당 Feature의 **평균을 빼고 표준편차로 나누는 과정**입니다.

```text
                 x - 평균
표준화된 값 = ─────────────
                 표준편차
```

표준화를 수행하면 각 Feature는 대체로 다음과 같은 분포를 갖게 됩니다.

```text
평균      ≈ 0
표준편차  ≈ 1
```

즉, 데이터의 단위나 값의 크기가 서로 달라도  
비슷한 기준에서 비교하고 연산할 수 있도록 변환합니다.

---

### 6.2 StandardScaler 사용하기

Scikit-learn에서 제공하는 `StandardScaler`를 사용합니다.

```python
scaler = StandardScaler()

X = scaler.fit_transform(X)
```

각 코드의 역할은 다음과 같습니다.

| 코드 | 역할 |
|---|---|
| `StandardScaler()` | 표준화를 수행할 객체 생성 |
| `fit(X)` | X의 각 Feature별 평균과 표준편차 계산 |
| `transform(X)` | 계산된 평균과 표준편차를 이용하여 X 변환 |
| `fit_transform(X)` | `fit()` + `transform()`을 한 번에 수행 |

즉,

```python
X = scaler.fit_transform(X)
```

은 다음 두 코드를 한 번에 수행한 것과 같습니다.

```python
scaler.fit(X)
X = scaler.transform(X)
```

---

### 6.3 표준화 전 데이터

표준화 전 Iris 데이터의 일부를 확인해 봅시다.

```python
print(X[:5])
```

예를 들어 데이터는 다음과 같은 실제 측정값을 가지고 있습니다.

```text
[[5.1 3.5 1.4 0.2]
 [4.9 3.0 1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.0 3.6 1.4 0.2]]
```

하나의 행은 하나의 붓꽃을 의미합니다.

```text
[5.1, 3.5, 1.4, 0.2]
  │    │    │    │
  │    │    │    └─ Petal Width
  │    │    └────── Petal Length
  │    └─────────── Sepal Width
  └──────────────── Sepal Length
```

---

### 6.4 표준화 수행

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

표준화를 수행하면 원래의 cm 단위 측정값이  
**평균 0, 표준편차 1을 기준으로 한 값**으로 변환됩니다.

따라서 표준화 이후에는 값이 음수가 될 수도 있습니다.

```text
원본 데이터                     표준화 데이터

5.1                              -0.58
4.9        StandardScaler        -0.89
4.7   ─────────────────────→     -1.21
5.4                               0.18
...
```

> 💡 **음수가 나와도 잘못된 것이 아닙니다.**  
> 평균보다 작은 값은 음수, 평균보다 큰 값은 양수로 표현될 수 있습니다.

---

### 6.5 표준화 결과 확인

실제로 평균과 표준편차를 확인할 수 있습니다.

```python
print("평균 :", X.mean(axis=0))
print("표준편차 :", X.std(axis=0))
```

결과는 대체로 다음과 같습니다.

```text
평균      ≈ [0. 0. 0. 0.]
표준편차  ≈ [1. 1. 1. 1.]
```

여기서

```python
axis=0
```

은 **각 열(Feature)을 기준으로 계산**한다는 의미입니다.

즉,

```text
                Feature
           ↓      ↓      ↓      ↓
X =      [ x₁     x₂     x₃     x₄ ]
         [ x₁     x₂     x₃     x₄ ]
         [ x₁     x₂     x₃     x₄ ]
           ↓      ↓      ↓      ↓
         평균0   평균0   평균0   평균0
         표준편차1 ...
```

처럼 각각의 Feature가 독립적으로 표준화됩니다.

---
---
## 7. Weight와 Bias 초기화

퍼셉트론은 각각의 입력값에 가중치(Weight)를 곱한 후, 모든 값을 더하고 바이어스(Bias)를 추가하여 하나의 값을 계산합니다.

Iris 데이터는 하나의 샘플마다 4개의 Feature를 가지고 있으므로 퍼셉트론에도 4개의 Weight가 필요합니다.

```text
x₁ : Sepal Length
x₂ : Sepal Width
x₃ : Petal Length
x₄ : Petal Width
```

---

### 7.1 Weight 초기화

이번 실습에서는 퍼셉트론의 연산 과정을 쉽게 확인하기 위해 4개의 Weight를 모두 `0.5`로 초기화합니다.

```python
w = np.array([0.5, 0.5, 0.5, 0.5])
```

각 Weight는 하나의 Feature와 연결됩니다.

| 입력 Feature | 입력 | Weight |
|---|:---:|:---:|
| Sepal Length | `x₁` | `w₁ = 0.5` |
| Sepal Width | `x₂` | `w₂ = 0.5` |
| Petal Length | `x₃` | `w₃ = 0.5` |
| Petal Width | `x₄` | `w₄ = 0.5` |

즉,

```text
x₁ ── w₁ = 0.5 ──┐
x₂ ── w₂ = 0.5 ──┤
x₃ ── w₃ = 0.5 ──┼──→ 가중합 z
x₄ ── w₄ = 0.5 ──┤
                  │
b  ───── 0.1 ─────┘
```

---

### 7.2 Weight란?

Weight(가중치)는 각 입력 Feature가 퍼셉트론의 출력에 얼마나 영향을 미치는지를 나타내는 값입니다.

퍼셉트론에서는 각 입력값 `x`에 해당 Weight `w`를 곱합니다.

```text
x₁ × w₁
x₂ × w₂
x₃ × w₃
x₄ × w₄
```

Weight의 절댓값이 클수록 해당 Feature가 가중합 `z`에 미치는 영향도 커집니다.

> 💡 실제 퍼셉트론의 학습(Training)에서는 정답과 예측 결과를 이용하여 
> Weight를 반복적으로 수정하면서 적절한 값을 찾아갑니다.

이번 단계에서는 아직 학습하지 않고, 연산 과정을 확인하기 위해  
임의의 초기값 `0.5`를 사용합니다.

---

### 7.3 Bias 초기화

Bias는 다음과 같이 설정합니다.

```python
b = 0.1
```

Bias(편향)는 입력값과 Weight의 곱을 모두 더한 결과에 추가되는 값입니다.

```text
입력과 Weight의 가중합
        ↓
x₁w₁ + x₂w₂ + x₃w₃ + x₄w₄
        ↓
      + Bias
        ↓
        z
```

Bias는 퍼셉트론의 결정 기준(Decision Boundary)을 조정하는 역할을 합니다.

---

### 7.4 퍼셉트론의 기본 연산

퍼셉트론의 가중합(Weighted Sum)은 다음과 같이 계산합니다.

```text
z = x₁w₁ + x₂w₂ + x₃w₃ + x₄w₄ + b
```

이번 실습의 초기값을 대입하면,

```text
z = 0.5x₁ + 0.5x₂ + 0.5x₃ + 0.5x₄ + 0.1
```

이 됩니다.

Python에서는 이 연산을 다음과 같이 간단하게 표현할 수 있습니다.

```python
z = X @ w + b
```

여기서 `@`는 NumPy에서 행렬 곱(Matrix Multiplication)을 의미합니다.

---

### 7.5 X와 Weight의 크기 확인

현재 입력 데이터 `X`에는 100개의 Iris 데이터가 있고  
각 데이터는 4개의 Feature를 가지고 있습니다.

```python
print(X.shape)
print(w.shape)
```

결과:

```text
X.shape → (100, 4)
w.shape → (4,)
```

행렬 연산의 구조는 다음과 같습니다.

```text
          X                  w                z

      (100 × 4)            (4 × 1)         (100 × 1)

[x₁ x₂ x₃ x₄]             [w₁]
[x₁ x₂ x₃ x₄]             [w₂]
[x₁ x₂ x₃ x₄]      @      [w₃]      →     [z₁]
[     ...     ]            [w₄]            [z₂]
[     ...     ]                              ...
                                             z₁₀₀
```

즉, **100개의 Iris 데이터 각각에 대해 하나의 `z` 값**이 계산됩니다.

---

### 💡 핵심 정리

```text
입력 Feature
x₁  x₂  x₃  x₄
│   │   │   │
×   ×   ×   ×
│   │   │   │
w₁  w₂  w₃  w₄
 \   \   |   /
  └── 가중합 ──┘
        +
      Bias
        │
        ▼
        z
```

| 요소 | 의미 | 이번 실습 |
|---|---|---|
| `X` | 입력 데이터 | 100 × 4 |
| `w` | Weight | `[0.5, 0.5, 0.5, 0.5]` |
| `b` | Bias | `0.1` |
| `z` | 가중합 결과 | 각 Sample당 1개 |

---

### 다음 단계

이제 입력 데이터 `X`, Weight `w`, Bias `b`가 준비되었습니다.

다음 단계에서는 실제로

```python
z = X @ w + b
```

를 계산하여 퍼셉트론의 **순전파(Feed-Forward)** 과정을 수행합니다.

## 8. Feed-Forward

모든 데이터에 대해 가중합을 계산합니다.

```python
z = X @ w + b
```

여기서 `@`는 행렬 곱셈을 의미합니다.

```text
X : (100, 4)
w : (4,)

결과

z : (100,)
```

즉, 100개의 데이터 각각에 대해 하나의 `z` 값이 계산됩니다.

---

## 9. Step Function

계산된 `z`를 Step Function에 입력합니다.

```python
y_pred = np.where(z >= 0, 1, -1)
```

즉,

```text
z >= 0 → +1
z < 0  → -1
```

로 분류합니다.

---

## 10. 정확도 확인

```python
classified = np.sum(y_pred == Y)

accuracy = np.mean(y_pred == Y)

print("정분류 개수:", classified)
print(f"Accuracy: {accuracy:.3f}")
```

---

## 11. Loss 확인

실제값과 예측값의 차이를 확인하기 위해
MSE와 MAE를 계산합니다.

```python
mse = np.mean((Y - y_pred) ** 2)

mae = np.mean(np.abs(Y - y_pred))

print(f"MSE = {mse:.3f}")
print(f"MAE = {mae:.3f}")
```

---