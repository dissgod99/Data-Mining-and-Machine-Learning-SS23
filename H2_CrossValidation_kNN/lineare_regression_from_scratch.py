import numpy as np


X = np.array([200, 110, 400, 350, 400], dtype=np.float32)
y = np.array([250, 230, 340, 300, 550], dtype=np.float32) 

# y_i = m * x_i + b

def calculate__linear_regression_params(X: np.ndarray, y: np.ndarray) -> tuple[np.float32, np.float32]:
    x_mean = np.mean(X)
    y_mean = np.mean(y)

    x_diff = X - x_mean
    y_diff = y - y_mean

    sum_above = np.sum(x_diff * y_diff)
    sum_square = np.sum(x_diff ** 2)

    m = sum_above / sum_square
    b = y_mean - m * x_mean
    return m, b

def linear_regression(x: np.float32, m: np.float32, b: np.float32):
    return m * x + b

def split_train_test(X: np.ndarray, y: np.ndarray, test_size: np.float32, part=True) -> tuple[np.ndarray, np.ndarray]:
    #take the testset from the beginning
    limit_test = int(len(X) * test_size) 
    if part:
        X_train = X[:-limit_test]
        y_train = y[:-limit_test]
        X_test = X[-limit_test:]
        y_test = y[-limit_test:]
    else:
        X_train = X[limit_test:]
        y_train = y[limit_test:]
        X_test = X[:-limit_test]
        y_test = y[:-limit_test]
    return X_train, y_train, X_test, y_test


# Linear regression parameters


m, b = calculate__linear_regression_params(X=X, y=y)
print(f"m == {m}\nb == {b}")
print(f"y == {m:.2} * x + {b:.2f}")

X_train, y_train, X_test, y_test = split_train_test(X=X, y=y, test_size=0.2)

print("Train Linear Regression")
m, b = calculate__linear_regression_params(X=X_train, y=y_train)
print(X_train, y_train)
print(f"m == {m}\nb == {b}")
print(f"y == {m:.2} * x + {b:.2f}")

print(f"Test x == {X_test}: y_pred == {linear_regression(X_test, m=m, b=b)}")










