"""
Модуль функций потерь и их антиградиентов.

Этот модуль содержит реализации различных функций потерь (Loss Functions)
и их производных (антиградиентов), необходимых для обучения градиентного бустинга.

Содержание:
    - MSE (Mean Squared Error): Квадратичная ошибка.
    - MAE (Mean Absolute Error): Абсолютная ошибка.
    - Huber Loss: Комбинация MSE и MAE (устойчива к выбросам).
    - Quantile Loss: Для квантильной регрессии.

Каждая функция представлена в двух вариантах:
    1. `loss_*`: Вычисляет значение ошибки (скаляр или вектор).
    2. `antigrad_*`: Вычисляет антиградиент (вектор сдвигов).

Пример использования:
    >>> import numpy as np
    >>> from study_timeseries.losses import loss_mae, loss_mse
    >>> y_true = np.array([100, 200])
    >>> y_pred = np.array([110, 190])
    >>> loss_mse(y_true, y_pred)
    array([100., 100.])
"""

import numpy as np

__all__ = [
    "loss_mae", "antigrad_mae",
    "loss_mse", "antigrad_mse",
    "loss_huber", "antigrad_huber",
    "loss_rmsle", "antigrad_rmsle",
    "loss_quantile", "antigrad_quantile",
]


# 1. ФУНКЦИИ ПОТЕРЬ (loss_*)
def loss_mae(y_true, y_pred):
    loss = np.abs(y_true - y_pred)
    return loss


def loss_mse(y_true, y_pred):
    loss = (y_true - y_pred) ** 2
    return loss


def loss_huber(y_true, y_pred, delta):
    error = y_true - y_pred
    loss = np.where(
        np.abs(error) <= delta,
        (error ** 2) / 2,
        delta * np.abs(error) - delta / 2
    )
    return loss


def loss_rmsle(y_true, y_pred):
    # Сжимаем данные логарифмом (log1p это log(1+x), чтобы не было log(0))
    log_true = np.log1p(y_true)
    # clip нужен, чтобы не взять логарифм от отрицательного числа
    log_pred = np.log1p(np.clip(y_pred, 0, None))
    # Считаем квадрат разницы логарифмов
    loss = (log_true - log_pred) ** 2
    return loss


def loss_quantile(y_true, y_pred, q):
    error = y_true - y_pred
    loss = np.where(
        # Если ошибка > 0 (недопрогноз), умножаем на q
        error > 0, q * error,
        # Если ошибка <= 0 (перепрогноз), умножаем на (1-q)
        (1 - q) * np.abs(error)
    )
    return loss


# 2. АНТИГРАДИЕНТЫ (antigrad_*)
def antigrad_mae(y_true, y_pred):
    grad = np.sign(y_true - y_pred)
    return grad


def antigrad_mse(y_true, y_pred):
    grad = 2 * (y_true - y_pred)
    return grad


def antigrad_huber(y_true, y_pred, delta):
    error = y_true - y_pred
    grad = np.where(
        np.abs(error) <= delta, error,
        delta * np.sign(error)
    )
    return grad


def antigrad_rmsle(y_true, y_pred):
    eps = 1e-15
    y_pred_safe = np.clip(y_pred, eps, None)
    log_diff = np.log1p(y_true) - np.log1p(y_pred_safe)
    # Это производная сложной функции (chain rule) от loss_rmsle
    grad = 2 * log_diff / (1 + y_pred_safe)
    return grad


def antigrad_quantile(y_true, y_pred, q):
    # Если факт > прогноз -> возвращаем q
    # Если факт <= прогноз -> возвращаем -(1-q) или q-1
    grad = (
            q * (y_true > y_pred).astype(float) -
            (1 - q) * (y_true <= y_pred).astype(float)
    )

    return grad
