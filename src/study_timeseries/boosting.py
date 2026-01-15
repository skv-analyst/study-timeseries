"""
Реализация алгоритмов Градиентного Бустинга.

Этот модуль содержит основной класс-эстиматор, совместимый с scikit-learn API.
Он реализует алгоритм построения ансамбля решающих деревьев, обучаемых
на антиградиентах выбранной функции потерь.

Основные классы:
    - SimpleBoosting(BaseEstimator).
    - SimpleGradientBoostingRegressor(BaseEstimator)

Особенности:
    - Поддержка кастомных функций потерь (через losses.py).
    - Визуализация процесса обучения (через историю лоссов).

Зависимости:
    - sklearn.base.BaseEstimator
    - sklearn.tree.DecisionTreeRegressor
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from .losses import (
    loss_mse, antigrad_mse,
    loss_mae, antigrad_mae,
    loss_huber, antigrad_huber,
    loss_rmsle, antigrad_rmsle,
    loss_quantile, antigrad_quantile
)


class SimpleBoosting(BaseEstimator):
    def __init__(self,
                 n_estimators=3,
                 max_depth=1,
                 random_state=42,
                 debug=True
                 ):
        """
        Args:
            n_estimators (int): Количество деревьев (итераций бустинга)
            max_depth (int): Максимальная глубина одного дерева
            random_state (int): Сид для воспроизводимости
            debug (bool): Сохранять ли промежуточные результаты
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.debug = debug

        # Список обученных "слабых" моделей
        self.trees = []

        # Логи для отладки
        if self.debug:
            self.residuals_history = []  # История остатков
            self.trees_predictions_history = []  # История прогнозов каждого дерева

    def calculate_residuals(self, y_true, current_prediction):
        """ Считаем разницу между фактом и тем, что мы уже напредсказывали """
        return y_true - current_prediction

    def fit(self, X, y):
        """
        Args:
            X: Матрица признаков
            y: Вектор целевой переменной
        """
        # 0. Инициализация: начинаем с нулевого прогноза для всех наблюдений
        # (или можно начать со среднего по y, но для простоты пусть будет 0)
        current_ensemble_prediction = np.zeros(X.shape[0])

        for iter_num in range(self.n_estimators):

            # 1. Считаем, где мы ошибаемся сейчас
            current_residuals = self.calculate_residuals(y, current_ensemble_prediction)

            if self.debug:
                self.residuals_history.append(current_residuals)

            # 2. Создаем новое дерево
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state
            )

            # 3. Обучаем дерево предсказывать текущую ошибку
            tree.fit(X, current_residuals)

            # 4. Получаем прогноз ("поправку") от этого дерева
            this_tree_prediction = tree.predict(X)

            if self.debug:
                self.trees_predictions_history.append(this_tree_prediction)

            # 5. Сохраняем модель
            self.trees.append(tree)

            # 6. Обновляем общий прогноз ансамбля
            # добавляем поправку к тому, что уже было
            current_ensemble_prediction += this_tree_prediction

        # Возвращаем сам экземпляр класса
        return self

    def predict(self, X):
        # Итоговый прогноз — это сумма прогнозов всех деревьев
        final_prediction = np.zeros(X.shape[0])

        for tree in self.trees:
            final_prediction += tree.predict(X)

        return final_prediction


class SimpleGradientBoostingRegressor(BaseEstimator):
    def __init__(self,
                 n_estimators=100,
                 max_depth=3,
                 learning_rate=0.1,
                 loss='mse',
                 ath_quantile=0.5,
                 constant_init='zero',
                 random_state=42,
                 debug=True):
        """
        Args:
            n_estimators (int): Количество деревьев.
            max_depth (int): Глубина дерева.
            learning_rate (float): Темп обучения (shrinkage).
            loss (str): Функция потерь.
            ath_quantile (float): Параметр для Huber/Quantile.
            constant_init (str): Инициализация ('zero', 'mean', 'median').
            random_state (int): Сид для воспроизводимости деревьев.
            debug (bool): Сохранять историю для графиков.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_depth = max_depth
        self.random_state = random_state
        self.ath_quantile = ath_quantile
        self.constant_init = constant_init
        self.debug = debug

        # Состояние модели
        self.trees = []
        self.base_prediction_value = 0
        self.loss_history = []  # История падения ошибки на Train

        # Отладка
        if self.debug:
            self.residuals_history = []
            self.trees_predictions_history = []

    # --- Вспомогательная логика (Маршрутизация в losses.py) ---

    def _get_initial_prediction(self, y):
        if self.constant_init == 'zero':
            return 0.0
        elif self.constant_init == 'mean':
            return np.mean(y)
        elif self.constant_init == 'median':
            return np.median(y)
        else:
            raise ValueError(f"Unknown constant_init: {self.constant_init}")

    def _calculate_antigradient(self, y_true, y_pred):
        if self.loss == 'mse':
            return antigrad_mse(y_true, y_pred)
        elif self.loss == 'mae':
            return antigrad_mae(y_true, y_pred)
        elif self.loss == 'rmsle':
            return antigrad_rmsle(y_true, y_pred)
        elif self.loss == 'huber':
            diff = y_true - y_pred
            delta = np.quantile(np.abs(diff), self.ath_quantile)
            return antigrad_huber(y_true, y_pred, delta)
        elif self.loss == 'quantile':
            return antigrad_quantile(y_true, y_pred, self.ath_quantile)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def _calculate_loss_value(self, y_true, y_pred):
        # Используем для записи в историю обучения
        if self.loss == 'mse':
            return np.mean(loss_mse(y_true, y_pred))
        elif self.loss == 'mae':
            return np.mean(loss_mae(y_true, y_pred))
        elif self.loss == 'rmsle':
            return np.mean(loss_rmsle(y_true, y_pred))
        elif self.loss == 'huber':
            diff = y_true - y_pred
            delta = np.quantile(np.abs(diff), self.ath_quantile)
            return np.mean(loss_huber(y_true, y_pred, delta))
        elif self.loss == 'quantile':
            return np.mean(loss_quantile(y_true, y_pred, self.ath_quantile))
        return np.inf

    # --- Основной цикл обучения ---

    def fit(self, X, y):
        # 0. Сброс состояния
        self.trees = []
        self.loss_history = []
        if self.debug:
            self.residuals_history = []
            self.trees_predictions_history = []

        # 1. Базовый прогноз (F_0)
        self.base_prediction_value = self._get_initial_prediction(y)
        current_prediction = np.full(y.shape[0], self.base_prediction_value)

        # 2. Цикл по деревьям
        for iter_num in range(self.n_estimators):

            # A. Считаем антиградиент (на что учить дерево)
            # Вся магия loss-функций спрятана тут
            pseudo_residuals = self._calculate_antigradient(y, current_prediction)

            if self.debug:
                self.residuals_history.append(pseudo_residuals)

            # B. Создаем и обучаем дерево
            # Обрати внимание: X и pseudo_residuals передаем целиком, без subsample
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=(self.random_state + iter_num)
            )
            tree.fit(X, pseudo_residuals)

            # C. Делаем прогноз дерева (h_m)
            this_tree_prediction = tree.predict(X)

            if self.debug:
                self.trees_predictions_history.append(this_tree_prediction)

            # D. Обновляем общий прогноз ансамбля (F_m = F_{m-1} + lr * h_m)
            current_prediction += self.learning_rate * this_tree_prediction

            # E. Сохраняем модель и метрики
            self.trees.append(tree)

            current_loss = self._calculate_loss_value(y, current_prediction)
            self.loss_history.append(current_loss)

        return self

    def predict(self, X):
        # 1. Начинаем с базы
        pred = np.full(X.shape[0], self.base_prediction_value)

        # 2. Суммируем вклады всех деревьев
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)

        return pred