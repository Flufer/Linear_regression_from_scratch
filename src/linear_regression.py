class LinearRegression:
    """
    Линейная регрессия с градиентным спуском
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.w = 0  # вес
        self.b = 0  # смещение
        # self.loss_history = []

    def fit(self, X, y):
        """
        Обучение модели

        Args:
            X (list): Список признаков
            y (list): Список целевых значений
        """
        n = len(X)

        for iteration in range(self.n_iterations):
            # Прямое распространение
            y_pred = [self.w * x + self.b for x in X]

            # Вычисление градиентов
            dw = (-2 / n) * sum(X[i] * (y[i] - y_pred[i]) for i in range(n))
            db = (-2 / n) * sum(y[i] - y_pred[i] for i in range(n))

            # Обновление параметров
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Сохранение ошибки
            # loss = sum((y[i] - y_pred[i]) ** 2 for i in range(n)) / n
            # self.loss_history.append(loss)

        return self

    def predict(self, X):
        """
        Предсказание

        Args:
            X (list или число): Признаки для предсказания

        Returns:
            list или число: Предсказанные значения
        """
        if isinstance(X, list):
            return [self.w * x + self.b for x in X]
        return self.w * X + self.b

    def get_params(self):
        return {'w': self.w, 'b': self.b}
    