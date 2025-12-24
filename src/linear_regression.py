class LinearRegression:
    """
    Простейшая логистическая регрессия с градиентным спуском
    """

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weight = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.n_iters):
            y_pred = self.predict(X)
            # вычисляем градиент
            dw = [0.0] * n_features
            db = 0.0
            for i in range(n_samples):
                error = y_pred[i] - y[i]
                for j in range(n_features):
                    dw[j] += error * X[i][j]
                dw += error
            # обновляем веса
            for j in range(n_features):
                self.weight[j] -= self.lr * dw[j] / n_samples
            self.bias -= self.lr * db / n_samples

    def predict(self, X):
        y_pred = []
        for i in X:
            pred = sum(w * xi for w, xi in zip(self.weight, X)) + self.bias
            y_pred.append(pred)
        return y_pred


