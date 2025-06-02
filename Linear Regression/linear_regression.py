import numpy as np

class LinearRegression:
    def __init__(self,learning_rate=0.01,n_iteration=1000):
        self.learning_rate = learning_rate
        self.n_iteration = n_iteration
        self.theta = None

    def gradient_descent_fit(self,X,y):
        X_b = np.c_[np.ones((X.shape[0],1)),X]
        m = X_b.shape[0]

        self.theta = np.random.randn(X_b.shape[1],1)

        # This is not vectorizes version which is pretty slow
        # gradients = np.zeros((n+1, 1))
        # for j in range(n+1):
        #     for i in range(m):
        #         gradients[j] += (prediction[i] - y[i]) * X_b[i, j]
        # gradients /= m

        for _ in range(self.n_iteration):
            gradient = (1/m) * X_b.T.dot(X_b.dot(self.theta) - y)
            self.theta -= self.learning_rate * gradient
        
    def normal_equation_fit(self,X,y):
        X_b = np.c_[((X.shape[0],1)),X]
        m = X_b.shape[0]

        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    def predict(self,X):
        X_b = np.c_[np.ones((X.shape[0],1)),X]
        return X_b.dot(self.theta)
    
    def get_params(self):
        return {
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iteration,
            "theta": self.theta
        }
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot  # R² score