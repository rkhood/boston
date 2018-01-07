'''
Fits linear model to Boston house prices dataset
'''

from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
import mle


def read_data():

        data = load_boston()
        return standard_scale(data.data), data.target, data.feature_names


def standard_scale(x):

        return (x - x.mean(0)) / x.std(0)


def mse(y_true, y_pred):

        return ((y_true - y_pred)**2).mean()


if __name__ == '__main__':

        x, y, names = read_data()

        split = 2 * len(x) // 3
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]

        # my linear model
        w = mle.fit(x_train, y_train, lam=2)
        y_pred = mle.predict(w, x_test)

        print('my linear model')
        for n, i in zip(names, w[1:]):
                if abs(i) > 1e-3:
                        print(n, i)
        print('mse:', mse(y_test, y_pred))

        # sklearn's linear model
        line = Lasso()
        line.fit(x_train, y_train)

        print('\nsklean linear model')
        for n, i in zip(names, line.coef_):
                if abs(i) > 1e-3:
                        print(n, i)
        print('mse:', mse(y_test, line.predict(x_test)))
