import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
import matplotlib.pyplot as plt


def load_data():
    """
    read data from csv
    """
    data = np.genfromtxt('pima-indians-diabetes.data.csv', delimiter=",")
    return data[:, 0:8], data[:, 8:]


def standardize(X):
    """
    standardize X to zero mean and unit variance
    """
    scaler = StandardScaler().fit(X)
    standardizedX = scaler.transform(X)
    return standardizedX


def shuffle(a, b):
    """
    shuffle two array with same order together
    """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def get_train_test():
    """
    split data to train and test
    """
    X, y = load_data()
    X = standardize(X)

    np.random.seed(10)
    shuffle_X, shuffle_y = shuffle(X, y)

    train_size = int(0.7 * len(X))
    train_X = shuffle_X[0:train_size, :]
    test_X = shuffle_X[train_size:, :]
    train_y = shuffle_y[0:train_size, :]
    test_y = shuffle_y[train_size:, :]

    return train_X, train_y, test_X, test_y


def pred_val(theta, X, hard=True):
    """
    give model prediction for X with given theta.
    hard indicate whether output is value or possibility
    """
    pred_prob = logistic_val_func(theta, X)
    pred_value = np.where(pred_prob > 0.5, 1, 0)
    if hard:
        return pred_value
    else:
        return pred_prob


def logistic_grad_func(theta, x, y):
    """
    compute grad for given x,y
    """
    diff = logistic_val_func(theta, x) - y
    pad_x = np.c_[np.ones(x.shape[0]), x]
    grad = (np.dot(pad_x.T, diff) / x.shape[0]).T
    return grad


def sigmoid(x):
    """
    sigmoid function
    """
    sig = 1.0 / (1 + np.exp(-x))
    return sig


def logistic_val_func(theta, x):
    """
    compute output
    """
    return sigmoid(np.dot(np.c_[np.ones(x.shape[0]), x], theta.T))


def logistic_cost_func(theta, x, y):
    """
    compute negative log likelihood
    """
    # compute cost (loss)
    y_hat = logistic_val_func(theta, x)
    # compute loss
    cost = np.sum(np.multiply(y, np.log(y_hat)) + np.multiply(1.0 - y, np.log(1.0 - y_hat)))
    cost *= 1.0 / x.shape[0]
    return -cost


def logistic_mini_batch_grad_desc(theta, X_train, Y_train,
                                  lr=0.03,
                                  epochs=500,
                                  batch_size=64,
                                  momentum=0.5,
                                  Adam=False,
                                  b1=0.9,
                                  b2=0.999,
                                  eps=1e-8,
                                  converge_change=.00001,
                                  verbose = False,
                                  ):
    m = len(X_train)
    n_batches = m // batch_size

    cost_iter = []
    cost = logistic_cost_func(theta, X_train, Y_train)
    cost_iter.append(cost)
    mt = 0
    vt = 0
    t = 1
    for epoch in range(epochs):
        pre_cost = cost
        old_dtheta = 0
        for batch_i in range(n_batches):
            t += 1
            batch_X = X_train[batch_i * batch_size:(batch_i + 1) * batch_size]
            batch_y = Y_train[batch_i * batch_size:(batch_i + 1) * batch_size]
            # compute gradient
            grad = logistic_grad_func(theta, batch_X, batch_y)
            if Adam:
                mt = b1 * mt + (1 - b1) * grad
                vt = b2 * vt + (1 - b2) * np.power(grad, 2)
                bcmt = mt / (1 - np.power(b1, t))
                bcvt = vt / (1 - np.power(b2, t))
                theta -= np.divide(lr * bcmt, np.sqrt(bcvt) + eps)
            else:
                dtheta = lr * grad + momentum * old_dtheta
                theta -= dtheta
                old_dtheta = dtheta
        if verbose:
            print "epoch:{0} cost:{1}".format(epoch, cost_iter[-1])

        cost = logistic_cost_func(theta, X_train, Y_train)
        cost_iter.append( cost)
        cost_change = abs(cost - pre_cost)
        if cost_change < converge_change:
            break
        X_train, Y_train = shuffle(X_train, Y_train)
    return theta, cost_iter


def my_logistic_regression_no_momentum():
    print "I'm naive model without any optimization"
    X_train, Y_train, X_test, Y_test = get_train_test()
    theta = np.random.rand(1, X_train.shape[1] + 1)
    fitted_theta, cost_iter = logistic_mini_batch_grad_desc(
        theta, X_train, Y_train, lr=0.05, epochs=50000, momentum=0)
    print('Accuracy: {}'.format(np.sum((pred_val(fitted_theta, X_test) == Y_test)) * 1.0 / X_test.shape[0]))
    return cost_iter


def my_logistic_regression_with_momentum():
    print "I'm model with momentum"
    X_train, Y_train, X_test, Y_test = get_train_test()
    theta = np.random.rand(1, X_train.shape[1] + 1)
    fitted_theta, cost_iter = logistic_mini_batch_grad_desc(theta, X_train, Y_train, lr=0.05, epochs=50000)
    print('Accuracy: {}'.format(np.sum((pred_val(fitted_theta, X_test) == Y_test)) * 1.0 / X_test.shape[0]))
    return cost_iter


def my_logistic_regression_adam():
    print "I'm model without adam optimization"
    X_train, Y_train, X_test, Y_test = get_train_test()
    theta = np.random.rand(1, X_train.shape[1] + 1)
    fitted_theta, cost_iter = logistic_mini_batch_grad_desc(theta, X_train, Y_train, lr=0.05, epochs=50000, Adam=True)
    print('Accuracy: {}'.format(np.sum((pred_val(fitted_theta, X_test) == Y_test)) * 1.0 / X_test.shape[0]))
    return cost_iter


def sklearn_logistic_regression():
    print "I'm sklearn model"
    X_train, Y_train, X_test, Y_test = get_train_test()
    regressor = linear_model.LogisticRegression()
    regressor.fit(X_train, np.ravel(Y_train))
    print('Accuracy:{}'.format(np.sum((regressor.predict(X_test) == np.ravel(Y_test))) * 1.0 / X_test.shape[0]))


if __name__ == '__main__':
    a = my_logistic_regression_no_momentum()
    b = my_logistic_regression_with_momentum()
    c = my_logistic_regression_adam()
    plt.plot(a, label='naive model')
    plt.plot(b, label='momentum')
    plt.plot(c, label='adam')
    plt.ylabel("loss")
    plt.xlabel("iteration")
    plt.legend()
    plt.show()
    sklearn_logistic_regression()
