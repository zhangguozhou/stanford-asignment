import numpy as np
import pickle as p
import os 


class SVM(object):

    def __init__(self):
        self.W = None

    def train(self, X_train, Y_labels, reg, delta, learning_rate, batch_num, num_iter):
        num_train = X_train.shape[0]
        num_dim = X_train.shape[1]
        num_class = max(Y_labels) + 1

        if self.W == None:
            self.W = 0.0001 * np.random.randn(num_class, num_dim)

        for i in range(num_iter):
            # 提取batch
            batch_sample = np.random.choice(
                num_train, batch_num, replace=False)
            X_batch = X_train[batch_sample, :]
            Y_batch = Y_labels[batch_sample].astype(int)


            loss, gred = self.SVM_Loss(X_batch, Y_batch, reg, delta)
            if i % 100 == 0:
            	print('loss: ', loss)
            self.W -= learning_rate * gred

    def predict(self, X):
        scores = self.W.dot(X.T)
        pre = np.zeros(X.shape[0])
        pre = np.argmax(scores ,axis = 0)

        return pre

    def SVM_Loss(self, X_train, Y_labels, regularization, delta):
        num_train = X_train.shape[0]
        num_class = max(Y_labels) + 1
        # loss
        scores = self.W.dot(X_train.T)

        correct_class_score = np.zeros(num_train, int)
        for i in range(num_train):
            correct_class_score[i] = scores[Y_labels[i]][i]

        margins = np.maximum(0, scores - correct_class_score + delta)

        for j in range(num_train):
            margins[Y_labels[j]][j] = 0
        loss = np.sum(margins) / num_train + regularization * \
            np.sum(np.square(self.W))

        # gredient
        Dw = np.zeros(margins.T.shape, int)

        Dw[margins.T > 0] = 1
        sum_Dw = np.sum(Dw, axis=1)
        Dw[range(num_train), Y_labels] -= sum_Dw

        gred = Dw.T.dot(X_train) / num_train + regularization * self.W

        return loss, gred


#载入数据
def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = p.load(fo, encoding='bytes')
        X = dict[b'data']
        Y = dict[b'labels']
    return X, Y


def load_CIFAR10(addr):
    Xtr = []
    Ytr = []
    for i in range(1, 6):
        f = os.path.join(addr, 'data_batch_' + str(i))
        X, Y = load_cifar_batch(f)
        Xtr = np.append(Xtr, X)
        Ytr = np.append(Ytr, Y)
    Xtr = Xtr.reshape(50000, 3072)
    Ytr = np.array(Ytr)
    Xte, Yte = load_cifar_batch(os.path.join(addr, 'test_batch'))
    Xte = Xte.reshape(10000, 3072)
    return Xtr, Ytr, Xte, Yte

if __name__ == '__main__':
	X_train,Y_train,X_test,Y_test = load_CIFAR10('C:\\Users\\张国洲\\Desktop\\stanford_asignment\\asignment1-Q2\\cifar-10-batches-py')
	svm = SVM()
	svm.train(X_train,Y_train,5,1,1e-8,300,10000)
	y_pred = svm.predict(X_test)
	accuracy = np.mean(y_pred == Y_test)
	print(accuracy)
	