import numpy as np
import pickle as p
import os


class softmax(object):

	def __init__(self):
		self.W = None

	def train(self, X_train, Y_labels, reg, delta, learning_rate, batch_num, iter_num):
		num_train = X_train.shape[0]  # 50000
		num_class = max(Y_labels) + 1  # 10
		num_dim = X_train.shape[1]  # 3072

		if self.W == None:
			self.W = 0.0001 * np.random.randn(num_class, num_dim)

		for i in range(iter_num):
			batch_sample = np.random.choice(num_train, batch_num, replace=False)
			X_batch = X_train[batch_sample, :]
			Y_batch = Y_labels[batch_sample].astype(int)
			loss, gred = self.softmax_loss(X_batch,Y_batch, reg, delta)
			self.W -= learning_rate*gred

			if i % 100 == 0:
				print('loss :',loss)

	def softmax_loss(self, X_train, Y_labels, reg, delta):
		num_train = X_train.shape[0]
		num_dim = X_train.shape[1]
		num_class = max(Y_labels)+1

		scores = X_train.dot(self.W.T) #N*C
		exp_scores = np.exp(scores)
		norm_scores = exp_scores / (np.sum(exp_scores,axis = 1)[:,np.newaxis])#N*C

		loss_i = - np.log(norm_scores)
		loss = np.sum(loss_i) / num_train + 0.5 * reg * np.sum(self.W * self.W)
		

		ground_ture = np.zeros(scores.shape)
		ground_ture[range(num_train),Y_labels] = 1

		gred = -1*(ground_ture-norm_scores).T.dot(X_train) / num_train + reg * self.W

		return loss, gred

	def predict(self,X):
		scores = X.dot(self.W.T)

		y_pred = np.zeros(X.shape[0])
		y_pred = np.argmax(scores, axis = 1)

		return y_pred


# 载入数据
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
	X_train,Y_train,X_test,Y_test = load_CIFAR10('C:\\Users\\张国洲\\Desktop\\stanford_asignment\\asignment1-Q3\\cifar-10-batches-py')
	soft_C = softmax()
	soft_C.train(X_train,Y_train,5,1,1e-10,300,5000)
	
	pred = soft_C.predict(X_test)
	accuracy = np.mean(pred == Y_test)
	print(accuracy)

