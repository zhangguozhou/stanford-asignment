import pickle as p 
import numpy as np 
import os

#载入数据
def load_cifar_batch(file):
	with open(file,'rb') as fo:
		dict = p.load(fo, encoding='bytes')
		X=dict[b'data']
		Y=dict[b'labels']
	return X,Y

def load_CIFAR10(addr):
	Xtr = []
	Ytr = []
	for i in range(1,6):
		f = os.path.join(addr,'data_batch_'+str(i))
		X,Y=load_cifar_batch(f)
		Xtr=np.append(Xtr,X)
		Ytr=np.append(Ytr,Y)
	Xtr=Xtr.reshape(50000,32,32,3)
	Ytr=np.array(Ytr)
	Xte,Yte=load_cifar_batch(os.path.join(addr,'test_batch'))
	Xte=Xte.reshape(10000,32,32,3)
	return Xtr, Ytr, Xte, Yte
	
def KNN(input,X_train,Y_train,labels,k):
	distance=[]
	for i in range(len(X_train)):
		d=np.sqrt(np.sum(np.square(input-X_train[i])))
		distance=np.append(distance, d)

	sortedindex=distance.argsort()

	vote=np.zeros(len(labels),int)
	for j in range(k):
		vote[int(Y_train[sortedindex[j]])] = vote[Y_train[sortedindex[j]]]+1 
	sortedvote=vote.argsort()
	return labels[sortedvote[-1]]

if __name__=="__main__":
	labels=['0','1','2','3','4','5','6','7','8','9']
	X_train,Y_train,X_test,Y_test=load_CIFAR10('C:\\Users\\张国洲\\Desktop\\stanford_asignment\\asignment1-Q1\\cifar-10-batches-py')
	for i in range(10,20):
		input = X_test[i]
		output = KNN(input,X_train,Y_train,labels,7)

		print(output,Y_test[i])