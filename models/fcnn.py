#NOTE. These functions have been designed for the roundtrip model. In particular, the backward model. The X and Y are inverted 

import time

import torch
import torch.nn as nn
import torch.optim as optim

def fit_fc_nn(X_train,y_train,X_test,y_test,
				n_epochs,batch_size,print_freq,learning_rate):#, hyperparameters):
	#normalize target
	X_train_norm=norm_data(X_train,X_train)
	X_test_norm=norm_data(X_test,X_train) #normalized by the SAME values
	
	#convert data to tensors (use normalized values)
	X_train_tensor=torch.tensor(X_train_norm.values,dtype=torch.float32)
	y_train_tensor=torch.tensor(y_train.values,dtype=torch.float32)

	X_test_tensor=torch.tensor(X_test_norm.values,dtype=torch.float32)
	y_test_tensor=torch.tensor(y_test.values,dtype=torch.float32)
	
	# set the device we will be using to train the model
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#device="cpu"
	print("We are using:",device)
	
	#train the model
	nn_fc_model=train_fc_nn(X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,
			device,n_epochs=n_epochs,batch_size=batch_size,print_freq=print_freq,learning_rate=learning_rate)
	
	return nn_fc_model

#############################################
	
def norm_data(X_data,X_norm):
	#X_data is the array we want to normalize
	#X_norm is what we are using to normalize
	X_norm_out=X_data.copy()
	
	X_norm_out['order2']= (X_norm_out['order2']-X_norm['order2'].mean())/X_norm['order2'].std()
	
	X_norm_out['order3']= (X_norm_out['order3']-X_norm['order3'].mean())/X_norm['order3'].std()
	

	X_norm_out['order4']= (X_norm_out['order4']-X_norm['order4'].mean())/X_norm['order4'].std()
	
	
	#print(X_norm_out['order4'])
	
	return X_norm_out

###############################################

def renorm_data(X_data,X_norm):
	X_renorm=X_data.copy()
	
	X_renorm[:,0]=(X_renorm[:,0]*X_norm['order2'].std())+X_norm['order2'].mean()
	X_renorm[:,1]=(X_renorm[:,1]*X_norm['order3'].std())+X_norm['order3'].mean()
	X_renorm[:,2]=(X_renorm[:,2]*X_norm['order4'].std())+X_norm['order4'].mean()
	
	return X_renorm
	
######################################3	
	
	
def train_fc_nn(X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,
			device,n_epochs,batch_size,print_freq,learning_rate):
	
	startTime = time.time()
	#Create the model and define the loss and optimizer
	nn_fc_model=perceptron().to(device)
	#nn_model=cnn().to(device)
	#print(nn_fc_model)

	loss_func=nn.MSELoss() #mean squared error
	optimizer = optim.Adam(nn_fc_model.parameters(), lr=learning_rate)

	#main training loop
	train_error=[]
	test_error=[]
	epoch_list=[]
	for epoch in range(n_epochs):
    	# set the model in training mode
		nn_fc_model.train()

		train_loss=0
		for i in range(0,len(y_train_tensor),batch_size):
			X_batch=X_train_tensor[i:i+batch_size]
			y_batch=y_train_tensor[i:i+batch_size]
			X_batch, y_batch =(X_batch.to(device), y_batch.to(device))
			X_predict_nn=nn_fc_model(y_batch)
			loss=loss_func(X_predict_nn,X_batch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss

    	#get training error
		train_loss=train_loss/(len(y_train_tensor)/batch_size)
		train_error.append(train_loss.detach().cpu().numpy())
    	#evaluate test error
		with torch.no_grad():
			nn_fc_model.eval()
			(X_test_tensor, y_test_tensor) = (X_test_tensor.to(device), y_test_tensor.to(device))
			X_predict_test=nn_fc_model(y_test_tensor)
			test_loss=loss_func(X_predict_test,X_test_tensor)
			test_error.append(test_loss.detach().cpu().numpy())

		epoch_list.append(epoch+1)
    
		if(epoch%print_freq==0 or epoch+1==n_epochs):
			print(f'Finished epoch {epoch},latest loss {train_loss}')
	#print(train_error)
	#print(test_error)
	endTime = time.time()
	print("Total time taken to train the model: {:.2f}s".format(endTime - startTime))
	
	return nn_fc_model

	
	
	
def make_fc_nn_prediction(nn_fc_model,y_test,X_train,device):
	#convert input to tensor
	#y_test_tensor=torch.tensor(y_test.values,dtype=torch.float32)
	y_test_tensor=torch.tensor(y_test.to_numpy(),dtype=torch.float32)
	y_test_tensor=y_test_tensor.to(device)
	
	#make prediction
	X_predict_fc_nn_tensor=nn_fc_model(y_test_tensor)
	X_predict_fc_nn=X_predict_fc_nn_tensor.detach().cpu().numpy()
	
	#renormalize (using the SAME, as before, X_train)
	X_predict_fc_nn_renorm=renorm_data(X_predict_fc_nn,X_train)
	
	return X_predict_fc_nn_renorm
	
	
	
	
#define the neural network
class perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1=nn.Linear(19,30)
        self.act1= nn.ReLU()
        self.hidden2=nn.Linear(30,30)
        self.act2=nn.ReLU()
        self.hidden3=nn.Linear(30,30)
        self.act3=nn.ReLU()
        self.hidden4=nn.Linear(30,25)
        self.act4=nn.ReLU()
        self.hidden5=nn.Linear(25,15)
        self.act5=nn.ReLU()
        self.hidden6=nn.Linear(15,10)
        self.act6=nn.ReLU()
        self.output=nn.Linear(10,3)

    def forward(self,x):
        x=self.act1(self.hidden1(x))
        x=self.act2(self.hidden2(x))
        x=self.act3(self.hidden3(x))
        x=self.act4(self.hidden4(x))
        x=self.act5(self.hidden5(x))
        x=self.act6(self.hidden6(x))
        x=self.output(x)
        return x
