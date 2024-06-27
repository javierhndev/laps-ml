#NOTE. These functions have been designed for the roundtrip model. In particular, the backward model. The X and Y are inverted 

import time

import torch
import torch.nn as nn
import torch.optim as optim

from models.fcnn import norm_data,renorm_data



def fit_cnn(X_train,y_train,X_test,y_test,
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
	cnn_model=train_cnn(X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,
			device,n_epochs=n_epochs,batch_size=batch_size,print_freq=print_freq,learning_rate=learning_rate)
	
	return cnn_model
	
	
#################################


def train_cnn(X_train_tensor,y_train_tensor,X_test_tensor,y_test_tensor,
			device,n_epochs,batch_size,print_freq,learning_rate):
	startTime = time.time()
	
	#Create the model and define the loss and optimizer
	cnn_model=cnn().to(device)
	#print(nn_model)

	loss_func=nn.MSELoss() #mean squared error
	optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

	#main training loop
	train_error=[]
	test_error=[]
	epoch_list=[]
	for epoch in range(n_epochs):
		# set the model in training mode
		cnn_model.train()

		train_loss=0
		for i in range(0,len(y_train_tensor),batch_size):
			X_batch=X_train_tensor[i:i+batch_size]
			y_batch=y_train_tensor[i:i+batch_size]
			y_batch=y_batch.unsqueeze(1)
			X_batch, y_batch =(X_batch.to(device), y_batch.to(device))
			X_predict_cnn=cnn_model(y_batch)
			loss=loss_func(X_predict_cnn,X_batch)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss

		#get training error
		train_loss=train_loss/(len(y_train_tensor)/batch_size)
		train_error.append(train_loss.detach().cpu().numpy())
		#evaluate test error
		with torch.no_grad():
			cnn_model.eval()
			y_test_model=y_test_tensor.unsqueeze(1)
			X_test_model=X_test_tensor#.unsqueeze(1)
			(X_test_model, y_test_model) = (X_test_model.to(device), y_test_model.to(device))
			X_predict_test=cnn_model(y_test_model)
			test_loss=loss_func(X_predict_test,X_test_model)
			test_error.append(test_loss.detach().cpu().numpy())

		epoch_list.append(epoch+1)
    
		if(epoch%print_freq==0 or epoch+1==n_epochs):
			print(f'Finished epoch {epoch},latest loss {train_loss}')
	#print(train_error)
	#print(test_error)
	endTime = time.time()
	print("Total time taken to train the model: {:.2f}s".format(endTime - startTime))
	
	return cnn_model
	
############################3

def make_cnn_prediction(cnn_model,y_test,X_train,device):
	#convert input to tensor
	y_test_tensor_in=torch.tensor(y_test.values,dtype=torch.float32)
	y_test_tensor=y_test_tensor_in.unsqueeze(1)
	y_test_tensor=y_test_tensor.to(device)
	
	#make prediction
	X_predict_cnn_tensor=cnn_model(y_test_tensor)
	X_predict_cnn=X_predict_cnn_tensor.detach().cpu().numpy()
	
	#renormalize (using the SAME, as before, X_train)
	X_predict_cnn_renorm=renorm_data(X_predict_cnn,X_train)
	
	return X_predict_cnn_renorm

######################

#define the neural network
class cnn(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.act1= nn.ReLU()

        self.pool1 = nn.MaxPool1d(kernel_size=2,stride=1)

        self.conv2 = nn.Conv1d(in_channels=6, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.act2= nn.ReLU()
        
        self.pool2 = nn.MaxPool1d(kernel_size=3,stride=2)

        self.conv3 = nn.Conv1d(in_channels=12, out_channels=20, kernel_size=3, stride=1, padding=1)
        self.act3= nn.ReLU()
        
        self.pool3 = nn.MaxPool1d(kernel_size=2,stride=2)

        self.flat = nn.Flatten()

        self.fc4 = nn.Linear(80, 60)
        self.act4= nn.ReLU()

        self.fc5 = nn.Linear(60, 50)
        self.act5= nn.ReLU()

        self.fc6 = nn.Linear(50, 25)
        self.act6= nn.ReLU()

        self.output=nn.Linear(25,3)

    def forward(self,x):
        #input batchx1x19   output batchx6x19
        x=self.act1(self.conv1(x))
        #input batchx6x19   output batchx6x17
        x=self.pool1(x)
        
        #input batchx6x17   output batchx12x17
        x=self.act2(self.conv2(x))
        #input batchx12x17   output batchx12x8
        x=self.pool2(x) 

        #input batchx12x8   output batchx20x8
        x=self.act3(self.conv3(x))
        #input batchx20x8   output batchx20x4
        x=self.pool3(x) 
        
        #input batchx20x4 output batchx80
        x=self.flat(x)
        
        #input batchx80   output batchx60
        x=self.act4(self.fc4(x))
        #input batchx60   output batchx50
        x=self.act5(self.fc5(x))
        #input batchx50   output batchx25
        x=self.act6(self.fc6(x))
        #input batchx25   output batchx3
        x=self.output(x)
        return x
