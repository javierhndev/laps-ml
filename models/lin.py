from sklearn.linear_model import LinearRegression

def fit_lin_model(X_train,y_train):
	lin_model=LinearRegression().fit(X_train,y_train)
	
	return lin_model
	
def make_lin_prediction(lin_model,X_data):
	y_prediction=lin_model.predict(X_data)
	
	return y_prediction
