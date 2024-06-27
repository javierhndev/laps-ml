from sklearn.ensemble import RandomForestRegressor

def fit_random_forest(X_train,y_train,n_estimators, max_features,max_depth, random_state=18):
	rf_model=RandomForestRegressor(n_estimators=n_estimators, max_features=max_features,max_depth=max_depth, random_state=random_state)
	
	rf_model.fit(X_train,y_train)
	
	return rf_model
	
def make_rf_prediction(rf_model,X_data):
	y_prediction=rf_model.predict(X_data)
	
	return y_prediction

