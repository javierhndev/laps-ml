from . import rf

from sklearn.model_selection import train_test_split

#split the data in train/test sets for the roundtrip model
#X: df_input (Dazzler parameters)
#Y: df_time (Wizzler)
#rand_split: Set it to true if you want to split in random subsets everytime 
def split_dataset_roundtrip(df_input,df_time,rand_split=False):
    #define input and output for the model
    X=df_input[['order2','order3','order4']]
    y=df_time
    #split into train and test set
    if (rand_split==True):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
    return X_train, X_test, y_train, y_test



#train a roundtrip model using the Random forest
def train_roundtrip_rf(X_train, X_test, y_train, y_test,
                       n_estimators, max_features,max_depth, random_state):
    #train the forward model
    rf_model_forward=rf.fit_random_forest(X_train,y_train,n_estimators, max_features,max_depth, random_state)
    #y_predict_forest_forward=rf.make_rf_prediction(rf_model_forward,X_test)

    #train backward model with RF
    rf_model_backward=rf.fit_random_forest(y_train,X_train,n_estimators, max_features,max_depth, random_state)
    
    return rf_model_forward,rf_model_backward



#make prediction
def roundtrip_predict(y_test,rf_model_backward,rf_model_forward):
    #roundtrip prediction
    X_predict_forest_backward=rf.make_rf_prediction(rf_model_backward,y_test)
    y_predict_roundtrip=rf.make_rf_prediction(rf_model_forward,X_predict_forest_backward)
    return y_predict_roundtrip
