from . import lin
from . import rf
from . import fcnn
from . import cnn

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


##############################

def train_roundtrip_lin(X_train, X_test, y_train, y_test):
    #train the forward model
    lin_model_forward=lin.fit_lin_model(X_train,y_train)
    #train backward model
    lin_model_backward=lin.fit_lin_model(y_train,X_train)

    return lin_model_forward, lin_model_backward


#train a roundtrip model using the Random forest
def train_roundtrip_rf(X_train, X_test, y_train, y_test,
                       n_estimators, max_features,max_depth, random_state):
    #train the forward model
    rf_model_forward=rf.fit_random_forest(X_train,y_train,n_estimators, max_features,max_depth, random_state)
    #y_predict_forest_forward=rf.make_rf_prediction(rf_model_forward,X_test)

    #train backward model with RF
    rf_model_backward=rf.fit_random_forest(y_train,X_train,n_estimators, max_features,max_depth, random_state)
    
    return rf_model_forward,rf_model_backward

def train_roundtrip_fcnn(X_train, X_test, y_train, y_test,
                         n_estimators, max_features,max_depth, random_state,
                         n_epochs,batch_size,print_freq,learning_rate):
    #train the forward model (using RF)
    rf_model_forward=rf.fit_random_forest(X_train,y_train,n_estimators, max_features,max_depth, random_state)

    #train the backward model with FCNN
    fcnn_model_backward = fcnn.fit_fc_nn(X_train,y_train,X_test,y_test,
              n_epochs,batch_size,print_freq,learning_rate)
    return rf_model_forward,fcnn_model_backward

def train_roundtrip_cnn(X_train, X_test, y_train, y_test,
                        n_estimators, max_features,max_depth, random_state,
                        n_epochs,batch_size,print_freq,learning_rate):
    #train the forward model (using RF)
    rf_model_forward=rf.fit_random_forest(X_train,y_train,n_estimators, max_features,max_depth, random_state)

    #train the backward CNN model
    cnn_model_backward= cnn.fit_cnn(X_train,y_train,X_test,y_test,
              n_epochs,batch_size,print_freq,learning_rate)
    return rf_model_forward,cnn_model_backward

#############################3

#make prediction
def roundtrip_lin_predict(y_test,lin_model_backward,lin_model_forward):
    X_predict_lin_backward=lin.make_lin_prediction(lin_model_backward,y_test)
    y_predict_roundtrip=lin.make_lin_prediction(lin_model_forward,X_predict_lin_backward)
    return y_predict_roundtrip

def roundtrip_rf_predict(y_test,rf_model_backward,rf_model_forward):
    #roundtrip prediction
    X_predict_forest_backward=rf.make_rf_prediction(rf_model_backward,y_test)
    y_predict_roundtrip=rf.make_rf_prediction(rf_model_forward,X_predict_forest_backward)
    return y_predict_roundtrip

def roundtrip_fcnn_predict(y_test,X_train,fcnn_model_backward,rf_model_forward,device):
    #X_train was used for normalization so it needs to be reused during prediction
    X_predict_fcnn_backward=fcnn.make_fc_nn_prediction(fcnn_model_backward,y_test,X_train,device)
    y_predict_roundtrip=rf.make_rf_prediction(rf_model_forward,X_predict_fcnn_backward)
    return y_predict_roundtrip

def roundtrip_cnn_predict(y_test,X_train,cnn_model_backward,rf_model_forward,device):
    #X_train was used for normalization so it needs to be reused during prediction
    X_predict_cnn_backward=cnn.make_cnn_prediction(cnn_model_backward,y_test,X_train,device)
    y_predict_roundtrip=rf.make_rf_prediction(rf_model_forward,X_predict_cnn_backward)
    return y_predict_roundtrip