
import pandas as pd
import numpy as np

def read_dataset(filename):
    df_input=pd.read_hdf(filename,'df_input')
    df_time=pd.read_hdf(filename,'df_time')
    df_freq_int=pd.read_hdf(filename,'df_freq_int')

    #get the arrays and print them
    print('Input DF')
    print(df_input)

    print('----')
    print('Time values')
    df_time_val=df_time.loc[0]
    print(df_time_val)

    print('----')
    print("Intensity")
    df_time= df_time.drop([0]) #drop the time values
    df_time.reset_index(inplace=True,drop=True) #to start again from 0 (as the other DataFrames)
    #df_time = df_time.drop('index', axis=1)
    print(df_time)

    print('----')
    print("Freq values")
    df_freq_val=df_freq_int.loc[0]
    print(df_freq_val)

    print('----')
    print("Frequency intesity")
    df_freq_int= df_freq_int.drop([0]) #drop the freq values
    df_freq_int.reset_index(inplace=True,drop=True)
    #df_freq_int = df_freq_int.drop('index', axis=1)
    print(df_freq_int)
    
    return df_input,df_time_val,df_time,df_freq_val,df_freq_int


############################################
############################################3

#get the shot number from the test index
#(potential bug: pass the right df_input that has been used to create y_test. For example: df_input_clean)
def get_shot_num(shot_index,y_test,df_input):
    #print(y_test.iloc[shot_index])
    #print(y_test.iloc[shot_index])
    id_value=y_test.iloc[shot_index].name
    #print(df_time_clean.loc[id_value])
    #print(y_test_reset.loc[shot_index])
    #print(df_input_clean.loc[id_value])
    return(df_input['shot number'].loc[id_value])
    
  ############################################
  ############################################
    
#The function calculates the variance from the shots with the same input parameters and store it in a DF with SAME size (so there is redundant data)
#dup_shots should be the maximum number of duplicate shots -1 but is OK if higher (but slower algorithm)

def get_var(df_input,df_time,dup_shots):
	list_aux=[]
	for index,row in df_input.iterrows():
		o2=row['order2']
		o3=row['order3']
		o4=row['order4']
		#check if the previous and following rows were the same
		time_list=[]
		for i in range(index-dup_shots,index+dup_shots+1):
			if (i>=0 and i<=len(df_input)-1):
				#print(index,dup_shots,i)
				df_aux=df_input.iloc[i] #row to check
				if (df_aux['order2']==o2 and df_aux['order3']==o3 and df_aux['order4']==o4):
					time_list.append(df_time.iloc[i].to_list()) #store the rows with same parameters in a list
		time_list=np.array(time_list)
		time_list=np.var(time_list,axis=0)
		list_aux.append(time_list)
            
	#create the dataframe from the list
	df_time_var=pd.DataFrame(list_aux)
	#print(df_time)
	#print(df_time_var)
	return df_time_var
	
##########################################
#########################################

#Drop the shots below some value from the raw max distribution
def clean_shots_below(value,df_input,df_time,df_freq_int):
    maxval_freq=df_freq_int.max(axis=1)
    meanval=maxval_freq.mean()
    twosigma=2*maxval_freq.std()
    print('')
    print('Shots with max val below this will be discarded:',value)
    print('')

    #get the indexes of the shots that don't fulfilll the requirement
    #badpoints_up=maxval_freq.index[(maxval_freq>(meanval+twosigma))].tolist()
    badpoints_down=maxval_freq.index[(maxval_freq<value)].tolist()

    #Drop those shots from the databases
    df_input_clean=df_input.drop(badpoints_down)

    df_time_clean=df_time.drop(badpoints_down)

    df_freq_int_clean=df_freq_int.drop(badpoints_down)

    #reset their indexes
    df_input_clean.reset_index(inplace=True,drop=True) #to start again from 0 (as the other DataFrames)
    df_time_clean.reset_index(inplace=True,drop=True)
    df_freq_int_clean.reset_index(inplace=True,drop=True)

    #TESTING
    #print('Testing!')
    #print(df_freq_int_clean)
    #test the histogram
    #test_maxval_freq=df_freq_int_clean.max(axis=1)
    #test_maxval_freq.hist(bins=50)
    
    return df_input_clean,df_time_clean,df_freq_int_clean

############################################
###########################################

#Drop the shots above some value from the raw max distribution
def clean_shots_above(value,df_input,df_time,df_freq_int):
    maxval_freq=df_freq_int.max(axis=1)
    meanval=maxval_freq.mean()
    twosigma=2*maxval_freq.std()
    print('')
    print('Shots with max val above this will be discarded:',value)
    print('')

    #get the indexes of the shots that don't fulfilll the requirement
    #badpoints_up=maxval_freq.index[(maxval_freq>(meanval+twosigma))].tolist()
    badpoints_down=maxval_freq.index[(maxval_freq>value)].tolist()

    #Drop those shots from the databases
    df_input_clean=df_input.drop(badpoints_down)

    df_time_clean=df_time.drop(badpoints_down)

    df_freq_int_clean=df_freq_int.drop(badpoints_down)

    #reset their indexes
    df_input_clean.reset_index(inplace=True,drop=True) #to start again from 0 (as the other DataFrames)
    df_time_clean.reset_index(inplace=True,drop=True)
    df_freq_int_clean.reset_index(inplace=True,drop=True)

    #TESTING
    #print('Testing!')
    #print(df_freq_int_clean)
    #test the histogram
    #test_maxval_freq=df_freq_int_clean.max(axis=1)
    #test_maxval_freq.hist(bins=50)
    
    return df_input_clean,df_time_clean,df_freq_int_clean

    
    
    
