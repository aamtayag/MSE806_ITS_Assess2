
import os,time
import pandas
import pandas as pd
import numpy as np
import logging

from keras.models import load_model


model_lstm_weekdays = None
model_lstm_weekends = None
model_dnn_weekdays = None
model_dnn_weekends = None



g_step_backward = 4*16


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_runtime_path():
    s = os.getcwd()
    dir = s.split('/')[-1]
    if dir == 'model_training':
        return './'
    else:
        return './model_training/'
        pass

def init_log():
    LOG_FORMAT = "%(filename)s:%(lineno)d:%(funcName)s()  %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT,)

#model file name config
g_filename_lstm_weekdays = f'{get_runtime_path()}modeldir_v2/lstm_weekdays.keras'
g_filename_lstm_weekends = f'{get_runtime_path()}modeldir_v2/lstm_weekends.keras'
g_filename_dnn_weekend = f'{get_runtime_path()}modeldir_v2/dnn_weekends.keras'
g_filename_dnn_weekdays = f'{get_runtime_path()}modeldir_v2/dnn_weekdays.keras'


#model type name
g_model_type_lstm_weekdays = 'lstm_weekdays'
g_model_type_lstm_weekends = 'lstm_weekends'
g_model_type_dnn_weekdays = 'dnn_weekdays'
g_model_type_dnn_weekends = 'dnn_weekends'


g_localgroup_list = ['Core', 'Non-Core', 'East Austin', 'Mueller', 'West Campus', 'Rainey', 'Toomey', 'Emma', "Dawson's Lot", 'Walter', 'MACC Lot', 'Butler Shores', 'Austin High', 'Walsh', 'MoPac Lot', 'Deep Eddy Pool', 'Colorado River', 'Woods of Westlake', 'Bartholomew Pool', 'Northwest Pool', 'Garrison Pool']
g_parkinglotid_list = [i for i in range(17, 39)]
g_localgroup_dict = {g_localgroup_list[i]:i+1 for i in range(len(g_localgroup_list))}

g_parkinglotid_localgroup_dict = dict(zip(g_parkinglotid_list,g_localgroup_list+['Garrison Pool']))

def get_localgroupid_by_parkinglotid(parkinglotid):
    if parkinglotid not in g_parkinglotid_list:
        return None
    localgroup =  g_parkinglotid_localgroup_dict[parkinglotid]
    localgroupid = g_localgroup_dict[localgroup]
    return localgroupid



'''
preload all model when the program starts, and only needs to call the function once
'''
def preload_all_model():
    global model_lstm_weekdays
    global model_lstm_weekends
    global model_dnn_weekdays
    global model_dnn_weekends
    model_lstm_weekdays = load_model(g_filename_lstm_weekdays)
    model_lstm_weekends = load_model(g_filename_lstm_weekends)
    model_dnn_weekdays = load_model(g_filename_dnn_weekdays)
    model_dnn_weekends = load_model(g_filename_dnn_weekend)





'''
Usage:
    Description: predict the car parking diff for the next 15 minutes
    Input:
    modeltype:
        g_model_type_lstm_weekdays
        g_model_type_lstm_weekends
        g_model_type_dnn_weekdays
        g_model_type_dnn_weekends
    enter_count_seq_list: 
        the list of the car parking start count in the latest N sector, N needs to be the same as g_step_backward
    prediction_time: 
        the time needs to predict, should be larger than now and less than 24hrs ahead, the format needs to be the same as:  2021-07-17 19:41:25
    parkinglot_id: 
        the parkinglot id, designates in the table parkinglots


    Return: 
        the predict value of the car parking enter count at prediction_time
        Or None if the input parameter is invalid, or some error happens
        Caller needs to check the return value, if None, then the input parameter is invalid

'''


def predict_wrapper(modeltype, enter_count_seq_list: list,prediction_time: str,parkinglot_id: int):
    # parameter check
    # check the value of the format of prediction_time, the format needs to be the same as:  2021-07-17 19:41:25
    try:
        prediction_time = pd.to_datetime(prediction_time)
    except:
        logging.error('ERROR invalid prediction_time:%s', prediction_time)
        return None

    #check prediction_time needs to be greater than the current time
    if prediction_time < pd.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())):
        logging.error('ERROR invalid prediction_time:%s', prediction_time)
        return None

    #check prediction_time needs to be less than the current time + 24 hours
    if prediction_time > pd.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + pd.Timedelta('1 days'):
        logging.error('ERROR prediction_time is 24hs later than now, invalid prediction_time:%s', prediction_time)
        return None


    #calculate prediction_time for recursive_count
    current_localtime = pd.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    diff = prediction_time - current_localtime
    diff = diff.total_seconds() / 60
    if diff < 15:
        return enter_count_seq_list[-1]
    recursive_count = int(diff // 15)

    #check parkinglot_id is valid
    if parkinglot_id not in g_parkinglotid_list:
        logging.error('ERROR invalid parkinglot_id:%s', parkinglot_id)
        return None

    localgroup_id = get_localgroupid_by_parkinglotid(parkinglot_id)

    sector_time = current_localtime.floor('15T')
    sector_id = sector_time.hour * 4 + sector_time.minute // 15

    if modeltype not in [g_model_type_lstm_weekdays, g_model_type_lstm_weekends, g_model_type_dnn_weekdays,
                         g_model_type_dnn_weekends]:
        logging.error('ERROR invalid modeltype:%s', modeltype)
        return None
    if modeltype == g_model_type_lstm_weekdays or modeltype == g_model_type_lstm_weekends:
        if len(enter_count_seq_list) != g_step_backward:
            logging.error('ERROR invalid inputdatalist len:%s, expect:%s', len(inputdatalist), g_lstm_step_backward)
            return None
    if modeltype == g_model_type_dnn_weekdays or modeltype == g_model_type_dnn_weekends:
        if len(enter_count_seq_list) != g_step_backward:
            logging.error('ERROR invalid inputdatalist len:%s, expect:%s', len(inputdatalist), g_dnn_step_backward)
            return None



    model = None
    # load model
    if modeltype == g_model_type_lstm_weekdays:
        model = model_lstm_weekdays
    elif modeltype == g_model_type_lstm_weekends:
        model = model_lstm_weekends
    elif modeltype == g_model_type_dnn_weekdays:
        model = model_dnn_weekdays
    elif modeltype == g_model_type_dnn_weekends:
        model = model_dnn_weekends
    else:
        logging.error('ERROR invalid modeltype:%s', modeltype)
        return None

    for i in range(recursive_count):
        X_input = np.array(enter_count_seq_list).reshape(1, -1)
        X_category = np.array([sector_id]).reshape(1, -1)
        X_localgroup = np.array([localgroup_id]).reshape(1, -1)

        Y_predict = model.predict([X_input, X_category,X_localgroup], verbose=0)
        #logging.info('count:%s Y_predict:%s', i, Y_predict)
        enter_count_seq_list.pop(0)
        enter_count_seq_list.append(Y_predict[0][0])
        sector_time = sector_time + pd.Timedelta('15 minutes')
        sector_id = sector_time.hour * 4 + sector_time.minute // 15

    predict = Y_predict[0][0]
    if predict < 0:
        return 0
    return abs(round(predict))



def test_predict():
    preload_all_model()
    #create a list with np.array, and has the shape of (g_lstm_step_backward,1)
    inputdatalist = np.random.rand(g_step_backward, ).tolist()
    #logging.info('inputdatalist:%s', inputdatalist)

    predict = predict_wrapper(g_model_type_lstm_weekdays, inputdatalist, '2025-03-11 12:41:25',30)
    logging.info('g_model_type_lstm_weekdays predict:%s', predict)







if __name__ == '__main__':
    print('lite_predict.py')
    init_log()
    test_predict()
    pass