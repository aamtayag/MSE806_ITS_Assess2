import numpy as np

import model_training.lite_predict as model
import logging

def run_predict_demo():
    #firstly, preload all model when the program starts, and only needs to call the function once
    model.preload_all_model()


    #create a list with np.array, and has the shape of (g_lstm_step_backward,1)
    inputdatalist = np.random.rand(model.g_lstm_step_backward, ).tolist()

    #calculate predict_time value, and convert to  2024-07-17 19:41:25 format
    current_time = pd.Timestamp.now()
    predict_time = current_time + pd.Timedelta(minutes=15 * 5)
    formatted_predict_time = predict_time.strftime('%Y-%m-%d %H:%M:%S')

    #set parkinglot_id
    parkinglot_id = 30

    predict = model.predict_wrapper(model.g_model_type_lstm_weekdays, inputdatalist, formatted_predict_time,parkinglot_id)
    logging.info('g_model_type_lstm_weekdays predict:%s', predict)


    pass


if __name__=='__main__':
    print('model_predict_test.py')
    run_predict_demo()