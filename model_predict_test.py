import numpy as np
import model_training.a2_modeling as model
import logging

def run_predict_demo():
    #create a list with np.array, and has the shape of (g_lstm_step_backward,1)
    inputdatalist = np.random.rand(model.g_lstm_step_backward, ).tolist()
    logging.info('inputdatalist:%s', inputdatalist)

    predict = model.predict_wrapper(model.g_model_type_lstm_weekdays, [1]*model.g_lstm_step_backward, '2021-07-17 19:41:25')
    logging.info('g_model_type_lstm_weekdays predict:%s', predict)

    predict = model.predict_wrapper(model.g_model_type_lstm_weekends, [1]*model.g_lstm_step_backward, '2021-07-17 19:41:25')
    logging.info('g_model_type_lstm_weekends predict:%s', predict)

    predict = model.predict_wrapper(model.g_model_type_dnn_weekdays, [1]*model.g_dnn_step_backward, '2021-07-17 19:41:25')
    logging.info('g_model_type_dnn_weekdays predict:%s', predict)

    predict = model.predict_wrapper(model.g_model_type_dnn_weekends, [1]*model.g_dnn_step_backward, '2021-07-17 19:41:25')
    logging.info('g_model_type_dnn_weekends predict:%s', predict)



    pass


if __name__=='__main__':
    print('model_predict_test.py')
    run_predict_demo()