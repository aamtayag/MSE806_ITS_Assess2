# -*- coding: utf-8 -*-
import os,time
import pandas
import pandas as pd
import numpy as np
import logging
from keras.layers import Dense, Activation,LSTM, Dropout,BatchNormalization,Input,Embedding,Flatten,Concatenate
from keras.models import Sequential,load_model
from keras import models
from keras.optimizers import Adam
from keras import losses
from keras.callbacks import EarlyStopping,TensorBoard
from sklearn.model_selection import train_test_split
import multiprocessing as mp


train_data_path = '/Users/hehengpan/Downloads/Parking_Transactions_datetime.csv'
#train_data_path = '/Users/hehengpan/Downloads/Parking_Transactions_20210501_v1_sample.csv'
#train_data_path = '/Users/hehengpan/Downloads/Parking_Transactions_20210501_v1_sample_tail.csv'


#global configuration for data preprocess
g_preprocess_data_drop_abnormal_duration_days = 2


#global configuration for LSTM
g_lstm_feature_number = 1
g_lstm_step_backward = 4*16
g_lstm_predict_step = 1


#globa configuration for DNN
g_dnn_feature_number = 1
g_dnn_step_backward = 4*16
g_dnn_predict_step = 1


#training configuration
g_training_batch_size = 32*1000
g_training_epochs = 100

g_random_state = 100

#model file name config
g_filename_lstm_weekdays = f'./modeldir/lstm_model_weekdays_fwd{g_lstm_predict_step}.keras'
g_filename_lstm_weekends = f'./modeldir/lstm_model_weekends_fwd{g_lstm_predict_step}.keras'
g_filename_dnn_weekend = f'./modeldir/dnn_model_weekends_fwd{g_dnn_predict_step}.keras'
g_filename_dnn_weekdays = f'./modeldir/dnn_model_weekdays_fwd{g_dnn_predict_step}.keras'


#model type name
g_model_type_lstm_weekdays = 'lstm_weekdays'
g_model_type_lstm_weekends = 'lstm_weekends'
g_model_type_dnn_weekdays = 'dnn_weekdays'
g_model_type_dnn_weekends = 'dnn_weekends'







def init_log():
    LOG_FORMAT = "%(filename)s:%(lineno)d:%(funcName)s()  %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT,)

init_log()

# Function to apply pd.to_datetime on a chunk
def apply_to_datetime(chunk):
    chunk['Start Time'] = pd.to_datetime(chunk['Start Time'])
    chunk['End Time'] = pd.to_datetime(chunk['End Time'])
    return chunk


# use multiprocessing to format datetime
def data_format_datetime():
    filepath = '/Users/hehengpan/Downloads/Parking_Transactions.csv'
    df = pd.read_csv(filepath)

    num_processes = 15
    df_split = np.array_split(df, num_processes)

    # Use multiprocessing Pool to parallelize the operation
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(apply_to_datetime, df_split)

    df = pd.concat(results)
    df.to_csv('/Users/hehengpan/Downloads/Parking_Transactions_datetime.csv', index=False)
    logging.info('df:\n%s', df)


def get_keras_callbacks(modelname:str):
    current_time = time.strftime("%m-%d-%H:%M:%S", time.localtime())
    tensorboard_recursivelog = os.path.join("./tensorlog", "fit", modelname, f"run_{current_time}")
    os.makedirs(tensorboard_recursivelog, exist_ok=True)

    #root_dictory_prefix = './tensorlog/'
    #tensorboard_recursivelog = f"{root_dictory_prefix}log4tensorboard"
    tensorboard_callback = TensorBoard(log_dir=tensorboard_recursivelog,
                                            write_graph=True,
                                            write_images=True)
    early_stop_callback = EarlyStopping(monitor='loss',
                                             patience=5,
                                             restore_best_weights=True,
                                             mode='auto')

    return [
        tensorboard_callback
        #early_stop_callback
    ]





# deprecated, do not use
def sample_data():
    df = pd.read_csv(train_data_path)
    df = df.tail(int(len(df)/10))
    df.to_csv('/Users/hehengpan/Downloads/Parking_Transactions_20210501_v1_sample_tail.csv', index=False)


def global_data_shuffle():
    df = pd.read_csv(train_data_path)
    df = pandas.DataFrame(df, columns=['ID', 'Start Time', 'End Time', 'Duration in Minutes', 'Location Group'])
    logging.info('df:\n%s', df)
    # drop abnormal Duration data, Duration in Minutes larger than N days
    df = df[df['Duration in Minutes'] < 60*24*g_preprocess_data_drop_abnormal_duration_days]
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df['End Time'] = pd.to_datetime(df['End Time'])
    # add weekdays column, 1 is Monday, 7 is Sunday
    df['weekdays'] = df['Start Time'].dt.dayofweek + 1
    # add is_weekend column, 1 is weekend, 0 is not weekend
    df['is_weekend'] = (df['weekdays'] > 5).astype(int)
    # project Start Time to sector
    #df['start_sector'] = df['Start Time'].dt.floor('15T')
    # df['start_sector'] = df['start_sector'].apply(lambda x: x.hour*4 + x.minute//15)
    #df['end_sector'] = df['End Time'].dt.floor('15T')

    # split data to weekends and non-weekends
    df_weekend = df[df['is_weekend'] == 1]
    df_weekdays = df[df['is_weekend'] == 0]

    return df_weekdays, df_weekend



def modeling_pre_preprocess_data_lstm(df:pd.DataFrame):
    df['start_sector'] = df['Start Time'].dt.floor('15T')
    df['end_sector'] = df['End Time'].dt.floor('15T')

    # aggregate in and out car number in each sector
    df_sector_start = df.groupby(['start_sector']).size().reset_index(name='start_count')
    df_sector_start.rename(columns={'start_sector': 'sector_time'}, inplace=True)
    df_sector_start = df_sector_start.set_index('sector_time')

    df_end_start = df.groupby(['end_sector']).size().reset_index(name='end_count')
    df_end_start.rename(columns={'end_sector': 'sector_time'}, inplace=True)
    df_end_start = df_end_start.set_index('sector_time')



    #concatinate two table, use field sector as index, fill with 0 if no value
    df_sector = pd.concat([df_sector_start, df_end_start], axis=1, join='outer').fillna(0)
    df_sector.sort_index(inplace=True)
    df_sector['diff_count'] = df_sector['start_count'] - df_sector['end_count']
    #logging.info('df_sector OLD:\n%s', df_sector)

    max_sector_time_vlaue = max(df_sector.index)
    min_sector_time_vlaue = min(df_sector.index)

    #fill sector_time field value, max value as max_sector_time_vlaue, min value as min_sector_time_vlaue,
    #interval 15min, and start_count, end_count field fill with 0
    df_sector_time = pd.DataFrame(index=pd.date_range(min_sector_time_vlaue, max_sector_time_vlaue, freq='15T'))
    df_sector_time['sector_time'] = df_sector_time.index

    #concatinage df_sector_time and df_sector, fill with 0 if not exist
    df_sector = pd.concat([df_sector_time, df_sector], axis=1, join='outer').fillna(0)
    df_sector.sort_index(inplace=True)
    #preserver only sector_time and diff_count field
    df_sector = df_sector[['sector_time','diff_count']]



    #prepare sequencial data for LSTM

    #shift operation for previous value
    shiftcount = g_lstm_step_backward
    for shift in range(1, shiftcount):
        df_sector[f'diff_count-{shift}'] = df_sector['diff_count'].shift(+shift)

    #mapping sector_time to sector type or sector id
    df_sector['sector_type'] = df_sector['sector_time'].apply(lambda x: x.hour*4 + x.minute//15)

    # shift operation for future value
    df_sector[f'target'] = df_sector['diff_count'].shift(-g_lstm_predict_step)

    # dropna
    df_sector.dropna(inplace=True)

    #logging.info('df_sector shift:\n%s', df_sector)

    return df_sector


def modeling_pre_preprocess_data_dnn(df:pd.DataFrame):
    df['start_sector'] = df['Start Time'].dt.floor('15T')
    df['end_sector'] = df['End Time'].dt.floor('15T')

    # aggregate in and out car number in each sector
    df_sector_start = df.groupby(['start_sector']).size().reset_index(name='start_count')
    df_sector_start.rename(columns={'start_sector': 'sector_time'}, inplace=True)
    df_sector_start = df_sector_start.set_index('sector_time')

    df_end_start = df.groupby(['end_sector']).size().reset_index(name='end_count')
    df_end_start.rename(columns={'end_sector': 'sector_time'}, inplace=True)
    df_end_start = df_end_start.set_index('sector_time')



    #concatinate two table, use field sector as index, fill with 0 if no value
    df_sector = pd.concat([df_sector_start, df_end_start], axis=1, join='outer').fillna(0)
    df_sector.sort_index(inplace=True)
    df_sector['diff_count'] = df_sector['start_count'] - df_sector['end_count']
    #logging.info('df_sector OLD:\n%s', df_sector)

    max_sector_time_vlaue = max(df_sector.index)
    min_sector_time_vlaue = min(df_sector.index)

    #fill sector_time field value, max value as max_sector_time_vlaue, min value as min_sector_time_vlaue,
    #interval 15min, and start_count, end_count field fill with 0
    df_sector_time = pd.DataFrame(index=pd.date_range(min_sector_time_vlaue, max_sector_time_vlaue, freq='15T'))
    df_sector_time['sector_time'] = df_sector_time.index

    #concatinage df_sector_time and df_sector, fill with 0 if not exist
    df_sector = pd.concat([df_sector_time, df_sector], axis=1, join='outer').fillna(0)
    df_sector.sort_index(inplace=True)
    #preserver only sector_time and diff_count field
    df_sector = df_sector[['sector_time','diff_count']]



    #prepare data for dnn

    #shift operation for previous value
    shiftcount = g_dnn_step_backward
    for shift in range(1, shiftcount):
        df_sector[f'diff_count-{shift}'] = df_sector['diff_count'].shift(+shift)

    #mapping sector_time to sector type or sector id
    df_sector['sector_type'] = df_sector['sector_time'].apply(lambda x: x.hour*4 + x.minute//15)

    # shift operation for future value
    df_sector[f'target'] = df_sector['diff_count'].shift(-g_dnn_predict_step)

    # dropna
    df_sector.dropna(inplace=True)

    return df_sector




def create_network_lstm(input_shape, num_category, embedding_dim=4):
    # embedding input
    category_input = Input(shape=(1,), name='category_input')
    embedding = Embedding(num_category, embedding_dim)(category_input)
    embedding = Flatten()(embedding)  # flaten embedding to use in dense layer

    # car parking diff input
    #parking_diff_input = Input(shape=(input_shape[1],), name='parking_diff_input')
    parking_diff_input = Input(shape=input_shape, name='parking_diff_input')

    # LSTM layer for time series data
    lstm_out = LSTM(50, return_sequences=False)(parking_diff_input)

    # merge LSTM out and embedding layer output
    merged = Concatenate()([lstm_out, embedding])

    # full connection
    dense_out = Dense(50, activation='relu')(merged)
    output = Dense(1, activation='linear')(dense_out)  # predict the car parking diff


    model = models.Model(inputs=[parking_diff_input, category_input], outputs=output)
    #model.compile(optimizer=Adam(), loss='mean_squared_error')

    #model.compile(optimizer=Adam(), loss = losses.Huber())
    model.compile(optimizer=Adam(), loss='MAE')

    return model


def model_lstm_analysis():
    df_weekdays, df_weekend = global_data_shuffle()

    # ----------------  weekdays training ------------------------------
    if False:
        df_sector = modeling_pre_preprocess_data_lstm(df_weekdays)
        input_shape = (g_lstm_step_backward, g_lstm_feature_number)
        model = create_network_lstm(input_shape, 4 * 25)
        logging.info('model summary:\n%s', model.summary())

        # produce columns
        column_names_lstm = [f'diff_count-{shift}' for shift in range(g_lstm_step_backward-1,0,-1)] + ['diff_count']
        #logging.info('column_names_lstm:%s', column_names_lstm)

        X_parking_diff = df_sector[column_names_lstm].values
        X_category = df_sector['sector_type'].values
        Y = df_sector['target'].values

        X_train, X_val, y_train, y_val = train_test_split(X_parking_diff, Y, test_size=0.1, random_state=g_random_state)
        X_train_category, X_val_category, y_train_category, y_val_category = train_test_split(X_category, Y, test_size=0.1, random_state=g_random_state)

        logging.info('prepare to train model for weekdays, train data len:%s validation data len:%s', len(X_train),
                     len(X_val))
        model.fit([X_train,X_train_category],y_train,
                  epochs=g_training_epochs,
                  batch_size=g_training_batch_size,
                  callbacks=get_keras_callbacks('lstm_weekdays'),
                  validation_data=([X_val,X_val_category],y_val)
                  )

        model.save(g_filename_lstm_weekdays)


    # ----------------  weekend training ------------------------------
    if False:
        df_sector = modeling_pre_preprocess_data_lstm(df_weekend)
        input_shape = (g_lstm_step_backward, g_lstm_feature_number)
        model = create_network_lstm(input_shape, 4 * 25)
        #logging.info('model summary:\n%s', model.summary())

        # produce columns
        column_names_lstm = [f'diff_count-{shift}' for shift in range(g_lstm_step_backward - 1, 0, -1)] + ['diff_count']
        # logging.info('column_names_lstm:%s', column_names_lstm)

        X_parking_diff = df_sector[column_names_lstm].values
        X_category = df_sector['sector_type'].values
        Y = df_sector['target'].values

        X_train, X_val, y_train, y_val = train_test_split(X_parking_diff, Y, test_size=0.1, random_state=g_random_state)
        X_train_category, X_val_category, y_train_category, y_val_category = train_test_split(X_category, Y,
                                                                                              test_size=0.1,
                                                                                              random_state=g_random_state)
        logging.info('prepare to train model for weekends, train data len:%s validation data len:%s', len(X_train), len(X_val))
        model.fit([X_train, X_train_category], y_train,
                  epochs=g_training_epochs,
                  batch_size=g_training_batch_size,
                  callbacks=get_keras_callbacks('lstm_weekends'),
                  validation_data=([X_val, X_val_category], y_val)
                  )

        model.save(g_filename_lstm_weekends)

        pass



def create_network_dnn(input_shape, num_category, embedding_dim=4):
    # embedding input
    category_input = Input(shape=(1,), name='category_input')
    embedding = Embedding(num_category, embedding_dim)(category_input)
    embedding = Flatten()(embedding)  # flaten embedding to use in dense layer

    # car parking diff input
    parking_diff_input = Input(shape=input_shape, name='parking_diff_input')

    # merge full connection out and embedding layer output
    merged = Concatenate()([parking_diff_input, embedding])

    # full connection
    #merged = BatchNormalization()(merged)
    dense_out = Dense(100, activation='relu')(merged)
    dense_out = BatchNormalization()(dense_out)
    dense_out = Dropout(0.05)(dense_out)
    dense_out = Dense(50, activation='relu')(dense_out)
    output = Dense(1, activation='linear')(dense_out)  # predict the car parking diff

    model = models.Model(inputs=[parking_diff_input, category_input], outputs=output)
    model.compile(optimizer=Adam(), loss = losses.Huber())

    return model

def model_dnn_analysis():
    df_weekdays, df_weekend = global_data_shuffle()

    # ----------------  weekdays training ------------------------------
    if False:
        df_sector = modeling_pre_preprocess_data_dnn(df_weekdays)
        input_shape = (g_dnn_step_backward, )
        model = create_network_dnn(input_shape, 4 * 25)

        column_names_dnn = [f'diff_count-{shift}' for shift in range(g_dnn_step_backward - 1, 0, -1)] + ['diff_count']

        X_parking_diff = df_sector[column_names_dnn].values
        X_category = df_sector['sector_type'].values
        Y = df_sector['target'].values

        X_train, X_val, y_train, y_val = train_test_split(X_parking_diff, Y, test_size=0.1, random_state=g_random_state)
        X_train_category, X_val_category, y_train_category, y_val_category = train_test_split(X_category, Y,
                                                                                              test_size=0.1,
                                                                                              random_state=g_random_state)

        logging.info('prepare to train dnn model for weekdays, train data len:%s validation data len:%s', len(X_train),
                     len(X_val))
        model.fit([X_train, X_train_category], y_train,
                  epochs=g_training_epochs,
                  batch_size=g_training_batch_size,
                  callbacks=get_keras_callbacks('dnn_weekdays'),
                  validation_data=([X_val, X_val_category], y_val)
                  )

        model.save(g_filename_dnn_weekdays)

    # ----------------  weekend training ------------------------------
    if True:
        df_sector = modeling_pre_preprocess_data_dnn(df_weekend)
        input_shape = (g_dnn_step_backward,)
        model = create_network_dnn(input_shape, 4 * 25)

        column_names_dnn = [f'diff_count-{shift}' for shift in range(g_dnn_step_backward - 1, 0, -1)] + ['diff_count']

        X_parking_diff = df_sector[column_names_dnn].values
        X_category = df_sector['sector_type'].values
        Y = df_sector['target'].values

        X_train, X_val, y_train, y_val = train_test_split(X_parking_diff, Y, test_size=0.1, random_state=g_random_state)
        X_train_category, X_val_category, y_train_category, y_val_category = train_test_split(X_category, Y,
                                                                                              test_size=0.1,
                                                                                              random_state=g_random_state)

        logging.info('prepare to train dnn model for weekdays, train data len:%s validation data len:%s', len(X_train),
                     len(X_val))
        model.fit([X_train, X_train_category], y_train,
                  epochs=g_training_epochs,
                  batch_size=g_training_batch_size,
                  callbacks=get_keras_callbacks('dnn_weekdays'),
                  validation_data=([X_val, X_val_category], y_val)
                  )

        model.save(g_filename_dnn_weekend)

        pass

    pass


'''
Usage:
    Description: predict the car parking diff for the next 15 minutes
    Input:
    modeltype:
        g_model_type_lstm_weekdays
        g_model_type_lstm_weekends
        g_model_type_dnn_weekdays
        g_model_type_dnn_weekends
    inputdatalist: the list of the car parking diff in the latest N sector, N needs to be the same as g_lstm_step_backward or g_dnn_step_backward
    current_localtime: the current time, the format needs to be the same as:  2021-07-17 19:41:25
    
    Return: the predict value of the car parking diff for the next 15 minutes

'''
def predict_wrapper(modeltype, inputdatalist:list, current_localtime:str):
    #parameter check
    #check the value of the format of current_localtime, the format needs to be the same as:  2021-07-17 19:41:25
    try:
        current_localtime = pd.to_datetime(current_localtime)
    except:
        logging.error('ERROR invalid current_localtime:%s', current_localtime)
        return None

    sector_time = current_localtime.floor('15T')
    sector_id = sector_time.hour*4 + sector_time.minute//15

    if modeltype not in [g_model_type_lstm_weekdays, g_model_type_lstm_weekends, g_model_type_dnn_weekdays, g_model_type_dnn_weekends]:
        logging.error('ERROR invalid modeltype:%s', modeltype)
        return None
    if modeltype == g_model_type_lstm_weekdays or modeltype == g_model_type_lstm_weekends:
        if len(inputdatalist) != g_lstm_step_backward:
            logging.error('ERROR invalid inputdatalist len:%s, expect:%s', len(inputdatalist), g_lstm_step_backward)
            return None
    if modeltype == g_model_type_dnn_weekdays or modeltype == g_model_type_dnn_weekends:
        if len(inputdatalist) != g_dnn_step_backward:
            logging.error('ERROR invalid inputdatalist len:%s, expect:%s', len(inputdatalist), g_dnn_step_backward)
            return None


    model = None
    # load model
    if modeltype == g_model_type_lstm_weekdays:
        model = load_model(g_filename_lstm_weekdays)
    elif modeltype == g_model_type_lstm_weekends:
        model = load_model(g_filename_lstm_weekends)
    elif modeltype == g_model_type_dnn_weekdays:
        model = load_model(g_filename_dnn_weekdays)
    elif modeltype == g_model_type_dnn_weekends:
        model = load_model(g_filename_dnn_weekend)
    else:
        logging.error('ERROR invalid modeltype:%s', modeltype)
        return None

    X_input = np.array(inputdatalist).reshape(1,-1)
    X_category = np.array([sector_id]).reshape(1,-1)

    Y_predict = model.predict([X_input, X_category],verbose=0)
    #logging.info('Y_predict:%s', Y_predict)
    return Y_predict[0][0]



def test_predict():
    #create a list with np.array, and has the shape of (g_lstm_step_backward,1)
    inputdatalist = np.random.rand(g_lstm_step_backward, ).tolist()
    logging.info('inputdatalist:%s', inputdatalist)

    predict = predict_wrapper(g_model_type_lstm_weekdays, [1]*g_lstm_step_backward, '2021-07-17 19:41:25')
    logging.info('g_model_type_lstm_weekdays predict:%s', predict)

    predict = predict_wrapper(g_model_type_lstm_weekends, [1]*g_lstm_step_backward, '2021-07-17 19:41:25')
    logging.info('g_model_type_lstm_weekends predict:%s', predict)

    predict = predict_wrapper(g_model_type_dnn_weekdays, [1]*g_dnn_step_backward, '2021-07-17 19:41:25')
    logging.info('g_model_type_dnn_weekdays predict:%s', predict)

    predict = predict_wrapper(g_model_type_dnn_weekends, [1]*g_dnn_step_backward, '2021-07-17 19:41:25')
    logging.info('g_model_type_dnn_weekends predict:%s', predict)



    pass

if __name__=='__main__':
    logging.info('a2_modeling.py')
    #model_lstm_analysis()
    #model_dnn_analysis()

    test_predict()





