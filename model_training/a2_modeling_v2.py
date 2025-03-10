# -*- coding: utf-8 -*-
import os,time
import pandas
import pandas as pd
import numpy as np
import logging
from keras.layers import Dense, Activation,LSTM, Dropout,BatchNormalization,Input,Embedding,Flatten,Concatenate
from keras import regularizers
from keras.models import Sequential,load_model
from keras import models
from keras.optimizers import Adam
from keras import losses
from keras.callbacks import EarlyStopping,TensorBoard
from sklearn.model_selection import train_test_split
import multiprocessing as mp
import tensorflow as tf
from joblib import Parallel, delayed

'''
4                   Core  6823980
14              Non-Core  1897217
7            East Austin  1868805
24           West Campus  1003516
13               Mueller   798657
19        South Congress   487820
17                Rainey   487134
20                Toomey   448116
8                   Emma   146695
21      Unknown Location   137212
5           Dawson's Lot   129640
23                Walter   123244
0            Austin High   117712
11              MACC Lot   108372
2          Butler Shores    99370
25     Woods of Westlake    87016
12             MoPac Lot    84442
22                 Walsh    64312
3         Colorado River    63527
6         Deep Eddy Pool    29489
10             IH 35 Lot    24380
16            Q2 Stadium    20337
1       Bartholomew Pool     4341
15        Northwest Pool     3318
9          Garrison Pool     2105
18  Silicon and Titanium      148
'''


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

g_localgroup_list = ['Core', 'Non-Core', 'East Austin', 'Mueller', 'West Campus', 'Rainey', 'Toomey', 'Emma', "Dawson's Lot", 'Walter', 'MACC Lot', 'Butler Shores', 'Austin High', 'Walsh', 'MoPac Lot', 'Deep Eddy Pool', 'Colorado River', 'Woods of Westlake', 'Bartholomew Pool', 'Northwest Pool', 'Garrison Pool']
g_parkinglotid_list = [i for i in range(17, 39)]
g_localgroup_dict = {g_localgroup_list[i]:i+1 for i in range(len(g_localgroup_list))}

g_parkinglotid_localgroup_dict = dict(zip(g_parkinglotid_list,g_localgroup_list+['Garrison Pool']))

#g_localgroup_dict = {g_localgroup_list[i]:i for i in range(len(g_localgroup_list))}


train_data_path = '/Users/hehengpan/Downloads/Parking_Transactions_datetime.csv'
#train_data_path = '/Users/hehengpan/Downloads/Parking_Transactions_20210501_v1_sample.csv'
#train_data_path = '/Users/hehengpan/Downloads/Parking_Transactions_20210501_v1_sample_tail.csv'


#global configuration for data preprocess
g_preprocess_data_drop_abnormal_duration_days = 2


#global configuration for LSTM
g_lstm_feature_number = 1
g_lstm_step_backward = 4*16
g_lstm_predict_step = 1

g_predict_step = 1
g_step_backward = 4*16

#globa configuration for DNN
g_dnn_feature_number = 1
g_dnn_step_backward = 4*16
g_dnn_predict_step = 1


#training configuration
g_training_batch_size = 32*100
g_training_epochs = 100

g_random_state = 100



model_lstm_weekdays = None
model_lstm_weekends = None
model_dnn_weekdays = None
model_dnn_weekends = None


def init_log():
    LOG_FORMAT = "%(filename)s:%(lineno)d:%(funcName)s()  %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt=DATE_FORMAT,)



def get_start_count_columns():
    columnlist = [f'start_count-{i}' for i in range(g_step_backward-1, 0, -1)] + ['start_count']
    return columnlist

def get_embeded_columns():
    columnlist = ['localgroup_id', 'sector_type']
    return columnlist

def get_runtime_path():
    s = os.getcwd()
    dir = s.split('/')[-1]
    if dir == 'model_training':
        return './'
    else:
        return './model_training/'
        pass
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




def get_localgroupid_by_parkinglotid(parkinglotid):
    if parkinglotid not in g_parkinglotid_list:
        return None
    localgroup =  g_parkinglotid_localgroup_dict[parkinglotid]
    localgroupid = g_localgroup_dict[localgroup]
    return localgroupid




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



def pre_process_data():
    df = pd.read_csv(train_data_path)
    df = pandas.DataFrame(df, columns=['ID', 'Start Time', 'End Time', 'Duration in Minutes', 'Location Group'])
    logging.info('df:\n%s', df)
    # drop abnormal Duration data, Duration in Minutes larger than N days
    df = df[df['Duration in Minutes'] < 60 * 24 * g_preprocess_data_drop_abnormal_duration_days]
    # format Start Time and End Time
    df['Start Time'] = pd.to_datetime(df['Start Time'])
    df['End Time'] = pd.to_datetime(df['End Time'])

    # 把location group的值为Unknown Location的数据删除
    df = df[df['Location Group'] != 'Unknown Location']

    # add weekdays column, 1 is Monday, 7 is Sunday
    #df['weekdays'] = df['Start Time'].dt.dayofweek + 1
    # add is_weekend column, 1 is weekend, 0 is not weekend
    #df['is_weekend'] = (df['weekdays'] > 5).astype(int)
    # project Start Time to sector
    # df['start_sector'] = df['Start Time'].dt.floor('15T')
    # df['start_sector'] = df['start_sector'].apply(lambda x: x.hour*4 + x.minute//15)
    # df['end_sector'] = df['End Time'].dt.floor('15T')

    # split data to weekends and non-weekends
    #df_weekend = df[df['is_weekend'] == 1]
    #df_weekdays = df[df['is_weekend'] == 0]

    #return df_weekdays, df_weekend


    return df


def pre_process_sigle_localgroup(df:pd.DataFrame):
    df['start_sector'] = df['Start Time'].dt.floor('15T')
    #df['end_sector'] = df['End Time'].dt.floor('15T')

    # aggregate in and out car number in each sector
    df_sector_start = df.groupby(['start_sector']).size().reset_index(name='start_count')
    df_sector_start.rename(columns={'start_sector': 'sector_time'}, inplace=True)
    df_sector_start = df_sector_start.set_index('sector_time')

    #df_end_start = df.groupby(['end_sector']).size().reset_index(name='end_count')
    #df_end_start.rename(columns={'end_sector': 'sector_time'}, inplace=True)
    #df_end_start = df_end_start.set_index('sector_time')

    #logging.info('df_sector_start:\n%s', df_sector_start)


    # concatinate two table, use field sector as index, fill with 0 if no value
    #df_sector = pd.concat([df_sector_start, df_end_start], axis=1, join='outer').fillna(0)
    #df_sector.sort_index(inplace=True)
    #df_sector['diff_count'] = df_sector['start_count'] - df_sector['end_count']
    # logging.info('df_sector OLD:\n%s', df_sector)

    df_sector = df_sector_start.copy()

    max_sector_time_vlaue = max(df_sector.index)
    min_sector_time_vlaue = min(df_sector.index)

    # fill sector_time field value, max value as max_sector_time_vlaue, min value as min_sector_time_vlaue,
    # interval 15min, and start_count, end_count field fill with 0
    df_sector_time = pd.DataFrame(index=pd.date_range(min_sector_time_vlaue, max_sector_time_vlaue, freq='15T'))
    df_sector_time['sector_time'] = df_sector_time.index

    # concatinage df_sector_time and df_sector, fill with 0 if not exist
    df_sector = pd.concat([df_sector_time, df_sector], axis=1, join='outer').fillna(0)
    df_sector.sort_index(inplace=True)
    #logging.info('df_sector:\n%s', df_sector)
    # preserver only sector_time and diff_count field
    #df_sector = df_sector['start_count']
    df_sector = df_sector.reset_index().drop(columns=['index'])


    # prepare sequencial data for LSTM

    # shift operation for previous value
    shiftcount = g_step_backward
    for shift in range(1, shiftcount):
        df_sector[f'start_count-{shift}'] = df_sector['start_count'].shift(+shift)

    # mapping sector_time to sector type or sector id
    df_sector['sector_type'] = df_sector['sector_time'].apply(lambda x: x.hour * 4 + x.minute // 15)

    # shift operation for future value
    df_sector['target'] = df_sector['start_count'].shift(-g_predict_step)

    #根据 sector_time 字段, 判断是否是周末, 如果是周末, 则设置为1, 否则设置为0
    df_sector['is_weekend'] = df_sector['sector_time'].dt.dayofweek.apply(lambda x: 1 if x > 4 else 0)



    # dropna
    df_sector.dropna(inplace=True)





    # logging.info('df_sector shift:\n%s', df_sector)
    #logging.info('df_sector:\n%s', df_sector)

    return df_sector

def apply_zero_tag(x):
    for i in range(1, g_step_backward):
        if x[f'start_count-{i}'] != 0:
            return 0
    return 1


def parallel_apply_zero_tag(df, num_jobs=4):
    # Split the dataframe into smaller chunks
    df_split = np.array_split(df, len(df)/100000)

    # Use joblib Parallel to parallelize the operation
    results = Parallel(n_jobs=num_jobs)(delayed(lambda chunk: chunk.apply(apply_zero_tag, axis=1))(chunk) for chunk in df_split)

    # Concatenate the results back into a single series
    return pd.concat(results)



def modeling_pre_preprocess_data(df:pd.DataFrame):
    num_processes = mp.cpu_count()-1
    dflist = []
    for localgroup in g_localgroup_list:
        df_localgroup = df[df['Location Group'] == localgroup]
        df_localgroup = pre_process_sigle_localgroup(df_localgroup)
        df_localgroup['localgroup'] = localgroup
        dflist.append(df_localgroup)

    df_sector = pd.concat(dflist)

    #增加一列, 用于标识localgroup的位置id
    df_sector['localgroup_id'] = df_sector['localgroup'].apply(lambda x: g_localgroup_dict[x])

    #剔除掉字段start_count-1, start_count-2, start_count-3 等等字段值为0的行
    logging.info('start zero tag')
    df_sector['zero_tag'] = parallel_apply_zero_tag(df_sector, num_jobs=num_processes)
    logging.info('end zero tag')
    #logging.info('before len:%s', len(df_sector))
    df_sector = df_sector[df_sector['zero_tag'] == 0]
    #logging.info('after len:%s', len(df_sector))

    #删除字段zero_tag
    df_sector.drop(columns=['zero_tag'], inplace=True)

    logging.info('df_sector:\n%s', df_sector)
    logging.info('columns:%s', df_sector.columns.tolist())


    return df_sector



def create_network_lstm(input_shape,
                        num_category_sector_type,
                        embedding_dim_sector_type=4,
                        num_category_localgroup=len(g_localgroup_list),
                        embedding_dim_localgroup=4):
    # embedding category sector input
    category_sector_type_input = Input(shape=(1,), name='category_sector_type_input')
    embedding_sector_type = Embedding(num_category_sector_type, embedding_dim_sector_type)(category_sector_type_input)
    embedding_sector_type = Flatten()(embedding_sector_type)  # flaten embedding to use in dense layer

    # embedding category localgroup input
    category_localgroup_input = Input(shape=(1,), name='category_localgroup_input')
    embedding_localgroup = Embedding(num_category_localgroup, embedding_dim_localgroup)(category_localgroup_input)
    embedding_localgroup = Flatten()(embedding_localgroup)  # flaten embedding to use in dense layer


    # start count input
    start_count_input = Input(shape=input_shape, name='start_count_input')

    # LSTM layer for time series data
    #lstm_out = LSTM(50, return_sequences=False)(start_count_input)
    lstm_out = LSTM(g_step_backward, return_sequences=False)(start_count_input)

    # merge LSTM out and embedding layer output
    merged = Concatenate()([lstm_out, embedding_sector_type,embedding_localgroup])

    # full connection
    dense_out = Dense(50, activation='relu')(merged)
    #dense_out = Dropout(0.05)(dense_out)
    dense_out = Dense(10, activation='relu')(dense_out)
    output = Dense(1, activation='linear')(dense_out)
    #output = Dense(1, activation='relu')(dense_out)


    model = models.Model(inputs=[start_count_input, category_sector_type_input,category_localgroup_input], outputs=output)
    #model.compile(optimizer=Adam(), loss='mean_squared_error')

    #model.compile(optimizer=Adam(), loss = losses.Huber())
    #model.compile(optimizer=Adam(), loss='MAE')
    model.compile(optimizer=Adam(), loss='MSE')

    return model

def create_network_dnn(input_shape,
                        num_category_sector_type,
                        embedding_dim=4,
                        num_category_localgroup=len(g_localgroup_list),
                        embedding_dim_localgroup=4):
    # embedding input
    category_sector_type_input = Input(shape=(1,), name='category_input')
    embedding_sector_type = Embedding(num_category_sector_type, embedding_dim)(category_sector_type_input)
    embedding_sector_type = Flatten()(embedding_sector_type)  # flaten embedding to use in dense layer

    # embedding category localgroup input
    category_localgroup_input = Input(shape=(1,), name='category_localgroup_input')
    embedding_localgroup = Embedding(num_category_localgroup, embedding_dim_localgroup)(category_localgroup_input)
    embedding_localgroup = Flatten()(embedding_localgroup)  # flaten embedding to use in dense layer

    # car parking diff input
    start_count_input = Input(shape=input_shape, name='start_count_input')

    # merge full connection out and embedding layer output
    merged = Concatenate()([start_count_input, embedding_sector_type,embedding_localgroup])

    # full connection
    #merged = BatchNormalization()(merged)
    dense_out = Dense(100, activation='relu')(merged)
    #dense_out = BatchNormalization()(dense_out)
    dense_out = Dropout(0.05)(dense_out)
    dense_out = Dense(50, activation='relu')(dense_out)
    output = Dense(1, activation='linear')(dense_out)  # predict the car parking diff

    model = models.Model(inputs=[start_count_input, category_sector_type_input,category_localgroup_input], outputs=output)
    model.compile(optimizer=Adam(), loss = losses.Huber())


    return model





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




def test_pre_process_data():
    df = pre_process_data()
    df = modeling_pre_preprocess_data(df)


def lstm_train_process():
    training_batch_size = 32 * 100
    training_epochs = 5
    df = pd.read_csv('/Users/hehengpan/Downloads/df_sector.csv')

    #modelname = 'lstm_weekends'
    modelname = 'lstm_weekdays'
    df = df[df['is_weekend'] == 0]

    #logging.info('lstm train data:%s', df.info())
    #df = df.head(100000)
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    lstm_model = create_network_lstm((g_step_backward, 1),
                                     4*25,
                                     8,
                                     len(g_localgroup_list)+10,
                                     8)

    X_start_count = df[get_start_count_columns()].values
    X_category_sector_type = df['sector_type'].values
    X_localgroup_id = df['localgroup_id'].values

    Y = df['target'].values

    X_train, X_val, y_train, y_val = train_test_split(X_start_count, Y, test_size=0.1, random_state=g_random_state)
    X_train_category, X_val_category, y_train_category, y_val_category = train_test_split(X_category_sector_type, Y, test_size=0.1, random_state=g_random_state)
    X_train_localgroup, X_val_localgroup, y_train_localgroup, y_val_localgroup = train_test_split(X_localgroup_id, Y, test_size=0.1, random_state=g_random_state)


    lstm_model.fit([X_train, X_train_category, X_train_localgroup], y_train,
                     epochs=training_epochs,
                     batch_size=training_batch_size,
                     callbacks=get_keras_callbacks(modelname),
                     validation_data=([X_val, X_val_category, X_val_localgroup], y_val)
                     )

    lstm_model.save(f'./modeldir_v2/{modelname}.keras')


    pass



def dnn_train_process():
    training_batch_size = 32 * 100
    training_epochs = 5
    df = pd.read_csv('/Users/hehengpan/Downloads/df_sector.csv')

    #modelname = 'dnn_weekdays'
    modelname = 'dnn_weekends'
    df = df[df['is_weekend'] == 1]
    input_shape = (g_step_backward,)
    dnn_model = create_network_dnn(input_shape,
                                        4*25,
                                        8,
                                        len(g_localgroup_list)+10,
                                        8)



    X_start_count = df[get_start_count_columns()].values
    X_category_sector_type = df['sector_type'].values
    X_localgroup_id = df['localgroup_id'].values

    Y = df['target'].values

    X_train, X_val, y_train, y_val = train_test_split(X_start_count, Y, test_size=0.1, random_state=g_random_state)
    X_train_category, X_val_category, y_train_category, y_val_category = train_test_split(X_category_sector_type, Y,
                                                                                          test_size=0.1,
                                                                                          random_state=g_random_state)
    X_train_localgroup, X_val_localgroup, y_train_localgroup, y_val_localgroup = train_test_split(X_localgroup_id, Y,
                                                                                                  test_size=0.1,
                                                                                                  random_state=g_random_state)

    dnn_model.fit([X_train, X_train_category, X_train_localgroup], y_train,
                   epochs=training_epochs,
                   batch_size=training_batch_size,
                   callbacks=get_keras_callbacks(modelname),
                   validation_data=([X_val, X_val_category, X_val_localgroup], y_val)
                   )



    dnn_model.save(f'./modeldir_v2/{modelname}.keras')

    pass



def task_pre_process_data_and_save():
    logging.info('%s', g_localgroup_dict)
    df = pre_process_data()
    df = modeling_pre_preprocess_data(df)
    df.to_csv('/Users/hehengpan/Downloads/df_sector.csv', index=False)
    pass


def general_test():
    #test_pre_process_data()
    lstm_train_process()
    pass



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


    if prediction_time < pd.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())):
        logging.error('ERROR invalid prediction_time:%s', prediction_time)
        return None


    if prediction_time > pd.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())) + pd.Timedelta('1 days'):
        logging.error('ERROR prediction_time is 24hs later than now, invalid prediction_time:%s', prediction_time)
        return None



    current_localtime = pd.to_datetime(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    diff = prediction_time - current_localtime
    diff = diff.total_seconds() / 60
    if diff < 15:
        return enter_count_seq_list[-1]
    recursive_count = int(diff // 15)

    logging.info('recursive_count:%s', recursive_count)


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

    # X_input = np.array(enter_count_seq_list).reshape(1, -1)
    # X_category = np.array([sector_id]).reshape(1, -1)
    # X_localgroup = np.array([localgroup_id]).reshape(1, -1)
    #
    # Y_predict = model.predict([X_input, X_category,X_localgroup], verbose=3)
    # # logging.info('Y_predict:%s', Y_predict)
    # return Y_predict[0][0]

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

    return Y_predict[0][0]


def test_predict():
    preload_all_model()
    #create a list with np.array, and has the shape of (g_lstm_step_backward,1)
    inputdatalist = np.random.rand(g_step_backward, ).tolist()
    #logging.info('inputdatalist:%s', inputdatalist)

    predict = predict_wrapper(g_model_type_lstm_weekdays, inputdatalist, '2025-03-11 12:41:25',30)
    logging.info('g_model_type_lstm_weekdays predict:%s', predict)

    # predict = predict_wrapper(g_model_type_lstm_weekends, inputdatalist, '2021-07-17 19:41:25',3)
    # logging.info('g_model_type_lstm_weekends predict:%s', predict)
    #
    # predict = predict_wrapper(g_model_type_dnn_weekdays, inputdatalist, '2021-07-17 19:41:25',3)
    # logging.info('g_model_type_dnn_weekdays predict:%s', predict)
    #
    # predict = predict_wrapper(g_model_type_dnn_weekends, inputdatalist, '2021-07-17 19:41:25',3)
    # logging.info('g_model_type_dnn_weekends predict:%s', predict)
    #
    # ts = time.time()
    # for i in range(10):
    #     predict = predict_wrapper(g_model_type_dnn_weekends, inputdatalist, '2021-07-17 19:41:25', 3)
    # diff = time.time() - ts
    # logging.info('time:%s', diff)
    # pass





# def test_id_mapping():
#     logging.info('g_localgroup_dict:%s', g_localgroup_dict)
#     logging.info('g_localgroup_list:%s', g_localgroup_list)
#     logging.info(g_parkinglotid_list)
#     logging.info('%s, %s, %s',len(g_parkinglotid_list), len(g_localgroup_list), len(g_localgroup_dict))
#
#     logging.info('g_parkinglotid_localgroup_dict:%s', g_parkinglotid_localgroup_dict)
#
#     localgroupid = get_localgroupid_by_parkinglotid(20)
#     logging.info('localgroupid:%s', localgroupid)
#     pass
#
#



if __name__=='__main__':
    init_log()
    logging.info('a2_modeling.py')
    #model_lstm_analysis()
    #model_dnn_analysis()

    #test_predict()
    #task_pre_process_data_and_save()
    #general_test()

    #lstm_train_process()
    #dnn_train_process()
    test_predict()
    #test_id_mapping()




