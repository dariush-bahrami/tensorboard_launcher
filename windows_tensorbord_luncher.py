# -*- coding: utf-8 -*-
def get_tensorboard_logs_directory_path(logs_dir_name="logs",
                                        tesnorboard_logs_dir_name='fit'):
    import os

    tensorboard_logs_dir_path = logs_dir_name + '\\' + tesnorboard_logs_dir_name + '\\'

    if not os.path.exists(logs_dir_name):
        os.path.exists(logs_dir_name)

    if not os.path.exists(tensorboard_logs_dir_path):
        os.mkdir(tensorboard_logs_dir_path)
    
    return tensorboard_logs_dir_path


def get_time_stamp():
    import datetime
    time_stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return time_stamp


def write_batch_file(time_stamp, tensorboard_logs_dir_path,
                     batchfile_blueprint_path='luncher_batchfile_blueprint.txt'):

    with open(batchfile_blueprint_path,  mode='r', encoding='utf-8') as open_file:
        batchfile_blueprint = open_file.read()
        
    batchfile = batchfile_blueprint.replace('set logDirectory=',
                                            f'set logDirectory={time_stamp}', 1)
    
    luncher_batchfile_name = time_stamp + '_luncher' +  '.bat'
    luncher_batchfile_path = tensorboard_logs_dir_path + luncher_batchfile_name
    
    with open(luncher_batchfile_path, mode='w', encoding='utf-8') as open_file:
        open_file.write(batchfile)


def get_log_path(time_stamp, logs_dir_name="logs", tesnorboard_logs_dir_name='fit'):
    tensorboard_logs_dir_path = logs_dir_name + '\\' + tesnorboard_logs_dir_name + '\\'
    log_path = tensorboard_logs_dir_path + time_stamp   
    return log_path

def get_tensorboard_instance():
    import tensorflow as tf

    tensorboard_logs_dir_path = get_tensorboard_logs_directory_path()

    time_stamp = get_time_stamp()

    write_batch_file(time_stamp, tensorboard_logs_dir_path)

    log_dir = get_log_path(time_stamp, logs_dir_name="logs",
                           tesnorboard_logs_dir_name='fit')

    return tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                          histogram_freq=1,
                                          profile_batch = 100000000)