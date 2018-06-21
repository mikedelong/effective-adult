import logging
import os
import time

import pandas as pd

if __name__ == '__main__':
    start_time = time.time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    input_folder = '../input/'
    output_folder = '../output/'

    input_folder_exists = os.path.isdir(input_folder)
    if not input_folder_exists:
        logger.warning('input folder %s does not exist. Quitting.' % input_folder)
        quit()
    output_folder_exists = os.path.isdir(output_folder)
    if not output_folder_exists:
        logger.warning('output folder %s does not exist. Quitting.' % output_folder)
        quit()

    train_file = input_folder + 'adult.data'
    input_columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_years', 'marital_status', 'occupation',
                     'relationship', 'race', 'sex', 'capgain', 'caploss', 'hrsweekly', 'native_country', 'target']
    df = pd.read_csv(train_file, names=input_columns)
    logger.debug('the original training dataset has shape %d x %d' % df.shape)
    columns = df.columns.values
    logger.debug('the dataset has columns %s' % columns)

    logger.debug(df.head())

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
