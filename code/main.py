import logging
import os
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

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

    logger.debug(df.head(20))

    # remove rows with missing values to get a good generating basis
    for column in ['workclass', 'occupation', 'native_country']:
        df[column] = df[column].str.strip()
        count = df[column].isin(['?']).sum()
        logger.debug('column %s has %d missing values' % (column, count))
        df = df[df[column] != '?']
        logger.debug('after removing ?s from column %s we have %d rows' % (column, df.shape[0]))

    logger.debug(df.dtypes)

    categorical_variables = sorted(
        ['native_country', 'target', 'sex', 'race', 'relationship', 'education', 'occupation', 'workclass',
         'marital_status'])
    for label in categorical_variables:
        label_encoder = LabelEncoder()
        df[label] = label_encoder.fit_transform(df[label])

    logger.debug('scores predicting using all other variables:')
    for target_column in categorical_variables:
        X = df.drop([target_column], axis=1).values
        y = df[target_column].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf_dt = DecisionTreeClassifier(max_depth=10)

        clf_dt.fit(X_train, y_train)
        logger.debug('target: %s score: %.4f' % (target_column, clf_dt.score(X_test, y_test)))

    # now predict using just the numerical variables
    logger.debug('scores predicting using just numerical variables:')
    numerical_variables = sorted(['fnlwgt', 'capgain', 'caploss', 'hrsweekly'])
    for target_column in categorical_variables:
        X = df[numerical_variables]
        y = df[target_column].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf_dt = DecisionTreeClassifier(max_depth=10)

        clf_dt.fit(X_train, y_train)
        logger.debug('target: %s score: %.4f' % (target_column, clf_dt.score(X_test, y_test)))

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
