'''
The puropse of this file is to use run tests on the churn_library file
and log the results.

Author: Amandeep Mander
Date: December 2nd, 2021
'''

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    # Run EDA
    perform_eda(df)

    # Check if files were generated
    try:
        dir_len = len(os.listdir('./images/eda'))
        assert dir_len > 0
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_eda: The function is not producing output files")
        raise err


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    '''
    # List of categorical column names
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # List of encoded categorical column names
    encoded_cols = []
    for col in cat_columns:
        encoded_cols.append(col + '_Churn')

    df = encoder_helper(df, cat_columns)

    try:
        for col in encoded_cols:
            assert col in df.columns
        logging.info("Testing encoder_helper: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The function does not contain the encoded columns")
        raise err

    return df


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(df)

    try:
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: The function does not produce the train and test sets")
        raise err

    return X_train, X_test, y_train, y_test


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(X_train, X_test, y_train, y_test)

    try:
        models_dir_len = len(os.listdir('./models'))
        results_dir_len = len(os.listdir('./images/results'))
        assert models_dir_len > 0
        assert results_dir_len > 0
        logging.info("Testing train_models: SUCCESS")

    except AssertionError as err:
        logging.error(
            "Testing test_train_models: The function is not producing the required output files")
        raise err


if __name__ == "__main__":

    # test import data
    DATA = test_import(cls.import_data)

    # test eda
    test_eda(cls.perform_eda, DATA)

    # test encoder helper
    DATA = test_encoder_helper(cls.encoder_helper, DATA)

    # test feature engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering(
        cls.perform_feature_engineering, DATA)

    # test train model
    test_train_models(cls.train_models, X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
