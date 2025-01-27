import os 
import sys 
import numpy as np 
import pandas as pd
import dill  
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import mean_squared_error, r2_score


def save_object(obj, file_path):
    ''' This function is used to save the object to the file path'''
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file:
            dill.dump(obj, file)

        logging.info(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    ''' This function is used to evaluate the model'''
    try:
        model_report = {}
        for model_name, model in models.items():
            logging.info(f"Training {model_name}")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_report[model_name] = {
                'r2_score': r2_score(y_test, y_pred)
            }
            logging.info(f"{model_name} - R2: {r2_score(y_test, y_pred)}")
        return model_report
    except Exception as e:
        raise CustomException(e, sys)