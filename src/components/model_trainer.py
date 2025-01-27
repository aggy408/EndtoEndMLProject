import os 
import sys 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig: 
    model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try: 
            logging.info('Split Train and Test Input Data')
            X_train, y_train, X_test, y_test = (train_array[:,:-1], train_array[:,-1], test_array[:,:-1], test_array[:,-1])
            models = {
                'RandomForest': RandomForestRegressor(),
                'GradientBoosting': GradientBoostingRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'LinearRegression': LinearRegression(),
                'KNeighbors': KNeighborsRegressor(),
                'SVR': SVR(),
                'DecisionTree': DecisionTreeRegressor(),
                'XGBoost': XGBRegressor(),
                'CatBoost': CatBoostRegressor()
            }

            model_report: dict=evaluate_model(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test, models = models)

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # Find the best model based on r2_score
            best_model = max(model_report.items(), key=lambda item: item[1]['r2_score'])
            best_model_name = best_model[0]
            best_model_score = best_model[1]

            if best_model_score['r2_score'] < 0.6:
                raise CustomException("No Model is performing well", sys)

            logging.info(f"Best Model is {best_model_name} with R2 Score: {best_model_score['r2_score']}")

            save_object(
                file_path=self.model_config.model_file_path,
                obj=models[best_model_name]
            )

            predicted = models[best_model_name].predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e, sys)