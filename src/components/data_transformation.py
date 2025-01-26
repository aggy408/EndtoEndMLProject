import os 
import sys 
from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer(self): 
        ''' This function is used to get the data transformation pipeline'''
        
        try:
            numeric_features = ['writing_score', 'reading_score']
            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            logging.info("Numeric Transformation Pipeline completed")

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info("Categorical Transformation Pipeline completed")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num_vars', num_pipeline, numeric_features),
                    ('cat_vars', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_transformation(self, train_path, test_path):
        try: 
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Train and Test Data readging completed")

            logging.info("Obtaining the preprocessor")
            preprocessor_obj = self.get_data_transformer()

            target = 'math_score'
            numerical_columns = ["writing_score", "reading_score"]

            input_features_train = train_df.drop(columns=[target], axis=1)
            target_feature_train = train_df[target]

            input_features_test = test_df.drop(columns=[target], axis=1)
            target_feature_test = test_df[target]

            logging.info("Fitting the preprocessor")

            input_scaled_train = preprocessor_obj.fit_transform(input_features_train)
            input_scaled_test = preprocessor_obj.transform(input_features_test)

            logging.info("Transformation completed")

            train_arr = np.c_[input_scaled_train, np.array(target_feature_train)]
            test_arr = np.c_[input_scaled_test, np.array(target_feature_test)]

            save_object(
                file_path = self.transformation_config.preprocessor_path,
                obj = preprocessor_obj
            )
            logging.info("Preprocessor saved")

            return (
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_path
            )
        except Exception as e:
            raise CustomException(e, sys)