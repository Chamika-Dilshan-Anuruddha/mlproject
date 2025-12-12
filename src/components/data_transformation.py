import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [ 
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(stratergy='median')),
                    ("Oh_encoder", OneHotEncoder())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(stratergy='most_frequent')),
                    ("Oh_encoder", OneHotEncoder()),
                    ("Scaler", StandardScaler(with_mean=False)),

                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numetical columns: {numerical_columns}")

            preporcessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)

                ]
            )

            return preporcessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self,train_path, test_path):
        try:
            pass
        except Exception as e:
            raise CustomException(e,sys)