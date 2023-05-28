import imp
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path = os.path.join('artifact',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tansformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ['age', 'height_cm', 'weight_kg', 'potential', 
                'value_eur', 'wage_eur', 'preferred_foot', 'weak_foot', 'skill_moves', 'attacking_crossing', 
                'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 
                'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 
                'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 
                'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 
                'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 
                'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_standing_tackle', 
                'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
                'goalkeeping_positioning', 'goalkeeping_reflexes', 'pace_median', 'shooting_median', 'passing_median', 
                'dribbling_median', 'defending_median', 'physic_median'
            ]
            categorical_columns = ['work_rate', 'team_position']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numercial Columns: {numerical_columns}")
            logging.info(f"Categorical Columns: {categorical_columns}")

            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()
            
            target_column_name = "overall"
            numerical_columns = ['age', 'height_cm', 'weight_kg', 'potential', 
                'value_eur', 'wage_eur', 'preferred_foot', 'weak_foot', 'skill_moves', 'attacking_crossing', 
                'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 
                'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 
                'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 
                'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 
                'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 
                'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_standing_tackle', 
                'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
                'goalkeeping_positioning', 'goalkeeping_reflexes', 'pace_median', 'shooting_median', 'passing_median', 
                'dribbling_median', 'defending_median', 'physic_median'
            ]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path = self.data_tansformation_config.preprocessor_ob_file_path,
                obj = preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_tansformation_config.preprocessor_ob_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)