import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'artifact\model.pkl'
            preprocessor_path = 'artifact\preprocessor.pkl'
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scale = preprocessor.transform(features)
            preds = model.predict(data_scale)
            return preds
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__( self,
        age: int,
        height_cm: int,
        potential: int, 
        value_eur: int, 
        preferred_foot: int, 
        weak_foot: int, 
        skill_moves: int, 
        attacking_heading_accuracy: int, 
        skill_dribbling: int, 
        skill_ball_control: int, 
        movement_reactions: int,
        power_jumping: int, 
        power_strength: int, 
        mentality_vision: int, 
        pace_median: int,
        work_rate: str, 
        team_position: str):

        self.age = age
        self.height_cm = height_cm
        self.potential = potential
        self.value_eur = value_eur
        self.preferred_foot = preferred_foot
        self.weak_foot = weak_foot
        self.skill_moves = skill_moves
        self.attacking_heading_accuracy = attacking_heading_accuracy
        self.skill_dribbling = skill_dribbling
        self.skill_ball_control = skill_ball_control
        self.movement_reactions = movement_reactions
        self.power_jumping = power_jumping
        self.power_strength = power_strength
        self.mentality_vision = mentality_vision
        self.pace_median = pace_median
        self.work_rate = work_rate
        self.team_position = team_position

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "age": [self.age],
                "height_cm": [self.height_cm],
                "potential": [self.potential],
                "value_eur": [self.value_eur],
                "preferred_foot": [self.preferred_foot],
                "weak_foot": [self.weak_foot],
                "skill_moves": [self.skill_moves],
                "attacking_heading_accuracy": [self.attacking_heading_accuracy],
                "skill_dribbling": [self.skill_dribbling],
                "skill_ball_control": [self.skill_ball_control],
                "movement_reactions": [self.movement_reactions],
                "power_jumping": [self.power_jumping],
                "power_strength": [self.power_strength],
                "mentality_vision": [self.mentality_vision],
                "pace_median": [self.pace_median],
                "work_rate": [self.work_rate],
                "team_position": [self.team_position]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
