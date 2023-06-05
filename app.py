from email.mime import application
from importlib.metadata import requires
from importlib.util import resolve_name
import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app=application

# Route for a home page 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data=CustomData(
            age = request.form.get('age'),
            height_cm = request.form.get('height_cm'),            
            potential = request.form.get('potential'),
            value_eur = request.form.get('value_eur'),
            preferred_foot = request.form.get('preferred_foot'),
            weak_foot = request.form.get('weak_foot'),
            skill_moves = request.form.get('skill_moves'),
            attacking_heading_accuracy = request.form.get('attacking_heading_accuracy'),
            skill_dribbling = request.form.get('skill_dribbling'),
            skill_ball_control = request.form.get('skill_ball_control'),
            movement_reactions = request.form.get('movement_reactions'),
            power_jumping = request.form.get('power_jumping'),
            power_strength = request.form.get('power_strength'),
            mentality_vision = request.form.get('mentality_vision'),
            pace_median = float(request.form.get('pace_median')),
            work_rate = request.form.get('work_rate'),
            team_position = request.form.get('team_position'),
        )
        pred_df = data.get_data_as_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)