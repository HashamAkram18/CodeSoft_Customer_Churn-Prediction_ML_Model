import numpy as np
import pandas as pd
import os
import pickle
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request


# Define the app
app = Flask(__name__)

@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/train')
def train():
    # Execute preprocessing.py
    preprocess_process = subprocess.Popen(['python', 'src/preprocessing.py'], stdout=subprocess.PIPE)
    preprocess_output, preprocess_error = preprocess_process.communicate()

    # Execute training.py
    train_process = subprocess.Popen(['python', 'src/training.py'], stdout=subprocess.PIPE)
    train_output, train_error = train_process.communicate()

    # Check if both processes executed successfully
    if preprocess_process.returncode == 0 and train_process.returncode == 0:
        # Get the directory of the current script
        script_directory = os.path.dirname(__file__)

        # Define metrics directory
        metrics_dir = os.path.join(script_directory, 'src', 'metrics')

        # Load accuracy metric
        accuracy_path = os.path.join(metrics_dir, 'accuracy.txt')
        if os.path.exists(accuracy_path):
            with open(accuracy_path, 'r') as f:
                accuracy = f.read()
        else:
            accuracy = "N/A"  # or any other default value

        # Load R2 score metric
        r2_path = os.path.join(metrics_dir, 'r2_score.txt')
        if os.path.exists(r2_path):
            with open(r2_path, 'r') as f:
                r2 = f.read()
        else:
            r2 = "N/A"  # or any other default value

        print("Accuracy:", accuracy)
        print("R2 Score:", r2)    
                

        # Pass metrics to the template
        return render_template('train_model.html', accuracy=accuracy, r2=r2)
    else:
        return render_template('train_model.html', accuracy=None)




# Define the path to the pickled model
model_path = os.path.join('src', 'model', 'classifier.pkl')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Check if the request is a POST request
    if request.method == 'POST':
        # Load the pickled model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        # Get the input data from the form
        credit_score = float(request.form['CreditScore'])
        age = float(request.form['Age'])
        tenure = float(request.form['Tenure'])
        balance = float(request.form['Balance'])
        num_of_products = float(request.form['NumOfProducts'])
        has_cr_card = int(request.form['HasCrCard'])
        is_active_member = int(request.form['IsActiveMember'])
        estimated_salary = float(request.form['EstimatedSalary'])
        geography_germany = 1 if request.form.get('Geography_Germany') else 0
        geography_spain = 1 if request.form.get('Geography_Spain') else 0
        gender_male = 1 if request.form.get('Gender_Male') else 0

        # Create the input data as a numpy array
        input_data = np.array([[credit_score, geography_germany, geography_spain, gender_male, 
                               age, tenure, balance, num_of_products, has_cr_card, 
                               is_active_member, estimated_salary]])

        # Predict the output using the loaded model
        prediction = model.predict(input_data)

        # Map the prediction to a string
        prediction_str = "The customer is likely to Exit" if prediction[0] == 0 else "The customer will Stay"

        # Render the predict.html template with the prediction string
        return render_template('predict.html', prediction=prediction_str)
    else:
        return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)