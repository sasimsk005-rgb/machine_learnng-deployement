%%writefile app.py
import pandas as pd
from flask import Flask, request, jsonify
import joblib

app= Flask(__name__)

from sklearn.ensemble import RandomForestRegressor

# Instantiate a RandomForestRegressor model
# Set random_state for reproducibility
model = RandomForestRegressor(random_state=42)

# Fit the model to the training data
model.fit(X_train, y_train)

print("Random Forest Regressor model trained successfully.")


# Load the trained model
model = joblib.load('random_forest_model.joblib')

# Get the feature columns used during training from the kernel state
# In a real deployment, X_columns would typically be saved/loaded alongside the model
# For this demonstration, we assume X.columns is available in the environment during development
# For production, you might hardcode this list or save it with the model.
X_columns = ['age', 'sex', 'test_time', 'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        
        # Convert JSON data to a pandas DataFrame
        # Ensure the column order matches the training data
        df_input = pd.DataFrame([data])
        
        # Reorder columns to match the training features
        df_input = df_input[X_columns]

        # Make prediction using the loaded model
        prediction = model.predict(df_input)
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': prediction.tolist()[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
