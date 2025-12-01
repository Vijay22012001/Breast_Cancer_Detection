import pickle
import os
from flask import Flask, request, jsonify, render_template
from waitress import serve

# --- Model and Scaler Loading ---
# Note: Ensure the 'models' directory exists and contains these files.
try:
    scaler_model = pickle.load(open('models/scaler_cancer.pkl','rb'))
    model_svm = pickle.load(open('models/modelsvm_cancer.pkl','rb'))
    print("Models loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model files not found. Please ensure 'models/scaler_cancer.pkl' and 'models/modelsvm_cancer.pkl' exist.")
    # In a real deployment, you might raise an exception or provide a dummy model.

# --- Flask Application Setup ---
application = Flask(__name__)
app = application

@app.route("/")
def index():
    """Renders the landing page."""
    return render_template('index.html')

@app.route("/predictdata",methods=['GET', 'POST'])
def predict_datapoint():
    """Handles the form submission and prediction."""
    if request.method=="POST":
        # Extracting 30 features from the form data
        try:
            # Group A: Mean Features (10)
            radius_mean = float(request.form.get('radius_mean'))
            texture_mean = float(request.form.get('texture_mean'))
            perimeter_mean = float(request.form.get('perimeter_mean'))
            area_mean = float(request.form.get('area_mean'))
            smoothness_mean = float(request.form.get('smoothness_mean'))
            compactness_mean = float(request.form.get('compactness_mean'))
            concavity_mean = float(request.form.get('concavity_mean'))
            concave_points_mean = float(request.form.get('concave_points_mean'))
            symmetry_mean = float(request.form.get('symmetry_mean'))
            fractal_dimension_mean = float(request.form.get('fractal_dimension_mean'))

            # Group B: Standard Error (SE) Features (10)
            radius_se = float(request.form.get('radius_se'))
            texture_se = float(request.form.get('texture_se'))
            perimeter_se = float(request.form.get('perimeter_se'))
            area_se = float(request.form.get('area_se'))
            smoothness_se = float(request.form.get('smoothness_se'))
            compactness_se = float(request.form.get('compactness_se'))
            concavity_se = float(request.form.get('concavity_se'))
            concave_points_se = float(request.form.get('concave_points_se'))
            symmetry_se = float(request.form.get('symmetry_se'))
            fractal_dimension_se = float(request.form.get('fractal_dimension_se'))

            # Group C: Worst/Largest Features (10)
            radius_worst = float(request.form.get('radius_worst'))
            texture_worst = float(request.form.get('texture_worst'))
            perimeter_worst = float(request.form.get('perimeter_worst'))
            area_worst = float(request.form.get('area_worst'))
            smoothness_worst = float(request.form.get('smoothness_worst'))
            compactness_worst = float(request.form.get('compactness_worst'))
            concavity_worst = float(request.form.get('concavity_worst'))
            concave_points_worst = float(request.form.get('concave_points_worst'))
            symmetry_worst = float(request.form.get('symmetry_worst'))
            fractal_dimension_worst = float(request.form.get('fractal_dimension_worst'))

            # Prepare data for scaling and prediction (must be in the same order as trained)
            new_data = [
                radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
                compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,

                radius_se, texture_se, perimeter_se, area_se, smoothness_se,
                compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,

                radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
                compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
            ]

            # Scale and predict
            new_data_scale = scaler_model.transform([new_data])
            result = model_svm.predict(new_data_scale)
            
            # The result is 0 (Benign) or 1 (Malignant)
            prediction_text = "Malignant (Cancerous)" if result[0] == 1 else "Benign (Non-Cancerous)"

            # Pass the prediction text to the home template
            return render_template('home.html', results=prediction_text)

        except Exception as e:
            # Simple error handling for bad input
            error_message = f"An error occurred during prediction: {e}. Please check your inputs."
            return render_template('home.html', error=error_message)
    else:
        # For GET request, simply show the form
        return render_template("home.html")

if __name__=="__main__":
    # Using waitress for production-ready serving
    print("Starting server with Waitress on http://127.0.0.1:5000")
    serve(app, host="127.0.0.1", port=5000)