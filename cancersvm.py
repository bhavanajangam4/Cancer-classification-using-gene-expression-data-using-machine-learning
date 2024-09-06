#data handling
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from flask import Flask, render_template, request

# Step 1: Load and Preprocess Data
data = pd.read_csv(r"cancer_gene_expression.csv")

X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]

X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X, y)

# Preprocessing
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_resampled)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_encoded, test_size=0.2, random_state=42)

# Scale data between 0 and 1
min_max_scaler = MinMaxScaler()
X_train_norm = min_max_scaler.fit_transform(X_train)
X_test_norm = min_max_scaler.transform(X_test)

# Initialize PCA with the desired number of components
n_components = 1200
pca = PCA(n_components=n_components)

# Fit PCA on the normalized training data
pca.fit(X_train_norm)

# Transform the training and test data using the learned PCA
X_train_selected = pca.transform(X_train_norm)
X_test_selected = pca.transform(X_test_norm)

# Train SVM model
svm_model = SVC()
svm_model.fit(X_train_selected, y_train)

# Initialize Flask app
app = Flask(__name__)

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for prediction
#@app.route('/predict', methods=['POST'])
'''def predict():
    # Get input data from form
    input_data = request.form['input_data']
    # Preprocess input data
    input_array = np.array(input_data.split(','), dtype=np.float64)
    if len(input_array) != n_components:
        return "Invalid input! Please provide all features."
    # Perform prediction
    prediction = svm_model.predict(input_array.reshape(1, -1))
    predicted_cancer_type = label_encoder.inverse_transform(prediction)[0]
    return f"Predicted Cancer Type: {predicted_cancer_type}"'''

# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from form
    input_data = request.form['input_data']
    # Preprocess input data
    input_array = np.array(input_data.split(','), dtype=np.float64)
    if len(input_array) != n_components:
        return "Invalid input! Please provide all features."
    # Perform prediction
    prediction = svm_model.predict(input_array.reshape(1, -1))
    predicted_cancer_type = label_encoder.inverse_transform(prediction)[0]
    return predicted_cancer_type


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)