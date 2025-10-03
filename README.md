# Housing Price Prediction Flask App

This project is a simple Flask web application that serves a trained machine learning model for housing price prediction.

---

## ⚙️ Requirements

Before running the project, make sure you have the following files inside the `model/` directory:

- **`housing_model.pkl`**
- **`model_features.json`**

---

## 📂 Why are these files needed?

### `housing_model.pkl`

This is the trained **machine learning model** (serialized with `joblib`).

- It contains all the learned parameters (weights, intercepts, etc.) of the model.
- Without this file, Flask cannot make predictions because there is no model to load.

### `model_features.json`

This is the **feature order/metadata file**.

- It lists the input features (columns) the model was trained on.
- It ensures that the features provided to the model during prediction are in the **same order and format** as during training.
- Without it, predictions may fail or produce incorrect results.

Together, these two files allow the Flask app to:

1. Load the trained model.
2. Validate and preprocess user input.
3. Generate consistent and correct predictions.

---

## 🚀 Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/mealibek/housing-flask.git
   cd housing-flask
   ```
