# Heart Disease Prediction Project

A machine learning project for predicting the presence of heart disease using patient data. This repository contains the dataset, modeling notebook, and a saved Random Forest model.

---

## Project Overview

* **Objective**: Build and compare classification models (Decision Tree, Random Forest, SVM, Logistic Regression) to predict heart disease.
* **Approach**: Data exploration and preprocessing in a Jupyter notebook, handling class imbalance with SMOTE, hyperparameter tuning with `GridSearchCV`, and evaluation on a held-out test set.

---

## Files in this Repository

* `heart.csv`
  The raw dataset. Features include age, sex, resting blood pressure, cholesterol, fasting blood sugar, max heart rate, exercise‐induced angina, ST depression (`oldpeak`), ST slope, vessel count (`ca`), and the binary `Target` (0 = no disease, 1 = disease).

* `heart.ipynb`
  A Jupyter notebook containing:

  1. **Exploratory Data Analysis (EDA)**
  2. **Preprocessing**: one‑hot encoding, scaling, SMOTE oversampling
  3. **Model pipelines** for 4 algorithms
  4. **Hyperparameter tuning** using `GridSearchCV` (F1‑score)
  5. **Over‑fitting analysis** (train vs. test metrics)
  6. **Final model evaluation**

* `random_forst_model_1.pkl`
  A serialized **Logistic Regression** pipeline trained with the best hyperparameters, saved via `pickle.dump`.

  A serialized Random Forest model trained with the best hyperparameters.

* `Steps.txt`
  A brief outline of the steps taken during the project.

---

## Prerequisites

* Python 3.7+

* Create and activate a virtual environment:

  ```bash
  python -m venv venv
  source venv/bin/activate     # macOS/Linux
  venv\\Scripts\\activate    # Windows
  ```

* Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

---

## How to Use

1. **Explore and reproduce results**
   Open `heart.ipynb` in Jupyter Notebook:

   ```bash
   jupyter notebook heart.ipynb
   ```

2. **Load the saved model**

   ```python
   import pickle
   # Ensure the working directory contains 'random_forst_model_1.pkl'
   loaded_model = pickle.load(open('random_forst_model_1.pkl', 'rb'))
   # Evaluate on test data
   score = loaded_model.score(x_test, y_test)
   print(f"Test set score: {score}")
   ```

3. **Run grid search** (if you want to re‐tune):

   ```python
   from sklearn.model_selection import GridSearchCV
   # import your pipeline and define param_grid
   grid_search.fit(X_train, y_train)
   ```

---

## Key Results

* **Best performing model**: Logistic Regression (Test F1 ≈ 0.836)
* **Other models**: Random Forest (F1 0.800), SVC (0.745), Decision Tree (0.640).
* **Over‑fitting analysis**: Tuned pipelines reduced ΔF1 gaps significantly.

---

## Next Steps

* Further feature engineering (interaction terms, polynomial features)
* Ensemble methods (weighted voting, stacking)
* Probability calibration for improved decision thresholds
* Deployment: build an API or simple web interface

---

## License

This project is licensed under the MIT License.
