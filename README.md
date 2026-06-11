Airline Customer Satisfaction Prediction

Logistic Regression Classification Project

Project Overview:

This project analyses passenger survey data from Invistico Airline (129,880 records) to
build a binomial Logistic Regression model that predicts whether a passenger will be
satisfied or dissatisfied based on 21 features including demographics, travel class,
and in-flight service ratings.
The analysis was completed using Python and Scikit-learn, following a full machine learning
pipeline: data loading, exploratory analysis, preprocessing, feature encoding, model
training, evaluation, and business interpretation.

Dataset:
Property and Value
Source Invistico_Airline.csv
Rows 129,880
Features 21 (after encoding)
Target Variable satisfaction ( satisfied / dissatisfied )
Class Balance 54.7% satisfied, 45.3% dissatisfied
Missing Values 393 in Arrival Delay in Minutes (imputed with median)

Core Project Goals:
1. Preprocess and encode all features for classification modelling
2. Fit a Logistic Regression model with an 80/20 train/test split
3. Evaluate model performance using Confusion Matrix, Precision, and Recall
4. Interpret coefficients to identify the strongest drivers of passenger satisfaction
5. Deliver data-backed business recommendations for improving customer retention
6. 
Tools & Libraries:
Tool and Purpose
Python 3 Core language
pandas Data loading, cleaning, manipulation
NumPy Numerical operations
scikit-learn Model building, encoding, scaling, evaluation
Matplotlib Data visualisation
Seaborn Statistical charts
Jupyter Notebook Interactive analysis environment

Methodology:
1. Feature Engineering
Missing value imputation: Median used for Arrival Delay in Minutes (robust to
skew)
Label Encoding: Applied to satisfaction , Customer Type , Type of Travel , and
Class
StandardScaler: All features scaled to zero mean / unit variance to ensure proper
gradient descent convergence
2. Model
Algorithm: Binomial Logistic Regression
( sklearn.linear_model.LogisticRegression )
Solver: lbfgs (default), max_iter=2000
Train/Test split: 80% / 20%, random_state=42

Results:
Performance Metrics
Metric Score
Accuracy 82.58%
Precision 84.24%
Recall 84.09%

Confusion Matrix:
Predicted Dissatisfied Predicted Satisfied
Actually Dissatisfied 9,425 (TN) 2,250 (FP)
Actually Satisfied 2,276 (FN) 12,025 (TP)
Precision (84.24%): When the model flags a passenger as satisfied, it is correct 84%
of the time - limiting false positives (passengers incorrectly labelled as satisfied).
Recall (84.09%): The model correctly identifies 84% of all truly satisfied passengers —
minimising missed opportunities for targeted service improvements.
The balanced Precision/Recall confirms the model does not over-predict either class,
making it reliable for real business decisions.

Key Findings — Top Drivers of Satisfaction:
Based on model coefficients (after StandardScaler, so values are comparable):
Rank Feature Coefficient Direction
1 Inflight entertainment +0.987 ↑ Increases satisfaction
2 Customer Type (disloyal) -0.759 ↓ Decreases satisfaction
3 Type of Travel (personal) -0.419 ↓ Decreases satisfaction
4 On-board service +0.405 ↑ Increases satisfaction
5 Seat comfort +0.392 ↑ Increases satisfaction
6 Check-in service +0.360 ↑ Increases satisfaction
7 Ease of Online booking +0.310 ↑ Increases satisfaction
8 Leg room service +0.306 ↑ Increases satisfaction

Business Recommendations:
1. Invest in Inflight Entertainment - the single strongest positive predictor. Upgrading
content and reliability could increase satisfaction probability by ~8–12% for long-haul
passengers.
2. Improve On-Board Service - staff training focused on attentiveness and consistency,
especially in Economy class.
3. Enhance Seat Comfort - retrofitting Economy seats with better legroom targets the
most dissatisfied segment.
4. Build Loyalty Programmes - disloyal customers are significantly more dissatisfied.
Targeted re-engagement and rewards could close this gap.
5. Personalise for Leisure Travellers - personal travel passengers are less satisfied than
business travellers; family-friendly amenities and flexible check-in can improve their
experience.

Limitations & Next Steps:
Logistic Regression assumes a linear decision boundary - ensemble models (Random
Forest, XGBoost) may improve accuracy beyond 82.5%
Probability threshold (currently 0.5) could be tuned based on business cost trade-offs
between FP and FN
Feature interactions (e.g., Class × Inflight entertainment) should be explored
Model should be retrained periodically on fresh data to prevent performance drift

Repository Structure:
- logistic_regression_analysis.ipynb # Main analysis notebook (all cells executed)
- Invistico_Airline.csv # Dataset
-README.md # This file

Project submitted by Elizabeth Danladi Sabatu as part of the Logistic Regression Mini Project - DH 
