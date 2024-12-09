import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import confusion_matrix

# utility functions
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

@st.cache_resource
def train_xgb_model(X_train, y_train):
    model = xgb.XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X_train, y_train)
    return model

def probability_to_fico_score(probability, min_score=300, max_score=850):
    score_range = max_score - min_score
    return min_score + score_range * (1 - (probability ** 0.5))

# Inputs
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.sidebar.write("Using default dataset (credit_risk_dataset.csv)")
    df = pd.read_csv("credit_risk_dataset.csv")
    
st.title("Credit Risk Analysis and Scoring")
st.write("Explore credit risk predictions and Monte Carlo simulations.")

#  raw data
if st.checkbox("Show raw data"):
    st.write(df.head())

#  categorical variables
st.header("Preprocessing Data")
label_encoders = {}
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le


X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training Random Forest Classifier
st.header("Train Random Forest Classifier")
if st.button("Train Random Forest Model"):
    clf = train_random_forest(X_train, y_train)
    st.write("Model trained successfully!")
    y_pred_rf = clf.predict(X_test)
    y_proba_rf = clf.predict_proba(X_test)[:, 1]
    st.write("Random Forest Model Classification Report:")
    st.text(classification_report(y_test, y_pred_rf))
    st.write("ROC-AUC Score:", roc_auc_score(y_test, y_proba_rf))

    # Mapping probabilities to FICO scores
    st.header("Credit Scoring (Random Forest)")
    df_test_rf = pd.DataFrame(X_test, columns=X.columns)
    df_test_rf['Predicted Default Probability'] = y_proba_rf
    df_test_rf['FICO Credit Score'] = df_test_rf['Predicted Default Probability'].apply(probability_to_fico_score)
    st.write(df_test_rf[['Predicted Default Probability', 'FICO Credit Score']].head())

    # Visualization of FICO scores
    st.header("FICO Credit Score Distribution (Random Forest)")
    fig, ax = plt.subplots()
    sns.histplot(df_test_rf['FICO Credit Score'], bins=30, kde=True, color='green', ax=ax)
    st.pyplot(fig)

    csv_rf = df_test_rf.to_csv(index=False)
    st.download_button("Download Credit Scoring Results (Random Forest)", csv_rf, "credit_scoring_results_rf.csv", "text/csv")

# Training XGBoost Model
st.header("Train XGBoost Model")
if st.button("Train XGBoost Model"):
    xgb_model = train_xgb_model(X_train, y_train)
    st.write("XGBoost Model trained successfully!")
    y_pred_xgb = xgb_model.predict(X_test)
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]  # Ensure this is calculated correctly
    st.write("XGBoost Model Classification Report:")
    st.text(classification_report(y_test, y_pred_xgb))
    st.write("ROC-AUC Score:", roc_auc_score(y_test, y_proba_xgb))
 
    # Mapping probabilities to FICO scores
    st.header("Credit Scoring (XGBoost)")
    df_test_xgb = pd.DataFrame(X_test, columns=X.columns)
    df_test_xgb['Predicted Default Probability'] = y_proba_xgb
    df_test_xgb['FICO Credit Score'] = df_test_xgb['Predicted Default Probability'].apply(probability_to_fico_score)
    st.write(df_test_xgb[['Predicted Default Probability', 'FICO Credit Score']].head())

    # Visualization FICO scores
    st.header("FICO Credit Score Distribution (XGBoost)")
    fig, ax = plt.subplots()
    sns.histplot(df_test_xgb['FICO Credit Score'], bins=30, kde=True, color='blue', ax=ax)
    st.pyplot(fig)

    csv_xgb = df_test_xgb.to_csv(index=False)
    st.download_button("Download Credit Scoring Results (XGBoost)", csv_xgb, "credit_scoring_results_xgb.csv", "text/csv")

# Threshold-Based Credit Decision
st.header("Threshold-Based Credit Decision")
threshold = st.slider("Select Default Probability Threshold", 0.0, 1.0, 0.5)
# Ensure XGBoost model predictions exist before accessing y_proba_xgb
if 'y_proba_xgb' in locals() and y_proba_xgb is not None:
    if len(y_proba_xgb) > 0: 
        # Apply threshold to make loan decisions
        decisions = ["Loan Sanctioned" if prob >= threshold else "Loan Rejected" for prob in y_proba_xgb]

        # DataFrame with decision results
        decision_df = pd.DataFrame({
            'Predicted Default Probability': y_proba_xgb,
            'Loan Decision': decisions
        })
        
        sample_decisions = decision_df.sample(n=10, random_state=42)  # Adjust n to 10 for the desired number of samples
        
        # Displaying the sampled decisions in a table
        st.write("Sample of 10 Loan Decisions based on Threshold:")
        st.write(sample_decisions)

        # Converting the sampled decisions to CSV for download
        csv_decisions = sample_decisions.to_csv(index=False)
        st.download_button(
            label="Download Loan Decisions",
            data=csv_decisions,
            file_name="loan_decisions_sample.csv",
            mime="text/csv"
        )

    else:
        st.warning("No predictions available. Train the XGBoost model and retry.")
else:
    st.warning("Please train the XGBoost model first to make a threshold-based decision.")

# Monte Carlo Simulations
st.header("Monte Carlo Simulations")
if st.checkbox("Run Monte Carlo Simulations"):
    # simulation for income
    def monte_carlo_income_simulation(S0, mu, sigma, T, steps, simulations):
        dt = T / steps
        paths = np.zeros((steps, simulations))
        paths[0] = S0
        for t in range(1, steps):
            Z = np.random.standard_normal(simulations)
            paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        return paths

    # Parameters
    S0 = df['person_income'].mean()
    mu = 0.02
    sigma = 0.1
    T = 1
    steps = 252
    simulations = 500

    income_paths = monte_carlo_income_simulation(S0, mu, sigma, T, steps, simulations)

    # Visualization simulation
    st.write("Monte Carlo Simulation of Income")
    fig, ax = plt.subplots()
    for i in range(10):
        ax.plot(income_paths[:, i], alpha=0.5)
    ax.set_title("Monte Carlo Simulation of Income")
    ax.set_xlabel("Days")
    ax.set_ylabel("Income")
    st.pyplot(fig)

# EDA Visualization - Correlation and Distribution
st.header("Exploratory Data Analysis (EDA)")
if st.checkbox("Show Correlation Matrix"):
    corr_matrix = df.corr()
    st.write(corr_matrix)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

if st.checkbox("Show Distribution Plots for Numeric Columns"):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.write(f"Distribution of {col}")
        st.pyplot(fig)


