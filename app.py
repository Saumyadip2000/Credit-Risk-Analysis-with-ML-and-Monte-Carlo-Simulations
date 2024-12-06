import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Define utility functions
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

@st.cache_resource
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def probability_to_fico_score(probability, min_score=300, max_score=850):
    score_range = max_score - min_score
    return min_score + score_range * (1 - (probability ** 0.5))

# Sidebar - User Inputs
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.sidebar.write("Using default dataset (credit_risk_dataset.csv)")
    df = pd.read_csv("credit_risk_dataset.csv")

# Main Section
st.title("Credit Risk Analysis and Scoring")
st.write("Explore credit risk predictions and Monte Carlo simulations.")

# Display raw data
if st.checkbox("Show raw data"):
    st.write(df.head())

# Encode categorical variables
st.header("Preprocessing Data")
label_encoders = {}
categorical_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split data
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
st.header("Train Random Forest Classifier")
if st.button("Train Model"):
    clf = train_random_forest(X_train, y_train)
    st.write("Model trained successfully!")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    st.write("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    st.write("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

    # Map probabilities to FICO scores
    st.header("Credit Scoring")
    df_test = pd.DataFrame(X_test, columns=X.columns)
    df_test['Predicted Default Probability'] = y_proba
    df_test['FICO Credit Score'] = df_test['Predicted Default Probability'].apply(probability_to_fico_score)
    st.write(df_test[['Predicted Default Probability', 'FICO Credit Score']].head())

    # Visualize FICO scores
    st.header("FICO Credit Score Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_test['FICO Credit Score'], bins=30, kde=True, color='green', ax=ax)
    st.pyplot(fig)

    # Download results
    csv = df_test.to_csv(index=False)
    st.download_button("Download Credit Scoring Results", csv, "credit_scoring_results.csv", "text/csv")

# Monte Carlo Simulations
st.header("Monte Carlo Simulations")
if st.checkbox("Run Monte Carlo Simulations"):
    # Example simulation for income
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

    # Visualize simulation
    st.write("Monte Carlo Simulation of Income")
    fig, ax = plt.subplots()
    for i in range(10):
        ax.plot(income_paths[:, i], alpha=0.5)
    ax.set_title("Monte Carlo Simulation of Income")
    ax.set_xlabel("Days")
    ax.set_ylabel("Income")
    st.pyplot(fig)
