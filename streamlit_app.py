import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load("random_forest_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Bankruptcy Prediction App", page_icon="⚖️", layout="wide")
st.title("⚖️ Bankruptcy Prediction System")
st.markdown("---")

# Sidebar
st.sidebar.header("Upload Company Financial Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Main Section
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Uploaded Data")
    st.dataframe(df.head())
    
    # Make Predictions
    predictions = model.predict(df)
    df["Prediction"] = predictions
    
    # Mapping Results
    df["Prediction Label"] = df["Prediction"].map({1: "Bankrupt", 0: "Not Bankrupt"})
    
    # Display Results
    st.subheader("📢 Prediction Results")
    st.dataframe(df)
    
    # Visualization Section
    st.subheader("📈 Data Visualizations")
    
    # Pie Chart for Prediction Distribution
    fig, ax = plt.subplots()
    bankrupt_count = df["Prediction Label"].value_counts()
    ax.pie(bankrupt_count, labels=bankrupt_count.index, autopct='%1.1f%%', colors=['red', 'green'], startangle=90)
    ax.set_title("Bankruptcy Prediction Distribution")
    st.pyplot(fig)
    
    # Feature Correlation Heatmap
    st.subheader("🔍 Feature Correlation")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    
    # Bar Chart for Feature Importance
    st.subheader("🚀 Feature Importance")
    feature_importance = model.feature_importances_
    feature_names = df.columns[:-2]  # Exclude prediction columns
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=feature_importance, y=feature_names, palette="viridis", ax=ax)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    st.pyplot(fig)
    
    st.success("✅ Prediction Completed Successfully!")
else:
    st.info("📂 Please upload a CSV file to get started!")
