import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load scaler, model, and cluster summary
scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")
cluster_summary = pd.read_excel("cluster_statistics.xlsx", index_col=0)

# Prediction function
def predict_rfm_cluster(recency, frequency, monetary):
    # Scale the input
    scaled_values = scaler.transform([[recency, frequency, monetary]])
    
    # Predict cluster
    cluster_label = kmeans.predict(scaled_values)[0]

    # Get cluster stats
    row = cluster_summary.loc[cluster_label]
    
    # Extract stats
    avg_recency = round(row['recency_mean'], 2)
    avg_frequency = round(row['frequency_mean'], 2)
    avg_monetary = round(row['monetary_mean'], 2)
    cluster_size = int(row['recency_count'])  # same as frequency/monetary count

    # Build informative message
    message = (
        f"🎯 **You belong to customer segment #{cluster_label}** where:\n"
        f"- Customers typically purchase every **{avg_recency} days**\n"
        f"- Shop **{avg_frequency} times**, and\n"
        f"- Spend an average of **${avg_monetary}**.\n\n"
        f"👥 **Total customers in this segment:** {cluster_size}"
    )

    return {
        "predicted_cluster": int(cluster_label),
        "cluster_avg_recency": avg_recency,
        "cluster_avg_frequency": avg_frequency,
        "cluster_avg_monetary": avg_monetary,
        "cluster_size": cluster_size,
        "message": message
    }


# Page config
st.set_page_config(page_title="Customer Segment Prediction", layout="centered")

# App title and intro
st.title("🛍️ Customer Segment Prediction (RFM-Based)")

st.markdown("""
Welcome! This application helps you understand customer behavior in online retail by segmenting your customers using the powerful **RFM (Recency, Frequency, Monetary)** model and **K-Means clustering**.

💡 Just enter how recently a customer purchased, how often they shop, and how much they spend — and discover which customer segment they belong to!
""")

# Input form
with st.form("rfm_form"):
    recency = st.number_input("📅 How many days since you last purchased?", min_value=0, max_value=1000, value=30)
    frequency = st.number_input("🛒 How many times have you purchased?", min_value=0, max_value=100, value=5)
    monetary = st.number_input("💰 How much have you spent in total?", min_value=0, max_value=100000, value=1000)
    
    submitted = st.form_submit_button("🔍 Identify Customer Segment")

# Output results
if submitted:
    result = predict_rfm_cluster(recency, frequency, monetary)
    
    st.success(result["message"])

    st.markdown("### 📊 Typical Behavior in this Segment:")
    st.metric("🗓️ Average Days Between Purchases", f"{result['cluster_avg_recency']} days")
    st.metric("🛍️ Average Number of Purchases", f"{result['cluster_avg_frequency']} times")
    st.metric("💳 Average Spending", f"${result['cluster_avg_monetary']}")
