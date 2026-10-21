import streamlit as st
import pandas as pd
import joblib


# Page Configuration

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load Model

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


model = load_model()

# Title
st.title("📊 Telco Customer Churn Prediction")

st.write(
    "Predict whether a customer is likely to churn based on "
    "their demographic, service, contract, and billing information."
)


# Sidebar - Customer Information

st.sidebar.header("Customer Information")


gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior_citizen = st.sidebar.selectbox(
    "Senior Citizen",
    [0, 1]
)

partner = st.sidebar.selectbox(
    "Partner",
    ["Yes", "No"]
)

dependents = st.sidebar.selectbox(
    "Dependents",
    ["Yes", "No"]
)

tenure = st.sidebar.number_input(
    "Tenure (Months)",
    min_value=0,
    max_value=100,
    value=12
)

# Phone Services

st.sidebar.header("Phone Services")

phone_service = st.sidebar.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.sidebar.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)


# Internet Services

st.sidebar.header("Internet Services")

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.sidebar.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.sidebar.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.sidebar.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.sidebar.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.sidebar.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.sidebar.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

# Contract and Billing

st.sidebar.header("Contract & Billing")

contract = st.sidebar.selectbox(
    "Contract",
    ["Month-to-month", "One year", "Two year"]
)

paperless_billing = st.sidebar.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

# Charges

st.sidebar.header("Charges")

monthly_charges = st.sidebar.number_input(
    "Monthly Charges",
    min_value=0.0,
    value=70.0
)

total_charges = st.sidebar.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)

# Create Input DataFrame

input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [senior_citizen],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})


# Display Customer Information

st.subheader("Customer Information")

st.dataframe(
    input_data,
    use_container_width=True
)


# Prediction Button

if st.button(
    "🔍 Predict Churn",
    use_container_width=True
):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    probability_percent = probability * 100


    # Prediction Result
    st.subheader("Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ High Risk of Customer Churn"
        )

        st.write(
            f"Estimated churn probability: "
            f"**{probability_percent:.2f}%**"
        )

    else:

        st.success(
            "✅ Customer is Likely to Stay"
        )

        st.write(
            f"Estimated churn probability: "
            f"**{probability_percent:.2f}%**"
        )


    # Probability Bar

    st.subheader("Churn Probability")

    st.progress(
        probability
    )

    # Business Recommendation

    st.subheader("Business Recommendation")

    if prediction == 1:

        st.warning(
            """
            This customer has a higher probability of churning.

            Recommended actions:

            - Offer a personalized retention discount.
            - Encourage a longer-term contract.
            - Review the customer's monthly charges.
            - Provide additional support or service benefits.
            - Contact the customer proactively.
            """
        )

    else:

        st.info(
            """
            This customer has a lower probability of churning.

            Recommended actions:

            - Maintain good customer service.
            - Continue monitoring customer satisfaction.
            - Offer suitable upgrades or additional services.
            - Encourage long-term customer engagement.
            """
        )


# Footer

st.markdown("---")

st.caption(
    "Telecom Customer Churn Prediction | "
    "Machine Learning Project"
)
