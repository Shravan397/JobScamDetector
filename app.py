import streamlit as st
import joblib


# =========================================================
# LOAD MODEL AND VECTORIZER
# =========================================================

model = joblib.load("job_scam_logistic_regression.pkl")
vectorizer = joblib.load("job_scam_tfidf_vectorizer.pkl")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Job Scam Detector",
    page_icon="🔍",
    layout="wide"
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🔍 Job Scam Detector")

    st.write(
        "An AI-based system that analyzes job-posting text "
        "and estimates the likelihood of fraudulent content."
    )

    st.divider()

    st.subheader("🤖 How it works")

    st.write("1. Enter the job-posting information.")
    st.write("2. The text is combined into one document.")
    st.write("3. TF-IDF converts the text into numerical features.")
    st.write("4. Logistic Regression predicts fraud probability.")
    st.write("5. The system assigns a risk level.")

    st.divider()

    st.subheader("📊 Risk Levels")

    st.write("🟢 **Low:** 0–29.99")
    st.write("🟡 **Medium:** 30–59.99")
    st.write("🔴 **High:** 60–100")

    st.divider()

    st.caption(
        "Model: Logistic Regression + TF-IDF"
    )

    st.caption(
        "Classification threshold: 0.40"
    )


# =========================================================
# MAIN TITLE
# =========================================================

st.title("🔍 AI-Based Job Scam Detection and Risk Assessment System")

st.write(
    "Analyze a job advertisement using a machine learning model "
    "trained to detect potentially fraudulent job postings."
)

st.info(
    "⚠️ This system is a screening tool. A high or low risk score "
    "does not guarantee that a job is fraudulent or legitimate. "
    "Always independently verify the employer."
)


# =========================================================
# INPUT SECTION
# =========================================================

st.header("📝 Job Posting Information")

st.write(
    "Enter the available information from the job advertisement."
)

col1, col2 = st.columns(2)

with col1:

    title = st.text_input(
        "Job Title *",
        placeholder="Example: Data Entry Clerk"
    )

    company_profile = st.text_area(
        "Company Profile",
        height=150,
        placeholder="Enter information about the company..."
    )

    benefits = st.text_area(
        "Benefits / Salary",
        height=150,
        placeholder="Enter salary, benefits, incentives, etc..."
    )


with col2:

    description = st.text_area(
        "Job Description *",
        height=220,
        placeholder="Enter the complete job description..."
    )

    requirements = st.text_area(
        "Requirements",
        height=150,
        placeholder="Enter required skills, qualifications, experience..."
    )


st.caption("* Required fields")


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze = st.button(
    "🚨 Analyze Job Posting",
    use_container_width=True,
    type="primary"
)


if analyze:

    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    if not title.strip():
        st.error("Please enter the Job Title.")

    elif not description.strip():
        st.error("Please enter the Job Description.")

    else:

        # -------------------------------------------------
        # COMBINE TEXT
        # Same pipeline used during model training
        # -------------------------------------------------

        combined_text = (
            title + " " +
            company_profile + " " +
            description + " " +
            requirements + " " +
            benefits
        )

        # -------------------------------------------------
        # TF-IDF TRANSFORMATION
        # -------------------------------------------------

        features = vectorizer.transform([combined_text])

        # -------------------------------------------------
        # FRAUD PROBABILITY
        # -------------------------------------------------

        probability = model.predict_proba(features)[0][1]

        risk_score = probability * 100

        # -------------------------------------------------
        # CLASSIFICATION
        # -------------------------------------------------

        threshold = 0.40

        if probability >= threshold:
            prediction = "Potentially Fraudulent"
        else:
            prediction = "Likely Legitimate"

        # -------------------------------------------------
        # RISK LEVEL
        # -------------------------------------------------

        if risk_score < 30:
            risk_level = "Low"
        elif risk_score < 60:
            risk_level = "Medium"
        else:
            risk_level = "High"

        # =================================================
        # RESULTS
        # =================================================

        st.divider()

        st.header("📊 Analysis Result")

        # -------------------------------------------------
        # TOP METRICS
        # -------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Fraud Probability",
                f"{risk_score:.2f}%"
            )

        with col2:
            st.metric(
                "Risk Score",
                f"{risk_score:.2f} / 100"
            )

        with col3:
            st.metric(
                "Risk Level",
                risk_level
            )

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        st.subheader("Prediction")

        if prediction == "Potentially Fraudulent":

            st.error(
                f"⚠️ {prediction}"
            )

        else:

            st.success(
                f"✅ {prediction}"
            )

        # -------------------------------------------------
        # RISK BAR
        # -------------------------------------------------

        st.subheader("Risk Assessment")

        st.progress(
            min(int(risk_score), 100)
        )

        if risk_level == "High":

            st.error(
                "🔴 HIGH RISK — Exercise extreme caution. "
                "Verify the employer, recruiter, contact information "
                "and job offer before sharing personal information "
                "or making any payment."
            )

        elif risk_level == "Medium":

            st.warning(
                "🟡 MEDIUM RISK — Independently verify the employer "
                "and job details before proceeding."
            )

        else:

            st.success(
                "🟢 LOW RISK — The posting has a relatively low "
                "predicted scam risk, but independent verification "
                "is still recommended."
            )

        # =================================================
        # POSTING COMPLETENESS CHECK
        # =================================================

        st.subheader("📋 Posting Information Checks")

        check1, check2, check3 = st.columns(3)

        with check1:

            if company_profile.strip():
                st.success("✅ Company Profile provided")
            else:
                st.warning("⚠️ Company Profile missing")

        with check2:

            if requirements.strip():
                st.success("✅ Requirements provided")
            else:
                st.warning("⚠️ Requirements missing")

        with check3:

            if benefits.strip():
                st.success("✅ Benefits provided")
            else:
                st.warning("⚠️ Benefits missing")

        # =================================================
        # MODEL PERFORMANCE
        # =================================================

        st.divider()

        st.header("🤖 Model Performance")

        st.write(
            "Performance measured on the held-out test dataset."
        )

        metric1, metric2, metric3, metric4, metric5 = st.columns(5)

        with metric1:
            st.metric(
                "Accuracy",
                "97.54%"
            )

        with metric2:
            st.metric(
                "Precision",
                "69.81%"
            )

        with metric3:
            st.metric(
                "Recall",
                "86.72%"
            )

        with metric4:
            st.metric(
                "F1-Score",
                "77.35%"
            )

        with metric5:
            st.metric(
                "ROC-AUC",
                "98.67%"
            )

        st.caption(
            "Test-set results from the final Logistic Regression model."
        )

        # =================================================
# MODEL EVALUATION DASHBOARD
# =================================================

st.divider()

st.header("📈 Model Evaluation Dashboard")

st.write(
    "The final Logistic Regression model was evaluated on an "
    "untouched test dataset."
)

# Test-set performance
eval_col1, eval_col2, eval_col3 = st.columns(3)

with eval_col1:
    st.metric("Accuracy", "97.54%")
    st.metric("Precision", "69.81%")

with eval_col2:
    st.metric("Recall", "86.72%")
    st.metric("F1-Score", "77.35%")

with eval_col3:
    st.metric("ROC-AUC", "98.67%")
    st.metric("PR-AUC", "90.45%")

st.caption(
    "These values are from the final test-set evaluation performed "
    "during model development."
)

# Explanation
with st.expander("📖 What do these metrics mean?"):

    st.write(
        "**Accuracy:** Percentage of all job postings classified correctly."
    )

    st.write(
        "**Precision:** Percentage of postings predicted as fraudulent "
        "that were actually fraudulent."
    )

    st.write(
        "**Recall:** Percentage of actual fraudulent postings detected "
        "by the model."
    )

    st.write(
        "**F1-Score:** Balance between precision and recall."
    )

    st.write(
        "**ROC-AUC:** Measures how well the model separates fraudulent "
        "and legitimate postings across classification thresholds."
    )

    st.write(
        "**PR-AUC:** Measures performance using the precision-recall "
        "trade-off, which is particularly useful for imbalanced data."
    )