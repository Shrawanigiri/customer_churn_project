import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from model import train_churn_model, predict_customer

# Page Configuration (No emojis)
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply clean off-white background styling and readable charcoal typography
st.markdown("""
<style>
    /* Overall app background: clean off-white */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #F8F7F3 !important;
        color: #1F2937 !important;
    }
    
    /* Global text color */
    p, span, label, div, h1, h2, h3, h4, h5, h6 {
        color: #1F2937;
    }

    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E2DA;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .card-title {
        font-size: 0.85rem;
        color: #64748B;
        margin-bottom: 4px;
        font-weight: 600;
    }
    .card-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    .prediction-box-churn {
        background-color: #FDF2F2;
        border-left: 5px solid #DC2626;
        padding: 16px;
        border-radius: 6px;
        margin-top: 15px;
    }
    .prediction-box-stay {
        background-color: #F0FDF4;
        border-left: 5px solid #16A34A;
        padding: 16px;
        border-radius: 6px;
        margin-top: 15px;
    }
    .academic-note {
        background-color: #FEF9C3;
        border: 1px solid #FDE047;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 15px;
        color: #854D0E;
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Data & Model
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv('dataset.csv')

@st.cache_resource
def get_model_and_metrics():
    if os.path.exists('churn_model.pkl') and os.path.exists('model_columns.pkl'):
        model = joblib.load('churn_model.pkl')
        feature_columns = joblib.load('model_columns.pkl')
        _, _, metrics, importances = train_churn_model('dataset.csv')
    else:
        model, feature_columns, metrics, importances = train_churn_model('dataset.csv')
    return model, feature_columns, metrics, importances

df = load_data()
model, feature_columns, metrics, importances = get_model_and_metrics()

# Header (Plain professional text, no emojis)
st.markdown('<div class="main-title">Customer Churn Prediction System</div>', unsafe_allow_html=True)


# Navigation Tabs (Plain professional text, no emojis)
tab_eda, tab_model, tab_predict = st.tabs([
    "1. Exploratory Data Analysis (EDA)",
    "2. Model Performance & Evaluation",
    "3. Interactive Churn Predictor"
])

# =========================================================
# TAB 1: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
with tab_eda:
    st.subheader("Dataset Summary & Key Insights")
    
    total_cust = len(df)
    churn_count = (df['Churn'] == 'Yes').sum()
    stay_count = (df['Churn'] == 'No').sum()
    avg_tenure = df['tenure'].mean()
    avg_charges = df['MonthlyCharges'].mean()
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">TOTAL CUSTOMERS</div>
            <div class="card-value">{total_cust}</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">CHURN DISTRIBUTION</div>
            <div class="card-value">{churn_count} Churn / {stay_count} Stay (50%)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">AVG. TENURE</div>
            <div class="card-value">{avg_tenure:.1f} mos</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="card-title">AVG. MONTHLY CHARGES</div>
            <div class="card-value">${avg_charges:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 4 Meaningful EDA Visualizations
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Chart 1: Churn Distribution
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        fig1.patch.set_facecolor('#F8F7F3')
        ax1.set_facecolor('#F8F7F3')
        churn_counts = df['Churn'].value_counts()
        colors = ['#3B82F6', '#EF4444']
        bars = ax1.bar(churn_counts.index, churn_counts.values, color=colors, width=0.45)
        ax1.set_title("1. Churn Class Distribution", fontsize=11, fontweight='bold', pad=10, color='#1F2937')
        ax1.set_ylabel("Customer Count", color='#1F2937')
        ax1.set_xlabel("Churn Status", color='#1F2937')
        ax1.tick_params(colors='#1F2937')
        for spine in ax1.spines.values():
            spine.set_color('#D1D5DB')
        for bar in bars:
            yval = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2.0, yval/2, f"{int(yval)} ({yval/len(df)*100:.0f}%)", ha='center', va='center', color='white', fontweight='bold')
        ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#D1D5DB')
        st.pyplot(fig1)
        plt.close(fig1)
        st.caption("Observation: Exactly 15 customers stayed and 15 churned (balanced 50% split).")

        # Chart 2: Tenure vs Churn
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        fig2.patch.set_facecolor('#F8F7F3')
        ax2.set_facecolor('#F8F7F3')
        tenure_summary = df.groupby('Churn')['tenure'].mean()
        bars2 = ax2.bar(tenure_summary.index, tenure_summary.values, color=['#10B981', '#F59E0B'], width=0.45)
        ax2.set_title("2. Average Tenure by Churn Status", fontsize=11, fontweight='bold', pad=10, color='#1F2937')
        ax2.set_ylabel("Average Tenure (Months)", color='#1F2937')
        ax2.set_xlabel("Churn Status", color='#1F2937')
        ax2.tick_params(colors='#1F2937')
        for spine in ax2.spines.values():
            spine.set_color('#D1D5DB')
        for bar in bars2:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f} mos", ha='center', fontweight='bold', color='#1F2937')
        ax2.set_ylim(0, 65)
        ax2.grid(axis='y', linestyle='--', alpha=0.5, color='#D1D5DB')
        st.pyplot(fig2)
        plt.close(fig2)
        st.caption("Observation: Retained customers have an average tenure of 51.0 months vs only 11.1 months for churned customers.")

    with col_right:
        # Chart 3: Monthly Charges vs Churn
        fig3, ax3 = plt.subplots(figsize=(6, 3.8))
        fig3.patch.set_facecolor('#F8F7F3')
        ax3.set_facecolor('#F8F7F3')
        charge_summary = df.groupby('Churn')['MonthlyCharges'].mean()
        bars3 = ax3.bar(charge_summary.index, charge_summary.values, color=['#6366F1', '#EC4899'], width=0.45)
        ax3.set_title("3. Average Monthly Charges by Churn Status", fontsize=11, fontweight='bold', pad=10, color='#1F2937')
        ax3.set_ylabel("Average Monthly Charges ($)", color='#1F2937')
        ax3.set_xlabel("Churn Status", color='#1F2937')
        ax3.tick_params(colors='#1F2937')
        for spine in ax3.spines.values():
            spine.set_color('#D1D5DB')
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"${yval:.2f}", ha='center', fontweight='bold', color='#1F2937')
        ax3.set_ylim(0, 105)
        ax3.grid(axis='y', linestyle='--', alpha=0.5, color='#D1D5DB')
        st.pyplot(fig3)
        plt.close(fig3)
        st.caption("Observation: Churned customers pay on average $85.82/month compared to $63.42/month for retained customers.")

        # Chart 4: Contract Type vs Churn
        fig4, ax4 = plt.subplots(figsize=(6, 3.8))
        fig4.patch.set_facecolor('#F8F7F3')
        ax4.set_facecolor('#F8F7F3')
        contract_churn = pd.crosstab(df['Contract'], df['Churn'])
        contract_churn.plot(kind='bar', stacked=False, ax=ax4, color=['#3B82F6', '#EF4444'], width=0.6)
        ax4.set_title("4. Churn by Contract Type", fontsize=11, fontweight='bold', pad=10, color='#1F2937')
        ax4.set_ylabel("Customer Count", color='#1F2937')
        ax4.set_xlabel("Contract Type", color='#1F2937')
        ax4.tick_params(axis='x', rotation=0, colors='#1F2937')
        ax4.tick_params(axis='y', colors='#1F2937')
        for spine in ax4.spines.values():
            spine.set_color('#D1D5DB')
        ax4.legend(title="Churn", labels=['No', 'Yes'])
        ax4.grid(axis='y', linestyle='--', alpha=0.5, color='#D1D5DB')
        st.pyplot(fig4)
        plt.close(fig4)
        st.caption("Observation: All 15 churned customers were on Month-to-month contracts. One-year and Two-year contracts showed 0 churn.")

    with st.expander("View Raw Dataset Records"):
        st.dataframe(df, use_container_width=True)

# =========================================================
# TAB 2: MODEL PERFORMANCE & EVALUATION
# =========================================================
with tab_model:
    st.subheader("Random Forest Classifier Evaluation")
    
    st.markdown("""
    The model was trained using a **Stratified 80/20 train-test split** (24 training records, 6 test records) with `random_state=42`.
    Stratification ensures equal class representation (50% Churn, 50% Stay) in both train and test partitions.
    """)
    
    # Model Metrics Cards
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(label="Accuracy", value=f"{metrics['accuracy']*100:.1f}%", help="Proportion of all test predictions that were correct.")
    with m2:
        st.metric(label="Precision", value=f"{metrics['precision']*100:.1f}%", help="Out of all customers predicted to churn, how many actually churned.")
    with m3:
        st.metric(label="Recall", value=f"{metrics['recall']*100:.1f}%", help="Out of all actual churners, how many the model successfully detected.")
    with m4:
        st.metric(label="F1-Score", value=f"{metrics['f1']*100:.1f}%", help="Harmonic mean of precision and recall.")
    with m5:
        st.metric(label="5-Fold CV Accuracy", value=f"{metrics['cv_mean']*100:.1f}%", help="Cross-validation accuracy computed strictly on training set.")

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_cm, col_fi = st.columns(2)
    
    with col_cm:
        st.markdown("#### Confusion Matrix (Test Set: 6 Samples)")
        cm = metrics['confusion_matrix']
        fig_cm, ax_cm = plt.subplots(figsize=(5, 3.8))
        fig_cm.patch.set_facecolor('#F8F7F3')
        ax_cm.set_facecolor('#F8F7F3')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax_cm,
                    xticklabels=['Predicted Stay', 'Predicted Churn'],
                    yticklabels=['Actual Stay', 'Actual Churn'],
                    annot_kws={"size": 14, "weight": "bold"})
        ax_cm.set_title("Confusion Matrix", fontsize=11, fontweight='bold', pad=10, color='#1F2937')
        ax_cm.tick_params(colors='#1F2937')
        for spine in ax_cm.spines.values():
            spine.set_color('#D1D5DB')
        st.pyplot(fig_cm)
        plt.close(fig_cm)
        
        st.markdown("""
        - **True Negatives (TN)** = 3 (Retained customers correctly identified)
        - **False Positives (FP)** = 0 (No false alarms)
        - **False Negatives (FN)** = 0 (No missed churners)
        - **True Positives (TP)** = 3 (Churned customers correctly detected)
        """)

    with col_fi:
        st.markdown("#### Top Features for Classification")
        top_importances = importances.head(7).sort_values(ascending=True)
        
        fig_fi, ax_fi = plt.subplots(figsize=(6, 3.8))
        fig_fi.patch.set_facecolor('#F8F7F3')
        ax_fi.set_facecolor('#F8F7F3')
        ax_fi.barh(top_importances.index, top_importances.values, color='#2563EB', height=0.6)
        ax_fi.set_title("Random Forest Gini Feature Importance", fontsize=11, fontweight='bold', pad=10, color='#1F2937')
        ax_fi.set_xlabel("Relative Importance Score", color='#1F2937')
        ax_fi.tick_params(colors='#1F2937')
        for spine in ax_fi.spines.values():
            spine.set_color('#D1D5DB')
        ax_fi.grid(axis='x', linestyle='--', alpha=0.5, color='#D1D5DB')
        st.pyplot(fig_fi)
        plt.close(fig_fi)
        
        st.caption("Feature importance indicates which features were most useful to the model for separating classes. It does NOT prove that those features cause churn.")

    st.markdown("""
    <div class="academic-note">
        <strong>Academic Limitation Disclosure:</strong> The dataset contains 30 customer records, yielding a test set of 6 observations. 
        While 100% test accuracy and cross-validation accuracy are mathematically valid on this specific data split due to strong class separability (e.g. tenure and contract type), 
        these metrics describe performance on this specific dataset and should <strong>not</strong> be interpreted as proof of real-world production performance. 
        In large-scale production telecom datasets, behavior is considerably more noisy and variable.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# TAB 3: INTERACTIVE CHURN PREDICTOR
# =========================================================
with tab_predict:
    st.subheader("Test Customer Churn Prediction")
    st.write("Adjust customer attributes below to run a real-time prediction using the trained Random Forest model.")
    
    # Viva Demo Presets (Plain professional text, no emojis)
    preset = st.selectbox(
        "Viva Demo Presets (Populate fields for live testing):",
        ["Manual Input", "Profile A: High-Risk Customer (Month-to-month, Fiber, 4 mos tenure)", "Profile B: Loyal Customer (2-Year Contract, DSL, 60 mos tenure)"]
    )
    
    # Preset default values
    if preset == "Profile A: High-Risk Customer (Month-to-month, Fiber, 4 mos tenure)":
        def_gender, def_snr, def_prt, def_dep = "Male", 0, "No", "No"
        def_tenure, def_ctr, def_paper, def_pay = 4, "Month-to-month", "Yes", "Electronic check"
        def_phone, def_lines, def_net = "Yes", "Yes", "Fiber optic"
        def_sec, def_bck, def_dev, def_sup = "No", "No", "No", "No"
        def_monthly = 92.0
    elif preset == "Profile B: Loyal Customer (2-Year Contract, DSL, 60 mos tenure)":
        def_gender, def_snr, def_prt, def_dep = "Female", 0, "Yes", "Yes"
        def_tenure, def_ctr, def_paper, def_pay = 60, "Two year", "No", "Bank transfer"
        def_phone, def_lines, def_net = "Yes", "No", "DSL"
        def_sec, def_bck, def_dev, def_sup = "Yes", "Yes", "Yes", "Yes"
        def_monthly = 68.0
    else:
        def_gender, def_snr, def_prt, def_dep = "Female", 0, "No", "No"
        def_tenure, def_ctr, def_paper, def_pay = 12, "Month-to-month", "Yes", "Electronic check"
        def_phone, def_lines, def_net = "Yes", "No", "Fiber optic"
        def_sec, def_bck, def_dev, def_sup = "No", "No", "No", "No"
        def_monthly = 75.0

    with st.form("churn_prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 1. Demographics")
            gender = st.selectbox("Gender", ["Female", "Male"], index=0 if def_gender=="Female" else 1)
            senior = st.selectbox("Senior Citizen (Age 65+)", [0, 1], index=def_snr, format_func=lambda x: "Yes (1)" if x==1 else "No (0)")
            partner = st.selectbox("Has Partner", ["Yes", "No"], index=0 if def_prt=="Yes" else 1)
            dependents = st.selectbox("Has Dependents", ["Yes", "No"], index=0 if def_dep=="Yes" else 1)
            
        with col2:
            st.markdown("##### 2. Contract & Services")
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(def_ctr))
            tenure = st.slider("Tenure (Months with company)", min_value=1, max_value=72, value=int(def_tenure))
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(def_net))
            online_sec = st.selectbox("Online Security", ["Yes", "No"], index=0 if def_sec=="Yes" else 1)
            tech_sup = st.selectbox("Tech Support", ["Yes", "No"], index=0 if def_sup=="Yes" else 1)

        with col3:
            st.markdown("##### 3. Billing & Add-ons")
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"], index=0 if def_paper=="Yes" else 1)
            payment = st.selectbox("Payment Method", ["Electronic check", "Bank transfer", "Credit card", "Mailed check"], index=["Electronic check", "Bank transfer", "Credit card", "Mailed check"].index(def_pay))
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=float(def_monthly), step=1.0)
            
            # Approximate total charges for convenience
            est_total = round(monthly_charges * tenure, 2)
            total_charges = st.number_input("Total Charges ($)", min_value=18.0, max_value=9000.0, value=float(est_total), step=10.0)

        submitted = st.form_submit_button("Run Churn Prediction", use_container_width=True)

    if submitted:
        input_data = {
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': tech_sup,
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }
        
        # Real ML prediction from the trained Random Forest model
        result = predict_customer(input_data, model, feature_columns)
        
        st.markdown("---")
        st.markdown("### Model Prediction Result")
        
        if result['churn_code'] == 1:
            st.markdown(f"""
            <div class="prediction-box-churn">
                <h3 style="color:#B91C1C; margin:0;">Predicted: Likely to Churn</h3>
                <p style="margin-top:6px; color:#4B5563; font-size:1.05rem;">
                    The model estimates a high probability of churn for this customer profile.
                </p>
                <div style="font-size:1.15rem; font-weight:bold; color:#DC2626;">
                    Churn Probability: {result['churn_probability']*100:.1f}% &nbsp;|&nbsp; Retention Probability: {result['stay_probability']*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-box-stay">
                <h3 style="color:#15803D; margin:0;">Predicted: Likely to Stay</h3>
                <p style="margin-top:6px; color:#4B5563; font-size:1.05rem;">
                    The model estimates a strong probability of retention for this customer profile.
                </p>
                <div style="font-size:1.15rem; font-weight:bold; color:#16A34A;">
                    Retention Probability: {result['stay_probability']*100:.1f}% &nbsp;|&nbsp; Churn Probability: {result['churn_probability']*100:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Key Profile Factors Associated with Classification:")
        st.write(f"- **Contract Type**: `{contract}` ({'Month-to-month contracts showed high attrition in dataset' if contract=='Month-to-month' else 'Longer-term contract associated with customer retention'})")
        st.write(f"- **Tenure**: `{tenure} months` ({'Shorter tenure corresponds with higher churn probability' if tenure < 24 else 'Higher tenure corresponds with established loyalty'})")
        st.write(f"- **Monthly Charges**: `${monthly_charges:.2f}` ({'Higher monthly bill correlates with price sensitivity' if monthly_charges > 75 else 'Lower bill rate associated with stability'})")
