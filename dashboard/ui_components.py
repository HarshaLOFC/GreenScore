import streamlit as st
import pandas as pd

# -------------------------------------------------
# Custom CSS (Green + Blue Sustainability Theme)
# -------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
        body {
            background-color: #0b1220;
        }

        .card {
            background: linear-gradient(135deg, #0f172a, #0b3a2e);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0px 6px 24px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(34, 197, 94, 0.25);
        }

        h1, h2, h3 {
            color: #e5f9f0;
        }

        p {
            color: #c7e6dc;
        }

        .metric-title {
            font-size: 14px;
            color: #93c5fd;
        }

        .metric-value {
            font-size: 28px;
            font-weight: 700;
            color: #22c55e;
        }

        .stSlider > div {
            color: #22c55e;
        }

        .stButton>button {
            background: linear-gradient(135deg, #22c55e, #3b82f6);
            color: white;
            border-radius: 10px;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            border: none;
        }

        .stButton>button:hover {
            background: linear-gradient(135deg, #16a34a, #2563eb);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------
# Intro Section
# -------------------------------------------------
def render_intro():
    st.markdown(
        """
        <div class="card">
            <h1>🌱 GreenScore</h1>
            <h3>Sustainability-Aware Benchmarking of ML Models</h3>
            <p>
                A decision-support system that evaluates machine learning models
                based on <b>performance</b>, <b>energy consumption</b>, and
                <b>carbon footprint</b>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# -------------------------------------------------
# Sidebar
# -------------------------------------------------
def render_sidebar():
    st.sidebar.header("⚙️ Evaluation Settings")

    # Dataset mode
    dataset_mode = st.sidebar.radio(
        "Select Dataset Mode",
        ["Controlled Mode (Built-in)", "Custom Dataset"]
    )

    custom_df = None
    target_column = None

    if dataset_mode == "Custom Dataset":
        st.sidebar.subheader("📂 Upload Dataset")
        uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

        if uploaded_file is not None:
            custom_df = pd.read_csv(uploaded_file)
            target_column = st.sidebar.selectbox(
                "Select Target Column",
                custom_df.columns
            )

    st.sidebar.markdown("---")

    # Model selection
    st.sidebar.subheader("🤖 Select Models")
    selected_models = {
        "Logistic Regression": st.sidebar.checkbox("Logistic Regression", True),
        "Random Forest": st.sidebar.checkbox("Random Forest", True),
        "Neural Network": st.sidebar.checkbox("Neural Network", True),
    }

    if sum(selected_models.values()) < 2:
        st.sidebar.warning("⚠️ Select at least 2 models for meaningful comparison")

    st.sidebar.markdown("---")

    # Priority sliders (step = 25)
    st.sidebar.subheader("⚖️ Priority Settings (%)")
    st.sidebar.caption(
        "Adjust priorities to observe how sustainability preferences "
        "affect the recommended model."
    )

    acc_p = st.sidebar.slider(
        "Accuracy Priority",
        0, 100, 50, step=25
    )
    energy_p = st.sidebar.slider(
        "Energy Priority",
        0, 100, 25, step=25
    )
    carbon_p = st.sidebar.slider(
        "Carbon Priority",
        0, 100, 25, step=25
    )
    time_p = st.sidebar.slider(
        "Training Time Priority",
        0, 100, 0, step=25
    )

    weights = {
        "accuracy": acc_p / 100,
        "energy": energy_p / 100,
        "carbon": carbon_p / 100,
        "time": time_p / 100
    }

    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button("🚀 Run Green Evaluation")

    return (
        dataset_mode,
        selected_models,
        weights,
        run_clicked,
        custom_df,
        target_column,
    )


# -------------------------------------------------
# KPI Cards
# -------------------------------------------------
def render_kpis(df):
    best = df.sort_values("GreenScore", ascending=False).iloc[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">🏆 Best Model</div>
                <div class="metric-value">{best['Model']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">🌿 GreenScore</div>
                <div class="metric-value">{best['GreenScore']:.2f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="card">
                <div class="metric-title">🌍 Lowest CO₂ (tons)</div>
                <div class="metric-value">{df['CO2 (tons)'].min():.6f}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# -------------------------------------------------
# Final Recommendation
# -------------------------------------------------
def render_recommendation(df):
    best = df.sort_values("GreenScore", ascending=False).iloc[0]

    st.markdown(
        f"""
        <div class="card">
            <h2>✅ Final Recommendation</h2>
            <h3>{best['Model']}</h3>
            <p>
                Given the selected sustainability priorities, this model
                provides the most balanced trade-off between predictive
                performance and environmental impact.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
