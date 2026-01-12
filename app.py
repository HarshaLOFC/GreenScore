import streamlit as st

from dashboard.ui_components import (
    inject_custom_css,
    render_intro,
    render_sidebar,
    render_kpis,
    render_recommendation,
)

from dashboard.plots import (
    plot_accuracy_vs_carbon,
    plot_greenscore_bar,
)

from utils.metrics import load_results_csv
from pipeline.run_pipeline import run_pipeline


st.set_page_config(
    page_title="GreenScore",
    page_icon="🌱",
    layout="wide",
)

inject_custom_css()
render_intro()

(
    dataset_mode,
    selected_models,
    weights,
    run_clicked,
    custom_df,
    target_column,
) = render_sidebar()


if run_clicked:

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    try:
        with st.spinner("Evaluating models using GreenScore..."):
            run_pipeline(
                dataset_mode=dataset_mode,
                selected_models=selected_models,
                weights=weights,
                uploaded_file=custom_df,
                target_column=target_column,
            )

        st.success("Green evaluation completed successfully 🌱")

    except Exception as e:
        st.error(f"❌ Error during evaluation: {e}")
        st.stop()

    results_df = load_results_csv()
    results_df = results_df.sort_values("GreenScore", ascending=False)

    st.markdown("## 📊 Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    st.markdown("## 📈 Sustainability Trade-offs")
    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(
            plot_accuracy_vs_carbon(results_df),
            use_container_width=True
        )

    with col2:
        st.plotly_chart(
            plot_greenscore_bar(results_df),
            use_container_width=True
        )

    st.markdown("## 🧠 Model Insights")
    for _, row in results_df.iterrows():
        st.markdown(
            f"""
            **{row['Model']}**
            - Accuracy: {row['Accuracy']:.3f}
            - F1 Score: {row['F1-score']:.3f}
            - Energy: {row['Energy (kWh)']:.6f} kWh
            - CO₂: {row['CO2 (tons)']:.8f} tons
            - Time: {row['Time (s)']:.2f} s
            - GreenScore: {row['GreenScore']:.2f}
            """
        )

    st.markdown("## 🏆 Summary")
    render_kpis(results_df)

    st.markdown("---")
    render_recommendation(results_df)

else:
    st.info(
        "👈 Choose a dataset, adjust sustainability priorities, "
        "and click **Run Green Evaluation** to begin."
    )
