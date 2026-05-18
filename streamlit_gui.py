"""
Streamlit Web Interface for Cr³⁺ Phosphor Expert System
Author: Snežana Đurković
Year: 2026

Modern web-based GUI - just run: streamlit run phosphor_streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import io

# Import your modules
try:
    from expert_system_scoring import PhosphorExpertSystem
    from integrated_prediction_pipeline import train_final_models, predict_with_uncertainty, set_seeds
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Cr³⁺ Phosphor Expert System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2196F3;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
    }
    .tier1 {
        background-color: #c8e6c9;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .tier2 {
        background-color: #fff9c4;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    .tier3 {
        background-color: #ffccbc;
        padding: 0.5rem;
        border-radius: 5px;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'catboost_models' not in st.session_state:
    st.session_state.catboost_models = None
if 'nn_models' not in st.session_state:
    st.session_state.nn_models = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Title
st.markdown('<div class="main-header">🔬 Cr³⁺ Phosphor Discovery Expert System</div>', unsafe_allow_html=True)

# Check modules
if not MODULES_AVAILABLE:
    st.error("""
    ⚠️ **Missing Required Modules!**
    
    Please ensure the following files are in the same directory:
    - `expert_system_scoring.py`
    - `integrated_prediction_pipeline.py`
    """)
    st.stop()

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # File uploads
    st.subheader("📁 Input Files")
    training_file = st.file_uploader("Training Dataset (.xlsx)", type=['xlsx'], key='training')
    prediction_file = st.file_uploader("Prediction Dataset (.xlsx)", type=['xlsx'], key='prediction')
    
    st.divider()
    
    # Expert system settings
    st.subheader("🎯 Target Ranges")
    
    col1, col2 = st.columns(2)
    with col1:
        dqb_min = st.number_input("Dq/B Min", value=2.8, step=0.1, format="%.1f")
        emission_min = st.number_input("Emission Min (nm)", value=650, step=10)
    with col2:
        dqb_max = st.number_input("Dq/B Max", value=3.8, step=0.1, format="%.1f")
        emission_max = st.number_input("Emission Max (nm)", value=720, step=10)
    
    st.divider()
    
    # ML settings
    st.subheader("🤖 ML Settings")
    random_state = st.number_input("Random State", value=42, step=1)
    optimize_state = st.checkbox("Optimize Random State (slow)", value=False)
    
    st.divider()
    
    # Run button
    run_pipeline = st.button("▶️ Run Complete Pipeline", type="primary", use_container_width=True)

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🏆 Top Recommendations", "📈 Statistics", "📋 Full Results"])

with tab1:
    st.header("Welcome to the Cr³⁺ Phosphor Expert System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Status", "Ready" if not st.session_state.models_trained else "Models Trained ✓")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        candidates_count = len(st.session_state.results_df) if st.session_state.results_df is not None else 0
        st.metric("Candidates Evaluated", candidates_count)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if st.session_state.results_df is not None:
            tier1_count = len(st.session_state.results_df[st.session_state.results_df['Tier'] == 'Tier 1'])
            st.metric("Tier 1 Candidates", tier1_count)
        else:
            st.metric("Tier 1 Candidates", 0)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Instructions
    with st.expander("📖 How to Use This System"):
        st.markdown("""
        ### Step-by-Step Guide:
        
        1. **Upload Files** (left sidebar):
           - Training dataset with Dq/B values
           - Prediction dataset with new candidates
        
        2. **Configure Settings**:
           - Set target Dq/B range (e.g., 2.8-3.8 for red phosphors)
           - Set target emission range (e.g., 650-720 nm)
           - Adjust random state if needed
        
        3. **Run Pipeline**:
           - Click "Run Complete Pipeline"
           - Wait for processing (may take 1-5 minutes)
        
        4. **Review Results**:
           - Check "Top Recommendations" tab for best candidates
           - View "Statistics" for overall performance
           - Download "Full Results" for detailed analysis
        
        ### Output Files:
        - Excel file with all predictions and scores
        - Text report with top 10 recommendations
        - Automatic tier classification (1-4)
        """)
    
    # Workflow diagram
    with st.expander("🔄 System Workflow"):
        st.markdown("""
        ```
        Input Data
            ↓
        ML Prediction Engine
            ├─ CatBoost (Interpretable)
            └─ Neural Network (Accurate)
            ↓
        Expert System Evaluation
            ├─ Performance Scoring
            ├─ Confidence Assessment
            ├─ Feasibility Check
            └─ Novelty Ranking
            ↓
        Tier Classification
            ├─ Tier 1: Strongly Recommend
            ├─ Tier 2: Consider
            ├─ Tier 3: Edge Cases
            └─ Tier 4: Not Recommended
            ↓
        Synthesis Recommendations
        ```
        """)

# Run pipeline logic
if run_pipeline:
    if training_file is None or prediction_file is None:
        st.error("⚠️ Please upload both training and prediction datasets!")
    else:
        with st.spinner("🔄 Running pipeline... This may take a few minutes."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Load training data
                status_text.text("Loading training data...")
                progress_bar.progress(10)
                train_df = pd.read_excel(training_file)
                X_train = train_df.iloc[:, 1:-1].values
                y_train = train_df.iloc[:, -1].values
                
                # Train models
                status_text.text("Training ensemble models...")
                progress_bar.progress(30)
                set_seeds(random_state)
                catboost_models, nn_models, scaler = train_final_models(X_train, y_train, random_state)
                
                st.session_state.catboost_models = catboost_models
                st.session_state.nn_models = nn_models
                st.session_state.scaler = scaler
                st.session_state.models_trained = True
                
                # Load prediction data
                status_text.text("Making predictions...")
                progress_bar.progress(60)
                pred_df = pd.read_excel(prediction_file)
                formulas = pred_df.iloc[:, 0].values
                X_pred = pred_df.iloc[:, 1:].values
                
                # Predict
                catboost_preds, nn_preds, uncertainties = predict_with_uncertainty(
                    X_pred, catboost_models, nn_models, scaler
                )
                
                predictions_df = pd.DataFrame({
                    'Formula': formulas,
                    'CatBoost_Pred': catboost_preds,
                    'NN_Pred': nn_preds,
                    'Uncertainty': uncertainties
                })
                
                # Expert system evaluation
                status_text.text("Running expert system evaluation...")
                progress_bar.progress(80)
                
                expert = PhosphorExpertSystem(
                    target_dqb_range=(dqb_min, dqb_max),
                    target_emission_range=(emission_min, emission_max)
                )
                
                results_df = expert.evaluate_dataset(predictions_df)
                st.session_state.results_df = results_df
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ Pipeline completed successfully! Evaluated {len(results_df)} candidates.")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Display results
if st.session_state.results_df is not None:
    results_df = st.session_state.results_df
    
    with tab2:
        st.header("🏆 Top 10 Synthesis Recommendations")
        
        top10 = results_df.head(10)
        
        for idx, row in top10.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    tier_class = "tier1" if row['Tier'] == 'Tier 1' else ("tier2" if row['Tier'] == 'Tier 2' else "tier3")
                    st.markdown(f'<div class="{tier_class}">#{idx+1} {row["Formula"]}</div>', unsafe_allow_html=True)
                
                with col2:
                    st.write(f"**Dq/B:** {row['Predicted_DqB']:.2f} ± {row['Uncertainty']:.2f}")
                    st.write(f"**Emission:** {row['Predicted_Emission_nm']:.0f} nm")
                
                with col3:
                    st.metric("Score", f"{row['Total_Score']:.1f}")
                
                st.write(f"**{row['Recommendation']}**")
                st.caption(row['Rationale'])
                st.divider()
    
    with tab3:
        st.header("📈 Statistical Analysis")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            tier1_count = len(results_df[results_df['Tier'] == 'Tier 1'])
            st.metric("Tier 1", tier1_count, help="Strongly recommended for synthesis")
        
        with col2:
            tier2_count = len(results_df[results_df['Tier'] == 'Tier 2'])
            st.metric("Tier 2", tier2_count, help="Consider for synthesis")
        
        with col3:
            tier3_count = len(results_df[results_df['Tier'] == 'Tier 3'])
            st.metric("Tier 3", tier3_count, help="Edge cases for validation")
        
        with col4:
            avg_score = results_df['Total_Score'].mean()
            st.metric("Avg Score", f"{avg_score:.1f}")
        
        st.divider()
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Score distribution
            fig = px.histogram(
                results_df,
                x='Total_Score',
                nbins=20,
                title="Score Distribution",
                labels={'Total_Score': 'Total Score'},
                color_discrete_sequence=['#2196F3']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tier distribution
            tier_counts = results_df['Tier'].value_counts()
            fig = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                title="Tier Distribution",
                color_discrete_sequence=['#4CAF50', '#FFC107', '#FF9800', '#f44336']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Emission vs Dq/B scatter
        st.subheader("Predicted Emission vs Dq/B")
        fig = px.scatter(
            results_df.head(50),
            x='Predicted_DqB',
            y='Predicted_Emission_nm',
            color='Tier',
            size='Total_Score',
            hover_data=['Formula', 'Total_Score'],
            title="Emission Wavelength vs Crystal Field Parameter",
            labels={
                'Predicted_DqB': 'Predicted Dq/B',
                'Predicted_Emission_nm': 'Predicted Emission (nm)'
            },
            color_discrete_map={
                'Tier 1': '#4CAF50',
                'Tier 2': '#FFC107',
                'Tier 3': '#FF9800',
                'Tier 4': '#f44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("📋 Full Results Table")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tier_filter = st.multiselect(
                "Filter by Tier",
                options=['Tier 1', 'Tier 2', 'Tier 3', 'Tier 4'],
                default=['Tier 1', 'Tier 2']
            )
        
        with col2:
            min_score = st.slider("Minimum Score", 0, 100, 60)
        
        with col3:
            show_rows = st.number_input("Show top N rows", min_value=10, max_value=len(results_df), value=50)
        
        # Apply filters
        filtered_df = results_df[results_df['Tier'].isin(tier_filter)]
        filtered_df = filtered_df[filtered_df['Total_Score'] >= min_score]
        filtered_df = filtered_df.head(show_rows)
        
        # Display table
        st.dataframe(
            filtered_df[[
                'Formula', 'Predicted_DqB', 'Predicted_Emission_nm',
                'Total_Score', 'Performance_Score', 'Confidence_Score',
                'Feasibility_Score', 'Tier', 'Recommendation'
            ]],
            use_container_width=True,
            height=600
        )
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # Download Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Recommendations')
            output.seek(0)
            
            st.download_button(
                label="📥 Download Full Results (Excel)",
                data=output,
                file_name="expert_system_recommendations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col2:
            # Download CSV
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv,
                file_name="expert_system_recommendations.csv",
                mime="text/csv"
            )

else:
    with tab2:
        st.info("👈 Upload files and run the pipeline to see recommendations")
    with tab3:
        st.info("👈 Upload files and run the pipeline to see statistics")
    with tab4:
        st.info("👈 Upload files and run the pipeline to see full results")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p><strong>Cr³⁺ Phosphor Discovery Expert System</strong></p>
    <p>Developed by Snežana Đurković | INN Vinča, Belgrade | 2026</p>
    <p>OMAS Group - Optical Materials and Spectroscopy</p>
</div>
""", unsafe_allow_html=True)
