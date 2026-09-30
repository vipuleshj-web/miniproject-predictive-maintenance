import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set page layout & configuration
st.set_page_config(
    page_title="ESP32 Predictive Maintenance AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for dark glassmorphism aesthetic
st.markdown("""
    <style>
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    
    .metric-val {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-lbl {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Risk Badge Styles */
    .risk-badge-low {
        background-color: #064e3b;
        color: #34d399;
        border: 1px solid #059669;
        padding: 12px 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
    }
    
    .risk-badge-medium {
        background-color: #78350f;
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 12px 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
    }
    
    .risk-badge-high {
        background-color: #7f1d1d;
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 12px 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to load model artifacts safely
@st.cache_resource
def load_artifacts():
    try:
        model_path = os.path.join("models", "best_model.joblib")
        scaler_path = os.path.join("models", "scaler.joblib")
        mode_models_path = os.path.join("models", "failure_mode_models.joblib")
        metrics_path = os.path.join("models", "metrics_summary.json")
        
        best_model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        mode_models = joblib.load(mode_models_path)
        
        with open(metrics_path, "r") as f:
            metrics_summary = json.load(f)
            
        return best_model, scaler, mode_models, metrics_summary
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}. Please ensure models are trained first.")
        return None, None, None, None

def engineer_single_input(vib_x, vib_y, vib_z, temp, voltage):
    vib_rms = np.sqrt(vib_x**2 + vib_y**2 + vib_z**2)
    voltage_dev = abs(voltage - 12.0)
    power_factor = voltage * vib_rms
    temp_anomaly = temp - 25.0
    
    features = pd.DataFrame([{
        'Vibration_x': vib_x,
        'Vibration_y': vib_y,
        'Vibration_z': vib_z,
        'Temperature': temp,
        'Voltage': voltage,
        'Vibration_RMS': vib_rms,
        'Voltage_Deviation': voltage_dev,
        'Power_Factor_Proxy': power_factor,
        'Temp_Anomaly': temp_anomaly
    }])
    
    return features

# Main App Layout
def main():
    best_model, scaler, mode_models, metrics_summary = load_artifacts()
    
    if best_model is None:
        st.stop()
        
    # Header Banner
    st.markdown("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 30px; color: #f8fafc;">⚡ ESP32 Predictive Maintenance AI Dashboard</h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 14px;">
                Real-time sensor telemetry monitoring (ADXL345 Vibration, DS18B20 Temp, Voltage Sensor), failure risk estimation, and root-cause diagnostics.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Top Key Metrics Bar
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    
    best_name = metrics_summary.get('best_model_name', 'RandomForest_Balanced')
    best_res = metrics_summary.get('all_results', {}).get(best_name, {})
    
    with col_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{best_name.split("_")[0]}</div><div class="metric-lbl">Classifier</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{best_res.get("Accuracy", 0)*100:.1f}%</div><div class="metric-lbl">Accuracy</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{best_res.get("F1_Score", 0):.3f}</div><div class="metric-lbl">F1-Score</div></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{best_res.get("ROC_AUC", 0):.3f}</div><div class="metric-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
    with col_m5:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{best_res.get("Recall", 0)*100:.1f}%</div><div class="metric-lbl">Detection Rate</div></div>', unsafe_allow_html=True)
        
    st.write("")
    
    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔮 Real-Time Hardware Sensor Predictor", 
        "📂 Batch CSV Telemetry Inference", 
        "📊 Model Intelligence & Analytics"
    ])
    
    # --- TAB 1: REAL-TIME PREDICTOR ---
    with tab1:
        st.subheader("ESP32 Sensor Telemetry & Simulation Control")
        st.caption("Adjust sliders to simulate live ADXL345, DS18B20, and Voltage sensor inputs.")
        
        col_left, col_right = st.columns([1, 1], gap="large")
        
        # Check presets from session state
        preset = st.session_state.get('preset', 'normal')
        
        if preset == 'vibration':
            def_vx, def_vy, def_vz, def_temp, def_volt = 4.5, 4.0, 6.5, 48.0, 11.9
        elif preset == 'heat':
            def_vx, def_vy, def_vz, def_temp, def_volt = 0.1, -0.05, 1.05, 88.5, 12.1
        elif preset == 'voltage':
            def_vx, def_vy, def_vz, def_temp, def_volt = 0.1, 0.0, 1.0, 45.0, 8.2
        else:
            def_vx, def_vy, def_vz, def_temp, def_volt = 0.05, -0.02, 1.0, 42.0, 12.0

        with col_left:
            st.markdown("### 🎛️ Live ESP32 Sensor Inputs")
            
            st.markdown("#### 1️⃣ ADXL345 3-Axis Accelerometer (Vibration)")
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                vib_x = st.slider("Vibration X [g]", -10.0, 10.0, float(def_vx), step=0.1)
            with col_v2:
                vib_y = st.slider("Vibration Y [g]", -10.0, 10.0, float(def_vy), step=0.1)
            with col_v3:
                vib_z = st.slider("Vibration Z [g]", -10.0, 10.0, float(def_vz), step=0.1)
                
            st.markdown("#### 2️⃣ DS18B20 Digital Temperature Sensor")
            temp = st.slider("Machine Temperature [°C]", 10.0, 110.0, float(def_temp), step=0.5)
            
            st.markdown("#### 3️⃣ Analog Voltage Sensor Module")
            voltage = st.slider("Supply Voltage [V DC]", 0.0, 20.0, float(def_volt), step=0.1)
            
            # Preset Scenarios
            st.markdown("#### ⚡ Preset Fault Scenarios")
            c_p1, c_p2, c_p3, c_p4 = st.columns(4)
            if c_p1.button("🟢 Normal"):
                st.session_state['preset'] = 'normal'
                st.rerun()
            if c_p2.button("🟠 Bearing Vibration"):
                st.session_state['preset'] = 'vibration'
                st.rerun()
            if c_p3.button("🔴 Overheating"):
                st.session_state['preset'] = 'heat'
                st.rerun()
            if c_p4.button("⚡ Voltage Drop"):
                st.session_state['preset'] = 'voltage'
                st.rerun()

        with col_right:
            st.markdown("### 📈 Live Risk Assessment & Diagnostics")
            
            input_df = engineer_single_input(vib_x, vib_y, vib_z, temp, voltage)
            
            scaled_input = scaler.transform(input_df)
            input_scaled_df = pd.DataFrame(scaled_input, columns=metrics_summary['feature_columns'])
            
            failure_prob = best_model.predict_proba(input_scaled_df)[0][1]
            
            st.markdown("#### Machinery Health Status:")
            if failure_prob < 0.25:
                st.markdown(f'<div class="risk-badge-low">🟢 LOW RISK ({failure_prob*100:.1f}% Failure Probability)<br><span style="font-size:14px;font-weight:normal;">Machinery operating within healthy physical limits.</span></div>', unsafe_allow_html=True)
            elif failure_prob < 0.60:
                st.markdown(f'<div class="risk-badge-medium">🟠 ATTENTION REQUIRED ({failure_prob*100:.1f}% Failure Probability)<br><span style="font-size:14px;font-weight:normal;">Elevated risk. Schedule maintenance inspection soon.</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-badge-high">🚨 CRITICAL RISK ({failure_prob*100:.1f}% Failure Probability)<br><span style="font-size:14px;font-weight:normal;">High probability of breakdown! Immediate intervention recommended.</span></div>', unsafe_allow_html=True)
                
            st.write("")
            
            # Display Engineered Physics Features
            st.markdown("#### 🔬 Calculated Physics Features:")
            c_f1, c_f2, c_f3 = st.columns(3)
            c_f1.metric("Vibration RMS", f"{input_df['Vibration_RMS'].iloc[0]:.2f} g")
            c_f2.metric("Voltage Dev", f"{input_df['Voltage_Deviation'].iloc[0]:.2f} V")
            c_f3.metric("Temp Anomaly", f"{input_df['Temp_Anomaly'].iloc[0]:.1f} °C")
            
            # Failure Mode Diagnostic Breakdown
            st.markdown("#### 🔍 Diagnostic Failure Mode Probabilities:")
            mode_probs = {}
            for mode, m_clf in mode_models.items():
                m_prob = m_clf.predict_proba(input_scaled_df)[0][1]
                mode_probs[mode] = m_prob
                
            mode_labels = {
                'VAF': 'Vibration Anomaly (VAF)',
                'OHF': 'Overheating Fault (OHF)',
                'VLF': 'Voltage Fault (VLF)',
                'CSF': 'Combined Stress (CSF)'
            }
            
            mode_df = pd.DataFrame({
                'Failure Mode': [mode_labels.get(m, m) for m in mode_models.keys()],
                'Probability': [mode_probs[m] for m in mode_models.keys()]
            })
            
            fig, ax = plt.subplots(figsize=(6, 2.3))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#1e293b')
            
            bars = ax.barh(mode_df['Failure Mode'], mode_df['Probability']*100, color='#3b82f6')
            top_idx = mode_df['Probability'].argmax()
            if mode_df['Probability'].iloc[top_idx] > 0.3:
                bars[top_idx].set_color('#ef4444')
                
            ax.set_xlim(0, 100)
            ax.set_xlabel("Probability (%)", color='#94a3b8')
            ax.tick_params(colors='#f8fafc', labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            st.pyplot(fig)

    # --- TAB 2: BATCH CSV INFERENCE ---
    with tab2:
        st.subheader("📂 Batch Telemetry CSV Prediction")
        st.markdown("Upload a CSV file containing machine sensor readings (`Vibration_x`, `Vibration_y`, `Vibration_z`, `Temperature`, `Voltage`) to run automated batch risk scoring.")
        
        uploaded_file = st.file_uploader("Upload CSV Sensor File", type=["csv"])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_upload)}** rows. Data sample:")
            st.dataframe(df_upload.head(5), use_container_width=True)
            
            if st.button("🚀 Run Batch Failure Risk Prediction"):
                with st.spinner("Processing features and scoring models..."):
                    from data_preprocessing import engineer_features, FEATURE_COLUMNS
                    df_proc = engineer_features(df_upload)
                    
                    X_batch = df_proc[FEATURE_COLUMNS]
                    X_batch_scaled = scaler.transform(X_batch)
                    X_batch_scaled_df = pd.DataFrame(X_batch_scaled, columns=FEATURE_COLUMNS)
                    
                    probs = best_model.predict_proba(X_batch_scaled_df)[:, 1]
                    preds = (probs >= 0.5).astype(int)
                    
                    df_upload['Failure_Probability'] = np.round(probs, 4)
                    df_upload['Predicted_Failure'] = preds
                    df_upload['Risk_Category'] = pd.cut(
                        probs, bins=[-0.01, 0.25, 0.60, 1.0], 
                        labels=['LOW RISK', 'ATTENTION', 'CRITICAL']
                    )
                    
                    total_machines = len(df_upload)
                    high_risk_count = (df_upload['Predicted_Failure'] == 1).sum()
                    
                    st.success("Batch Prediction Complete!")
                    
                    c_b1, c_b2, c_b3 = st.columns(3)
                    c_b1.metric("Total Samples Analyzed", total_machines)
                    c_b2.metric("Predicted Failures", high_risk_count)
                    c_b3.metric("Failure Rate", f"{(high_risk_count/total_machines)*100:.2f}%")
                    
                    st.markdown("### ⚠️ Critical Maintenance Alerts")
                    critical_df = df_upload[df_upload['Risk_Category'] == 'CRITICAL'].sort_values(by='Failure_Probability', ascending=False)
                    st.dataframe(critical_df, use_container_width=True)
                    
                    csv_data = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Full Predictions CSV",
                        data=csv_data,
                        file_name="esp32_predictive_maintenance_predictions.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Tip: You can upload [`predictive_maintenance.csv`](file:///d:/miniproject/predictive_maintenance.csv) for batch testing!")

    # --- TAB 3: MODEL INTELLIGENCE ---
    with tab3:
        st.subheader("📊 Model Performance & Sensor Feature Importance")
        
        col_m_left, col_m_right = st.columns(2)
        
        with col_m_left:
            st.markdown("### 🔑 Key Physical Drivers (Feature Importance)")
            feat_imp = metrics_summary.get('feature_importance', {})
            if feat_imp:
                imp_df = pd.DataFrame({
                    'Feature': list(feat_imp.keys()),
                    'Importance': list(feat_imp.values())
                }).sort_values(by='Importance', ascending=True)
                
                fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                fig_imp.patch.set_facecolor('#1e293b')
                ax_imp.set_facecolor('#1e293b')
                
                ax_imp.barh(imp_df['Feature'], imp_df['Importance'], color='#38bdf8')
                ax_imp.set_xlabel("Importance Score", color='#94a3b8')
                ax_imp.tick_params(colors='#f8fafc')
                ax_imp.spines['top'].set_visible(False)
                ax_imp.spines['right'].set_visible(False)
                ax_imp.spines['bottom'].set_color('#334155')
                ax_imp.spines['left'].set_color('#334155')
                st.pyplot(fig_imp)
                
        with col_m_right:
            st.markdown("### 🏆 Model Metrics Comparison")
            all_res = metrics_summary.get('all_results', {})
            res_df = pd.DataFrame(all_res).T[['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'PR_AUC']]
            st.dataframe(res_df.style.highlight_max(axis=0, color='#064e3b'), use_container_width=True)
            
            st.markdown("""
                #### 💡 Model Insights:
                - **RandomForest_Balanced** achieved top **F1-Score (1.000)** and **ROC-AUC (1.000)**.
                - Physical engineered features (**Voltage_Deviation**, **Power_Factor_Proxy**, **Vibration_RMS**) serve as primary diagnostic triggers.
            """)

if __name__ == "__main__":
    main()
