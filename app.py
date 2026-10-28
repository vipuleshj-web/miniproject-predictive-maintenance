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
    page_title="SmartMill ESP32 Predictive Maintenance AI Dashboard",
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
        font-size: 26px;
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
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
    }
    
    .risk-badge-medium {
        background-color: #78350f;
        color: #fbbf24;
        border: 1px solid #d97706;
        padding: 14px 20px;
        border-radius: 12px;
        font-size: 20px;
        font-weight: 700;
        text-align: center;
    }
    
    .risk-badge-high {
        background-color: #7f1d1d;
        color: #f87171;
        border: 1px solid #dc2626;
        padding: 14px 20px;
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
        st.error(f"Error loading model artifacts: {str(e)}. Please run train_model.py first.")
        return None, None, None, None

def engineer_single_input(temp, vibration, voltage_adc):
    voltage_volts = round((voltage_adc / 4095.0) * 25.0, 2)
    voltage_dev = round(abs(voltage_volts - 12.0), 3)
    temp_anomaly = round(temp - 25.0, 2)
    temp_excess = max(0.0, temp - 50.0)
    stress_index = round((vibration * 2.5) + (voltage_dev * 1.5) + (temp_excess * 0.1), 3)
    
    features = pd.DataFrame([{
        'temperature': temp,
        'vibration': vibration,
        'voltage': voltage_adc,
        'voltage_volts': voltage_volts,
        'voltage_dev': voltage_dev,
        'temp_anomaly': temp_anomaly,
        'stress_index': stress_index
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
            <h1 style="margin: 0; font-size: 28px; color: #f8fafc;">⚡ SmartMill ESP32 Predictive Maintenance AI Dashboard</h1>
            <p style="margin: 6px 0 0 0; color: #94a3b8; font-size: 14px;">
                Real-time ESP32 telemetry ingestion (DS18B20 Temp, Digital Vibration, 0-25V Voltage ADC), live failure risk assessment, and root-cause diagnostics.
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Top Key Metrics Bar
    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    
    best_name = metrics_summary.get('best_model_name', 'XGBoost_Weighted')
    best_res = metrics_summary.get('all_results', {}).get(best_name, {})
    
    with col_m1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{best_name.split("_")[0]}</div><div class="metric-lbl">Active Classifier</div></div>', unsafe_allow_html=True)
    with col_m2:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{best_res.get("Accuracy", 0)*100:.1f}%</div><div class="metric-lbl">Model Accuracy</div></div>', unsafe_allow_html=True)
    with col_m3:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{best_res.get("F1_Score", 0):.3f}</div><div class="metric-lbl">F1-Score</div></div>', unsafe_allow_html=True)
    with col_m4:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{best_res.get("ROC_AUC", 0):.3f}</div><div class="metric-lbl">ROC-AUC</div></div>', unsafe_allow_html=True)
    with col_m5:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{best_res.get("Recall", 0)*100:.1f}%</div><div class="metric-lbl">Fault Recall</div></div>', unsafe_allow_html=True)
        
    st.write("")
    
    # Navigation Tabs
    tab0, tab1, tab2, tab3 = st.tabs([
        "📡 Live ESP32 Telemetry Feed",
        "🎛️ Hardware Telemetry Simulator", 
        "📂 Batch Telemetry CSV Scoring", 
        "📊 Model Intelligence & Diagnostics"
    ])
    
    # --- TAB 0: LIVE ESP32 TELEMETRY ---
    with tab0:
        st.subheader("📡 Live ESP32 Telemetry Monitor")
        st.caption("Receiving live sensor broadcasts from ESP32 via FastAPI server on `http://<your-ip>:8000/sensor-data`.")
        
        telemetry_file = "live_telemetry.json"
        
        col_ctrl1, col_ctrl2 = st.columns([1, 4])
        with col_ctrl1:
            if st.button("🔄 Refresh Telemetry"):
                st.rerun()
                
        if os.path.exists(telemetry_file):
            try:
                with open(telemetry_file, "r") as f:
                    telemetry_data = json.load(f)
            except Exception:
                telemetry_data = []
        else:
            telemetry_data = []
            
        if telemetry_data:
            latest = telemetry_data[-1]
            
            c_l1, c_l2, c_l3, c_l4 = st.columns(4)
            c_l1.metric("Latest Temperature", f"{latest.get('temperature', 0):.1f} °C")
            c_l2.metric("Vibration Trigger", "🚨 ACTIVE (1)" if latest.get('vibration') == 1 else "🟢 NORMAL (0)")
            c_l3.metric("Voltage Supply", f"{latest.get('voltage_volts', 0):.2f} V", f"ADC: {latest.get('voltage_adc', 0)}")
            c_l4.metric("Last Updated", latest.get('timestamp', 'N/A'))
            
            st.write("")
            
            # Health Banner
            health = latest.get('machine_health', 'NORMAL')
            risk_pct = latest.get('failure_probability', 0.0) * 100
            fault = latest.get('detected_fault', 'None')
            
            if health == "NORMAL":
                st.markdown(f'<div class="risk-badge-low">🟢 SYSTEM NORMAL ({risk_pct:.1f}% Failure Probability)<br><span style="font-size:14px;font-weight:normal;">No anomalies detected. Machine running within physical specifications.</span></div>', unsafe_allow_html=True)
            elif health == "WARNING":
                st.markdown(f'<div class="risk-badge-medium">🟠 ATTENTION REQUIRED ({risk_pct:.1f}% Failure Probability)<br><span style="font-size:14px;font-weight:normal;">Fault Mode Detected: <b>{fault}</b>. Inspection recommended.</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-badge-high">🚨 CRITICAL FAULT ({risk_pct:.1f}% Failure Probability)<br><span style="font-size:14px;font-weight:normal;">Severe condition detected: <b>{fault}</b>! Immediate shutdown or maintenance advised.</span></div>', unsafe_allow_html=True)
                
            st.write("")
            st.markdown("### 📜 Recent Telemetry Logs (Last 20 Readings)")
            df_tel = pd.DataFrame(telemetry_data[-20:]).iloc[::-1]
            st.dataframe(df_tel, use_container_width=True)
        else:
            st.info("💡 Waiting for incoming data from ESP32. Ensure `api_server.py` is running and your ESP32 is powered and connected to Wi-Fi.")

    # --- TAB 1: HARDWARE SIMULATOR ---
    with tab1:
        st.subheader("🎛️ Interactive Hardware Sensor Simulator")
        st.caption("Manually test ML model responses by tweaking DS18B20 temperature, digital vibration, and voltage ADC readings.")
        
        col_left, col_right = st.columns([1, 1], gap="large")
        
        preset = st.session_state.get('preset', 'normal')
        
        if preset == 'vibration':
            def_temp, def_vib, def_adc = 49.0, 1, 1966
        elif preset == 'heat':
            def_temp, def_vib, def_adc = 85.0, 0, 1950
        elif preset == 'voltage_low':
            def_temp, def_vib, def_adc = 42.0, 0, 1350
        elif preset == 'voltage_high':
            def_temp, def_vib, def_adc = 43.0, 0, 2750
        elif preset == 'combined':
            def_temp, def_vib, def_adc = 76.0, 1, 1420
        else:
            def_temp, def_vib, def_adc = 42.0, 0, 1966

        with col_left:
            st.markdown("### 🎛️ Sensor Inputs")
            
            temp_in = st.slider("1️⃣ DS18B20 Temperature (°C)", 15.0, 110.0, float(def_temp), step=0.5)
            
            vib_toggle = st.selectbox(
                "2️⃣ Vibration Sensor (Instantaneous Digital Pin 27)",
                options=[0, 1],
                index=int(def_vib),
                format_func=lambda x: "0 - Normal / Quiet (No shock)" if x == 0 else "1 - Shock / Vibration Triggered"
            )
            
            adc_in = st.slider("3️⃣ Voltage Module Raw 12-bit ADC (Pin 34)", 0, 4095, int(def_adc), step=25)
            calculated_v = (adc_in / 4095.0) * 25.0
            st.caption(f"Equivalent Calibrated Voltage: **{calculated_v:.2f} V DC** (Nominal: 12.0V)")
            
            st.markdown("#### ⚡ Quick Fault Scenarios")
            c_p1, c_p2, c_p3, c_p4, c_p5 = st.columns(5)
            if c_p1.button("🟢 Normal"):
                st.session_state['preset'] = 'normal'
                st.rerun()
            if c_p2.button("🟠 Vibration"):
                st.session_state['preset'] = 'vibration'
                st.rerun()
            if c_p3.button("🔴 Overheat"):
                st.session_state['preset'] = 'heat'
                st.rerun()
            if c_p4.button("⚡ Voltage Sag"):
                st.session_state['preset'] = 'voltage_low'
                st.rerun()
            if c_p5.button("💥 Combined"):
                st.session_state['preset'] = 'combined'
                st.rerun()

        with col_right:
            st.markdown("### 📈 ML Failure Risk & Diagnostic Assessment")
            
            feat_df = engineer_single_input(temp_in, vib_toggle, adc_in)
            
            scaled_feat = scaler.transform(feat_df[metrics_summary['feature_columns']])
            scaled_feat_df = pd.DataFrame(scaled_feat, columns=metrics_summary['feature_columns'])
            
            prob = best_model.predict_proba(scaled_feat_df)[0][1]
            
            st.markdown("#### Machinery Health Status:")
            if prob < 0.25:
                st.markdown(f'<div class="risk-badge-low">🟢 LOW RISK ({prob*100:.1f}% Failure Probability)<br><span style="font-size:14px;font-weight:normal;">Normal operating state. All sensor values healthy.</span></div>', unsafe_allow_html=True)
            elif prob < 0.60:
                st.markdown(f'<div class="risk-badge-medium">🟠 ATTENTION REQUIRED ({prob*100:.1f}% Failure Probability)<br><span style="font-size:14px;font-weight:normal;">Elevated risk. Early sign of wear or parameter drift.</span></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-badge-high">🚨 CRITICAL RISK ({prob*100:.1f}% Failure Probability)<br><span style="font-size:14px;font-weight:normal;">High probability of breakdown! Immediate intervention needed.</span></div>', unsafe_allow_html=True)
                
            st.write("")
            st.markdown("#### 🔬 Physics-Informed Feature Values:")
            c_f1, c_f2, c_f3 = st.columns(3)
            c_f1.metric("Calculated Voltage", f"{feat_df['voltage_volts'].iloc[0]:.2f} V")
            c_f2.metric("Voltage Deviation", f"{feat_df['voltage_dev'].iloc[0]:.2f} V")
            c_f3.metric("Composite Stress Index", f"{feat_df['stress_index'].iloc[0]:.2f}")
            
            st.markdown("#### 🔍 Root-Cause Diagnostic Failure Modes:")
            mode_labels = {
                'VAF': 'Vibration Anomaly (VAF)',
                'OHF': 'Overheating Fault (OHF)',
                'VLF': 'Voltage Sag/Surge Fault (VLF)',
                'CSF': 'Combined Stress Failure (CSF)'
            }
            
            mode_probs = {}
            for mode, m_clf in mode_models.items():
                mode_probs[mode] = m_clf.predict_proba(scaled_feat_df)[0][1]
                
            mode_df = pd.DataFrame({
                'Failure Mode': [mode_labels.get(m, m) for m in mode_models.keys()],
                'Probability': [mode_probs[m] for m in mode_models.keys()]
            })
            
            fig, ax = plt.subplots(figsize=(6, 2.3))
            fig.patch.set_facecolor('#1e293b')
            ax.set_facecolor('#1e293b')
            
            bars = ax.barh(mode_df['Failure Mode'], mode_df['Probability']*100, color='#38bdf8')
            top_idx = mode_df['Probability'].argmax()
            if mode_df['Probability'].iloc[top_idx] > 0.35:
                bars[top_idx].set_color('#ef4444')
                
            ax.set_xlim(0, 100)
            ax.set_xlabel("Confidence (%)", color='#94a3b8')
            ax.tick_params(colors='#f8fafc', labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#334155')
            ax.spines['left'].set_color('#334155')
            st.pyplot(fig)

    # --- TAB 2: BATCH CSV INFERENCE ---
    with tab2:
        st.subheader("📂 Batch Telemetry CSV Scoring")
        st.markdown("Upload a CSV file containing machine sensor columns (`temperature`, `vibration`, `voltage`) to run automated batch risk evaluation.")
        
        uploaded_file = st.file_uploader("Upload CSV Telemetry File", type=["csv"])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df_upload)}** records. Preview:")
            st.dataframe(df_upload.head(5), use_container_width=True)
            
            if st.button("🚀 Run Batch Predictive Scoring"):
                with st.spinner("Scoring dataset..."):
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
                    
                    total = len(df_upload)
                    failures = (df_upload['Predicted_Failure'] == 1).sum()
                    
                    st.success("Batch Prediction Complete!")
                    c_b1, c_b2, c_b3 = st.columns(3)
                    c_b1.metric("Total Records", total)
                    c_b2.metric("Predicted Failures", failures)
                    c_b3.metric("Failure Rate", f"{(failures/total)*100:.2f}%")
                    
                    st.markdown("### ⚠️ Flagged Critical Samples")
                    crit = df_upload[df_upload['Risk_Category'] == 'CRITICAL'].sort_values(by='Failure_Probability', ascending=False)
                    st.dataframe(crit, use_container_width=True)
                    
                    csv_data = df_upload.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Scored CSV",
                        data=csv_data,
                        file_name="esp32_predictive_maintenance_predictions.csv",
                        mime="text/csv"
                    )
        else:
            st.info("Tip: You can upload [`predictive_maintenance.csv`](file:///d:/miniproject/predictive_maintenance.csv) for immediate testing!")

    # --- TAB 3: MODEL INTELLIGENCE ---
    with tab3:
        st.subheader("📊 Model Performance & Feature Importance")
        
        col_m_left, col_m_right = st.columns(2)
        
        with col_m_left:
            st.markdown("### 🔑 Key Physical Feature Importances")
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
            st.markdown("### 🏆 Model Comparison Metrics")
            all_res = metrics_summary.get('all_results', {})
            res_df = pd.DataFrame(all_res).T[['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'PR_AUC']]
            st.dataframe(res_df.style.highlight_max(axis=0, color='#064e3b'), use_container_width=True)

if __name__ == "__main__":
    main()
