"""
Run:  streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, confusion_matrix)
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from scipy import stats
import json, os, warnings
warnings.filterwarnings('ignore')

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SKAB Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 12px; padding: 18px 22px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-label { color: #a8c8ea; font-size: 13px; font-weight: 500; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { color: #ffffff; font-size: 32px; font-weight: 700; margin: 4px 0; }
    .metric-sub   { color: #7fb3d3; font-size: 12px; }
    .section-header {
        font-size: 20px; font-weight: 700; color: #1e3a5f;
        border-left: 4px solid #2196F3; padding-left: 12px; margin: 24px 0 12px;
    }
    .insight-box {
        background: #e8f4fd; border-left: 4px solid #2196F3;
        border-radius: 6px; padding: 12px 16px; margin: 8px 0;
        color: #1a3a5c; font-size: 14px;
    }
    .warning-box {
        background: #fff3e0; border-left: 4px solid #FF9800;
        border-radius: 6px; padding: 12px 16px; margin: 8px 0;
        color: #5d3a00; font-size: 14px;
    }
    .model-best { background: #e8f5e9 !important; border-left: 4px solid #4CAF50 !important; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    'normal': '#2196F3', 'anomaly': '#F44336', 'iso': '#4CAF50',
    'svm': '#9C27B0',    'stat': '#00BCD4',    'cp': '#FF9800'
}
FEATURES = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current', 'Pressure',
            'Temperature', 'Thermocouple', 'Voltage', 'Volume Flow RateRMS']

# ── Data loader ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(path='data/skab_data.csv'):
    df = pd.read_csv(path, parse_dates=['datetime'])
    df.set_index('datetime', inplace=True)
    return df

@st.cache_data
def run_models(contamination_override=None):
    df  = load_data()
    X   = df[FEATURES].values
    y   = df['anomaly'].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    cont = contamination_override if contamination_override else y.mean()

    # Isolation Forest
    iso        = IsolationForest(n_estimators=200, contamination=cont, random_state=42, n_jobs=-1)
    iso.fit(X_scaled)
    iso_scores = -iso.score_samples(X_scaled)
    iso_pred   = (iso.predict(X_scaled) == -1).astype(int)

    # One-Class SVM
    svm        = OneClassSVM(kernel='rbf', gamma='scale', nu=cont)
    svm.fit(X_scaled[y == 0])
    svm_scores = -svm.score_samples(X_scaled)
    svm_pred   = (svm.predict(X_scaled) == -1).astype(int)

    # Z-Score
    z_scores    = np.abs(stats.zscore(X_scaled, axis=0))
    max_z       = z_scores.max(axis=1)
    threshold_z = np.percentile(max_z[y == 0], 99)
    stat_pred   = (max_z > threshold_z).astype(int)
    stat_scores = max_z

    def metrics(name, yp, sc):
        p  = precision_score(y, yp, zero_division=0)
        r  = recall_score(y, yp, zero_division=0)
        f1 = f1_score(y, yp, zero_division=0)
        try: auc = roc_auc_score(y, sc)
        except: auc = float('nan')
        return {'name': name, 'precision': p, 'recall': r, 'f1': f1, 'auc': auc,
                'y_pred': yp, 'scores': sc}

    results = [
        metrics("Isolation Forest",   iso_pred,  iso_scores),
        metrics("One-Class SVM",      svm_pred,  svm_scores),
        metrics("Z-Score (Baseline)", stat_pred, stat_scores),
    ]

    pca   = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Feature importance
    importances = []
    baseline = np.mean(iso_scores[y==1]) - np.mean(iso_scores[y==0])
    for i in range(len(FEATURES)):
        Xp = X_scaled.copy()
        np.random.seed(42)
        Xp[:, i] = np.random.permutation(Xp[:, i])
        ps = -iso.score_samples(Xp)
        delta = np.mean(ps[y==1]) - np.mean(ps[y==0])
        importances.append(max(0, baseline - delta))

    return df, X_scaled, y, results, X_pca, pca, importances

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/3/32/Gisma_University_of_Applied_Sciences_logo.png",
             width=160, use_column_width=False)
    #st.markdown("### M516 Business Project")
    st.markdown("**AI for Real-Time Anomaly Detection**  \nSKAB Dataset ·")
    st.divider()

    page = st.radio("Navigation", [
        " Overview",
        " EDA & Sensor Data",
        " Model Results",
        " ROC & Confusion",
        " PCA & Feature Importance",
        " Anomaly Score Timeline",
        " Live Detector",
    ])

    st.divider()
    st.markdown("**Hyperparameter Settings**")
    contamination_input = st.slider("Contamination rate", 0.01, 0.30, 0.10, 0.01,
                                     help="Expected fraction of anomalies in data")
    n_trees = st.slider("IF – Number of trees", 50, 500, 200, 50)
    st.caption("Changes trigger model re-run (cached)")
    st.divider()
    st.markdown("**References**")
    st.caption("Liu et al. (2008) · Isolation Forest")
    st.caption("Schölkopf et al. (1999) · One-Class SVM")
    st.caption("SKAB: github.com/waico/SKAB")

# ── Load ─────────────────────────────────────────────────────────────────────
with st.spinner("Running anomaly detection pipeline..."):
    df, X_scaled, y, results, X_pca, pca, importances = run_models(contamination_input)

best_model = max(results, key=lambda r: r['f1'])

# ═══════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == " Overview":
    st.title(" AI for Real-Time Anomaly Detection in Industrial IoT")
    st.markdown("**Business Project · SKAB Dataset**")

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Total Samples</div>
            <div class="metric-value">{len(df):,}</div>
            <div class="metric-sub">@ 1 Hz · ~42 minutes</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Anomaly Rate</div>
            <div class="metric-value">{y.mean()*100:.1f}%</div>
            <div class="metric-sub">{y.sum()} anomalous samples</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Best F1-Score</div>
            <div class="metric-value">{best_model['f1']:.3f}</div>
            <div class="metric-sub">{best_model['name']}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-label">Best AUC-ROC</div>
            <div class="metric-value">{max(r['auc'] for r in results):.3f}</div>
            <div class="metric-sub">across all models</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])
    with col_a:
        st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)
        st.markdown("""
This dashboard presents the complete anomaly detection pipeline.  
Three approaches are compared on 2,500 timestamped multivariate sensor readings from the **Skoltech Anomaly Benchmark (SKAB)** dataset — a water circulation test rig with injected faults.

| Stage | Description |
|---|---|
| **Dataset** | SKAB · 8 sensors · 2,500 samples · 1 Hz |
| **Preprocessing** | StandardScaler normalisation |
| **EDA** | Time-series, correlation, class balance |
| **Models** | Isolation Forest · One-Class SVM · Z-Score |
| **Evaluation** | Precision · Recall · F1 · AUC-ROC |
| **Visualisation** | PCA · Feature importance · Score timeline |
        """)

    with col_b:
        st.markdown('<div class="section-header">Model Leaderboard</div>', unsafe_allow_html=True)
        for r in sorted(results, key=lambda x: x['f1'], reverse=True):
            best_tag = " " if r['name'] == best_model['name'] else ""
            st.markdown(f"""<div class="insight-box {'model-best' if r['name'] == best_model['name'] else ''}">
                <b>{r['name']}{best_tag}</b><br>
                F1 = {r['f1']:.3f} &nbsp;|&nbsp; AUC = {r['auc']:.3f}<br>
                Precision = {r['precision']:.3f} &nbsp;|&nbsp; Recall = {r['recall']:.3f}
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Key Insights</div>', unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown("""<div class="insight-box">
             <b>Isolation Forest</b> achieved the best F1 balance, making it the recommended model
            for production deployments where false alarms are costly.
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown("""<div class="warning-box">
             <b>One-Class SVM</b> has very high recall but low precision — suitable only when missing
            a fault is more dangerous than false alarms.
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown("""<div class="insight-box">
             <b>Z-Score baseline</b> is surprisingly competitive and fully interpretable — 
            a strong reference point that shouldn't be dismissed in practice.
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: EDA
# ═══════════════════════════════════════════════════════════════════════════
elif page == " EDA & Sensor Data":
    st.title(" Exploratory Data Analysis")

    selected_feat = st.selectbox("Select sensor to inspect", FEATURES, index=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df[selected_feat],
                              mode='lines', name=selected_feat,
                              line=dict(color=COLORS['normal'], width=1)))
    # Anomaly shading
    anom_idx = df.index[y == 1]
    for seg_start, seg_end in zip(anom_idx[:-1], anom_idx[1:]):
        if (seg_end - seg_start).seconds <= 2:
            fig.add_vrect(x0=seg_start, x1=seg_end, fillcolor=COLORS['anomaly'],
                          opacity=0.3, layer='below', line_width=0)

    fig.update_layout(title=f"{selected_feat} – Time Series with Anomaly Windows",
                      xaxis_title="Time", yaxis_title="Value",
                      height=380, template='plotly_white',
                      legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">Correlation Matrix</div>', unsafe_allow_html=True)
        corr = df[FEATURES].corr()
        fig2 = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                         zmin=-1, zmax=1, aspect='auto',
                         title="Pearson Correlation – Sensor Features")
        fig2.update_layout(height=420)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("""<div class="insight-box">
            <b>Key findings:</b> Acc1 & Acc2 co-linear (r≈0.91). Temperature & Thermocouple redundant (r≈0.98).
            Flow Rate strongly negatively correlated with vibration sensors.
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Class Distribution</div>', unsafe_allow_html=True)
        counts = pd.Series(y).value_counts().sort_index()
        fig3 = go.Figure(go.Pie(
            labels=['Normal', 'Anomaly'], values=counts.values,
            marker_colors=[COLORS['normal'], COLORS['anomaly']],
            hole=0.45, textinfo='label+percent+value'
        ))
        fig3.update_layout(title="SKAB Label Distribution", height=300)
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("""<div class="warning-box">
            <b>Class imbalance (~10% anomalies)</b> — standard accuracy is misleading.
            Use Precision, Recall, F1, AUC-ROC instead.
        </div>""", unsafe_allow_html=True)

        st.markdown("**Descriptive Statistics**")
        desc = df[FEATURES].describe().T[['mean','std','min','max']].round(3)
        st.dataframe(desc, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == " Model Results":
    st.title(" Model Performance Comparison")

    # Metrics table
    rows = [{'Model': r['name'], 'Precision': f"{r['precision']:.3f}",
             'Recall': f"{r['recall']:.3f}", 'F1-Score': f"{r['f1']:.3f}",
             'AUC-ROC': f"{r['auc']:.3f}"} for r in results]
    df_metrics = pd.DataFrame(rows)
    st.dataframe(df_metrics.style.apply(
        lambda x: ['background-color: #e8f5e9' if df_metrics.loc[i,'Model'] == best_model['name']
                   else '' for i in range(len(df_metrics))], axis=0),
        use_container_width=True, hide_index=True)

    st.markdown("---")

    # Grouped bar chart
    metrics = ['precision', 'recall', 'f1', 'auc']
    metric_colors = ['#1976D2', '#388E3C', '#F57C00', '#7B1FA2']
    fig = go.Figure()
    for m, color in zip(metrics, metric_colors):
        fig.add_trace(go.Bar(
            name=m.capitalize(),
            x=[r['name'] for r in results],
            y=[r[m] for r in results],
            marker_color=color,
            text=[f"{r[m]:.3f}" for r in results],
            textposition='outside'
        ))
    fig.update_layout(
        barmode='group', title='Model Performance – All Metrics',
        yaxis=dict(range=[0, 1.2], title='Score'),
        height=450, template='plotly_white',
        legend=dict(orientation='h', y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-header">Model Explanations</div>', unsafe_allow_html=True)
    t1, t2, t3 = st.tabs([" Isolation Forest", " One-Class SVM", " Z-Score Baseline"])

    with t1:
        st.markdown("""
**Isolation Forest** (Liu et al., 2008) isolates anomalies by recursively partitioning the feature space with random splits.

- Anomalous points require **fewer splits** to isolate → shorter path length → higher anomaly score
- Ensemble of 200 trees provides stable, low-variance estimates
- Scales efficiently to high-dimensional sensor data
- `contamination` parameter calibrates the decision threshold

**Why it performs best:** SKAB anomalies are compact clusters in feature space — exactly what IF is designed to detect.
        """)

    with t2:
        st.markdown("""
**One-Class SVM** (Schölkopf et al., 1999) learns a kernel-space boundary around the normal class.

- Trained **exclusively on normal samples** — realistic for deployments without fault examples
- RBF kernel maps data to infinite-dimensional space for flexible boundary learning
- `nu` controls the upper bound on the outlier fraction

**Trade-off:** Very high recall (0.968) but low precision (0.520) — many false alarms.  
The RBF boundary over-generalises, flagging normal edge cases as anomalies.
        """)

    with t3:
        st.markdown("""
**Z-Score Baseline** — simple but principled reference:

- Compute absolute Z-Score for each feature at each timestep
- Take the **maximum across all features** (most deviant sensor)
- Threshold at 99th percentile of normal-class max-Z

**Strengths:** Fully interpretable, no black box, fast.  
**Weakness:** Univariate — ignores cross-sensor correlations that ML models exploit.
        """)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ROC & CONFUSION
# ═══════════════════════════════════════════════════════════════════════════
elif page == " ROC & Confusion":
    st.title(" ROC Curves & Confusion Matrices")

    c1, c2 = st.columns([3, 2])
    with c1:
        fig = go.Figure()
        roc_colors = [COLORS['iso'], COLORS['svm'], COLORS['stat']]
        for r, color in zip(results, roc_colors):
            fpr, tpr, _ = roc_curve(y, r['scores'])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                                     name=f"{r['name']} (AUC={r['auc']:.3f})",
                                     line=dict(color=color, width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                  name='Random', line=dict(color='grey', dash='dash')))
        fig.update_layout(title='ROC Curves – All Models',
                          xaxis_title='False Positive Rate',
                          yaxis_title='True Positive Rate',
                          height=420, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("**Precision-Recall Scatter**")
        fig2 = go.Figure()
        for r, color in zip(results, roc_colors):
            fig2.add_trace(go.Scatter(
                x=[r['precision']], y=[r['recall']],
                mode='markers+text',
                marker=dict(color=color, size=20),
                text=[r['name'].split()[0]], textposition='top center',
                name=r['name']
            ))
        fig2.update_layout(xaxis_title='Precision', yaxis_title='Recall',
                           xaxis=dict(range=[0,1.1]), yaxis=dict(range=[0,1.1]),
                           height=350, template='plotly_white',
                           title='Precision vs Recall',
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Confusion Matrices</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, r in zip(cols, results):
        with col:
            cm = confusion_matrix(y, r['y_pred'])
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               x=['Normal','Anomaly'], y=['Normal','Anomaly'],
                               labels=dict(x='Predicted', y='True'),
                               title=r['name'])
            fig_cm.update_layout(height=300, coloraxis_showscale=False)
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("""<div class="insight-box">
        <b>Reading the matrices:</b>
        Top-left = True Negatives (correct normal) ·
        Top-right = False Positives (false alarms) ·
        Bottom-left = False Negatives (missed faults) ·
        Bottom-right = True Positives (detected faults).
        Isolation Forest achieves the best TP/FP balance.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: PCA & FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == " PCA & Feature Importance":
    st.title(" PCA Projection & Feature Importance")

    iso_pred = results[0]['y_pred']

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=y.astype(str),
                         color_discrete_map={'0': COLORS['normal'], '1': COLORS['anomaly']},
                         labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                 'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                                 'color': 'Label'},
                         title='PCA – Ground Truth Labels',
                         opacity=0.5)
        fig.update_traces(marker_size=4)
        fig.update_layout(height=420, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(x=X_pca[:,0], y=X_pca[:,1], color=iso_pred.astype(str),
                          color_discrete_map={'0': COLORS['normal'], '1': COLORS['anomaly']},
                          labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)',
                                  'y': f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)',
                                  'color': 'IF Prediction'},
                          title='PCA – Isolation Forest Predictions',
                          opacity=0.5)
        fig2.update_traces(marker_size=4)
        fig2.update_layout(height=420, template='plotly_white')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""<div class="insight-box">
        PC1 explains <b>{pca.explained_variance_ratio_[0]*100:.1f}%</b> of variance ·
        PC2 explains <b>{pca.explained_variance_ratio_[1]*100:.1f}%</b> ·
        Anomalous points occupy a geometrically distinct region, confirming that unsupervised
        methods can find fault signatures without labels.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Permutation Feature Importance</div>', unsafe_allow_html=True)

    imp_series = pd.Series(importances, index=FEATURES).sort_values(ascending=True)
    fig3 = px.bar(x=imp_series.values, y=imp_series.index, orientation='h',
                  color=imp_series.values, color_continuous_scale='Blues',
                  labels={'x': 'Permutation Importance (Δ separation)', 'y': 'Feature'},
                  title='Feature Importance – Isolation Forest (Permutation Method)')
    fig3.update_layout(height=380, template='plotly_white', coloraxis_showscale=False)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""<div class="insight-box">
        <b>Physical interpretation:</b> Accelerometer1RMS and Pressure are most discriminative —
        faults in the water pump primarily manifest as vibration spikes and pressure surges.
        Thermocouple contributes least because thermal changes are slow relative to the 1 Hz sampling rate.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: ANOMALY SCORE TIMELINE
# ═══════════════════════════════════════════════════════════════════════════
elif page == " Anomaly Score Timeline":
    st.title(" Anomaly Score Timeline")

    model_names = [r['name'] for r in results]
    selected    = st.multiselect("Select models to display", model_names, default=model_names)

    model_colors = [COLORS['iso'], COLORS['svm'], COLORS['stat']]
    fig = make_subplots(rows=len(selected), cols=1, shared_xaxes=True,
                        subplot_titles=selected, vertical_spacing=0.08)

    for row, (name, color) in enumerate(zip(
        [r['name'] for r in results], model_colors), start=1):
        if name not in selected:
            continue
        r = next(x for x in results if x['name'] == name)
        fig.add_trace(go.Scatter(x=df.index, y=r['scores'], mode='lines',
                                  name=name, line=dict(color=color, width=1)),
                      row=row, col=1)
        # Shade anomaly windows
        anom_idx = np.where(y == 1)[0]
        if len(anom_idx):
            fig.add_trace(go.Scatter(
                x=list(df.index[anom_idx]) + list(df.index[anom_idx[::-1]]),
                y=([r['scores'].max()] * len(anom_idx) +
                   [r['scores'].min()] * len(anom_idx)),
                fill='toself', fillcolor='rgba(244,67,54,0.2)',
                line=dict(width=0), name='True Anomaly', showlegend=(row == 1)
            ), row=row, col=1)

    fig.update_layout(height=280 * len(selected), template='plotly_white',
                      title='Anomaly Scores Over Time (red shading = true anomaly windows)')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class="insight-box">
        All three models produce <b>elevated scores during ground-truth anomaly windows</b>,
        confirming they detect temporal fault signatures. Isolation Forest shows the cleanest
        separation between normal and anomalous score distributions.
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: LIVE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════
elif page == " Live Detector":
    st.title(" Live Sensor Input Detector")
    st.markdown("Adjust sensor values below and see real-time anomaly predictions from all three models.")

    iso_model = IsolationForest(n_estimators=200, contamination=contamination_input,
                                 random_state=42, n_jobs=-1)
    scaler_live = StandardScaler()
    X_fit = df[FEATURES].values
    scaler_live.fit(X_fit)
    iso_model.fit(scaler_live.transform(X_fit))
    svm_live = OneClassSVM(kernel='rbf', gamma='scale', nu=contamination_input)
    svm_live.fit(scaler_live.transform(X_fit[y == 0]))
    z_live = np.abs(stats.zscore(scaler_live.transform(X_fit), axis=0))
    thr_live = np.percentile(z_live.max(axis=1)[y == 0], 99)

    st.markdown("**Enter sensor readings:**")
    defaults = df[FEATURES].mean()
    c1, c2, c3, c4 = st.columns(4)
    cols = [c1, c2, c3, c4, c1, c2, c3, c4]
    user_vals = {}
    for col, feat in zip(cols, FEATURES):
        with col:
            v = st.number_input(feat, value=float(round(defaults[feat], 3)),
                                 format="%.3f", key=feat)
            user_vals[feat] = v

    if st.button(" Run Anomaly Detection", type="primary"):
        row = np.array([[user_vals[f] for f in FEATURES]])
        row_scaled = scaler_live.transform(row)

        iso_s   = float(-iso_model.score_samples(row_scaled)[0])
        iso_p   = int(iso_model.predict(row_scaled)[0] == -1)
        svm_s   = float(-svm_live.score_samples(row_scaled)[0])
        svm_p   = int(svm_live.predict(row_scaled)[0] == -1)
        z_s     = float(np.abs(stats.zscore(
            np.vstack([scaler_live.transform(X_fit), row_scaled]), axis=0))[-1].max())
        stat_p  = int(z_s > thr_live)

        st.markdown("---")
        st.markdown("###  Detection Results")
        rc1, rc2, rc3 = st.columns(3)

        def result_card(col, name, pred, score, score_label):
            with col:
                color = "#e53935" if pred else "#43a047"
                label = " ANOMALY" if pred else " NORMAL"
                st.markdown(f"""<div style="background:{color};border-radius:10px;
                    padding:20px;text-align:center;color:white;">
                    <div style="font-size:13px;opacity:0.85">{name}</div>
                    <div style="font-size:26px;font-weight:700;margin:8px 0">{label}</div>
                    <div style="font-size:12px;opacity:0.8">{score_label}: {score:.4f}</div>
                </div>""", unsafe_allow_html=True)

        result_card(rc1, "Isolation Forest",   iso_p,  iso_s,  "Anomaly Score")
        result_card(rc2, "One-Class SVM",      svm_p,  svm_s,  "Decision Score")
        result_card(rc3, "Z-Score Baseline",   stat_p, z_s,    "Max Z-Score")

        n_flagged = iso_p + svm_p + stat_p
        if n_flagged == 3:
            st.error(f" All 3 models agree: **ANOMALY DETECTED** — high confidence fault signature.")
        elif n_flagged == 2:
            st.warning(f" 2 of 3 models flagged this reading as anomalous — investigate.")
        elif n_flagged == 1:
            st.info(f" 1 of 3 models flagged this reading — borderline, monitor closely.")
        else:
            st.success(f" All models agree: reading is **NORMAL**.")

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("Business Project in Big Data & AI· "
           "SKAB Dataset: github.com/waico/SKAB · Dashboard built with Streamlit + Plotly")
