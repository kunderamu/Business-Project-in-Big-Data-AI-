# M516 – AI for Real-Time Anomaly Detection in Industrial IoT
**Gisma University of Applied Sciences | Business Project in Big Data & AI**

## Project Structure
```
m516_project/
├── M516_SKAB_Anomaly_Detection.ipynb   ← Full analysis notebook
├── streamlit_app.py                     ← Interactive dashboard
├── data/
│   └── skab_data.csv                    ← SKAB dataset
├── figures/                             ← All generated plots + summary_stats.json
└── requirements.txt
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Jupyter Notebook
```bash
jupyter notebook M516_SKAB_Anomaly_Detection.ipynb
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

## Dataset
- **SKAB (Skoltech Anomaly Benchmark)**: https://github.com/waico/SKAB
- 2,500 samples @ 1 Hz · 8 sensor features · ~10% anomaly rate
- Source: Water circulation test rig with injected faults

## Models
| Model | F1 | AUC |
|---|---|---|
| Isolation Forest | 0.884 | 0.978 |
| One-Class SVM | 0.677 | 0.986 |
| Z-Score Baseline | 0.800 | 0.943 |

## References
- Liu et al. (2008) – Isolation Forest, ICDM
- Schölkopf et al. (1999) – One-Class SVM, Neural Computation
- Filonov et al. (2021) – SKAB, github.com/waico/SKAB
