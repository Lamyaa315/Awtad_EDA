# AWTAD: Quadriceps EMG Imbalance – Data Exploration & Dashboard

Exploratory data analysis and an interactive Streamlit dashboard for surface EMG recorded from the left and right
quadriceps during the barbell back squat. The goal is to find which signal features relate to a left/right quadriceps
imbalance.


## Dataset https://drive.google.com/drive/folders/1Afb87aRWru7r_N3qjg7K-RF-SBE913Kw?usp=sharing

- **Source:** collected by the AWTAD team (graduation project). 33 female participants, each recording one or two sets
  of 5 barbell back squats.
- **Sensors:** two Delsys Trigno EMG sensors (left and right quadriceps), 2148 Hz. Markers in each file mark rep
  boundaries.
- **Label:** Symmetry Index, `SI = |RMS_L − RMS_R| / max(RMS_L, RMS_R)`. A rep is an **imbalance** if SI ≥ 0.15.
- **Cleaned dataset:** `app/clean_data.csv`, 314 rows (one per rep) and 99 columns (features of both legs).

## Project structure

```
AWTAD_Unit3/
├── notebook/
│   ├── AWTAD_EDA_and_Modeling.ipynb   # the report + code: loading, cleaning, EDA, model, dashboard overview
│   ├── AWTAD_EDA_and_Modeling.html    # the same notebook with outputs, viewable in a browser
│   └── figures/                       # dashboard screenshots used in section 8
├── app/
│   ├── app.py                         # Streamlit dashboard (entry point)
│   ├── requirements.txt               # packages for the dashboard (used by Streamlit Cloud)
│   ├── viewer3d.html                  # 3D squat and muscle viewer (Three.js)
│   ├── clean_data.csv                 # cleaned feature table (written by the notebook)
│   ├── envelopes.json                 # left/right EMG envelope of every set, 25 Hz
│   ├── results_*.csv, signal_example*.csv   # results and one example recording, written by the notebook
│   ├── assets/                        # logo and illustrations
│   └── .streamlit/config.toml         # AWTAD theme
├── tools/
│   ├── export_envelopes.py            # rebuilds app/envelopes.json from the raw files
│   └── embed_images.py                # stores the dashboard screenshots inside the notebook
├── data/raw/                          # raw Trigno CSVs, one folder per participant (not on GitHub)
├── requirements.txt                   # everything, including Jupyter for the notebook
└── README.md
```

## How to run

1. Install the requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the dashboard (from the `app` folder):

   ```bash
   cd app
   streamlit run app.py
   ```

   It opens at http://localhost:8501.

The dashboard uses the AWTAD visual identity (plum, peach and blush on a warm off-white background). Every number
comes from the notebook: read from the files it writes into `app/`, recomputed with the notebook's exact method, or
copied from its printed tables. Extra checks that are not in the notebook are labelled "dashboard check".

- **Sidebar:** page navigation, a short dataset description, and filters (participants multiselect, set dropdown, SI
  range slider, class multiselect, reset button). The filters apply to the EDA page.
- **Overview:** headline numbers, SI distribution, set 1 vs set 2 stability, top features by AUC, true vs predicted SI
  and four key findings.
- **EDA:** tabs for the dataset (shape, column groups, preview, statistics), distributions (classes, weaker leg, SI per
  participant, any feature, fatigue), feature vs target (correlation ranking, box plot, scatter, left vs right,
  between-participant share), correlations (heatmap and redundant pairs) and outliers (box plot and 1.5 × IQR counts).
- **Data quality & preprocessing:** an interactive 3D squat (Three.js, rigged Mixamo character) whose quadriceps are
  coloured by the recorded EMG envelope of the chosen set (`envelopes.json`, written by `tools/export_envelopes.py`; the
  movement itself is illustrative), one real filtered set, and the 12 pipeline steps marked as
  found / done / checked, plus improvements that are not done yet.
- **Model performance:** regression and classification views with the model comparison by number of features, the
  overfitting check, the comparison table, actual vs predicted, residuals, Ridge weights, the ZC/SSC ablation, the
  learning curve, the confusion matrix, the ROC curve and per-participant predictions.
- **Insights & conclusions:** EDA, feature and model findings, limitations and next steps.

## Key findings

- Imbalance is a stable property of the person (set 1 vs set 2: r ≈ 0.96).
- Amplitude features (RMS, MAV, VAR, iEMG, WL, envelope) are the label itself, so they are excluded from the model.
- The strongest amplitude-free feature is the left/right asymmetry in slope sign changes (`ssc_rate_si`, ρ = 0.61).
- A Ridge model on 3 features predicts SI on unseen participants with R² ≈ 0.57.
