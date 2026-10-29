"""AWTAD – Quadriceps EMG imbalance dashboard.

Every number on these pages comes from the notebook (notebook/AWTAD_EDA_and_Modeling.ipynb): either read from the files
it writes into this folder, recomputed here with exactly the notebook's method, or copied from its printed tables
(NOTEBOOK_* below). Checks that the notebook does not do are labelled "dashboard check".

Run from this folder:  streamlit run app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             precision_score, r2_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
SI_THRESHOLD = 0.15
AMPLITUDE = ['rms', 'mav', 'iemg', 'var', 'wl', 'env_peak', 'env_mean']
META = ['subject', 'trial', 'set_name', 'rep', 'si_rms', 'label', 'weak_side', 'class']
MODEL_FEATURES = ['ssc_rate_si', 'zc_rate_si', 'ssc_rate_diff']

# ---- AWTAD palette. UI tokens, then data colours (validated: lightness, chroma, CVD and contrast checks)
PLUM, PEACH, BLUSH, BG, WHITE = '#4B3A57', '#F4D5C4', '#F8E9E5', '#FCFAF8', '#FFFFFF'
MAUVE, INK, INK_2, BORDER = '#A995AD', '#302A33', '#817984', '#EAE3E8'
D_PLUM, D_CLAY, D_TEAL, D_GOLD = '#7A5596', '#D4703F', '#2E9A86', '#C99A1F'
CLASS_COLORS = {'balanced': D_TEAL, 'imbalance': D_PLUM}
LEG_COLORS = {'left': D_PLUM, 'right': D_CLAY}
MODEL_COLORS = {'Ridge': D_PLUM, 'LogReg': D_PLUM, 'RandomForest': D_CLAY, 'kNN': D_TEAL, 'SVR': D_GOLD, 'SVM': D_GOLD,
                'Dummy': MAUVE, 'Dummy (mean)': MAUVE}
DIVERGING = [[0, D_CLAY], [.5, '#F6F1F3'], [1, D_PLUM]]

# ---- tables printed by the notebook (sections 7.1, 7.2 and 7.4). All 64 amplitude-free features, LOSO.
NOTEBOOK_CLASSIFIERS = pd.DataFrame({
    'model': ['Dummy', 'LogReg', 'SVM', 'RandomForest'],
    'balanced accuracy': [0.475, 0.613, 0.551, 0.545], 'F1': [0.536, 0.679, 0.673, 0.594],
    'AUC': [0.475, 0.710, 0.601, 0.615]})
NOTEBOOK_REGRESSORS = pd.DataFrame({
    'model': ['Dummy (mean)', 'Ridge', 'RandomForest', 'Ridge (no ZC/SSC)', 'RandomForest (no ZC/SSC)'],
    'MAE': [0.136, 0.092, 0.094, 0.111, 0.111], 'R²': [-0.058, 0.491, 0.428, 0.276, 0.254]})

st.set_page_config(page_title='AWTAD · EMG imbalance', page_icon=':material/monitor_heart:', layout='wide')

st.markdown(f"""
<style>
  .block-container {{ padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1240px; }}
  h1, h2, h3, h4 {{ color: {INK}; letter-spacing: -0.015em; }}
  [data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
  [data-testid="stSidebarNav"] a[aria-current="page"] {{ background: {BLUSH}; }}
  [data-testid="stSidebarNav"] a[aria-current="page"] span {{ color: {PLUM}; font-weight: 600; }}

  .side-note {{ color: {INK_2}; font-size: .84rem; line-height: 1.5; }}

  .crumb {{ color: {INK_2}; font-size: .78rem; letter-spacing: .06em; text-transform: uppercase; font-weight: 600; }}
  .crumb b {{ color: {PLUM}; }}
  .page-title {{ font-size: 2.05rem; font-weight: 700; color: {INK}; margin: 4px 0 6px 0; letter-spacing: -0.02em; }}
  .page-sub {{ color: {INK_2}; font-size: 1.02rem; max-width: 820px; line-height: 1.55; margin-bottom: 12px; }}
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 26px; }}
  .chip {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 999px; padding: 4px 12px; font-size: .8rem;
          color: {INK_2}; }}
  .chip b {{ color: {INK}; font-weight: 600; }}

  .kpi {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 20px; height: 100%; }}
  .kpi .label {{ color: {INK_2}; font-size: .78rem; font-weight: 600; letter-spacing: .05em; text-transform: uppercase; }}
  .kpi .value {{ color: {INK}; font-size: clamp(1.35rem, 2.2vw, 2rem); font-weight: 700; line-height: 1.2; margin-top: 6px;
                font-variant-numeric: tabular-nums; }}
  .kpi .note {{ color: {INK_2}; font-size: .82rem; margin-top: 4px; }}
  .kpi.hero {{ background: {PLUM}; border-color: {PLUM}; }}
  .kpi.hero .label, .kpi.hero .note {{ color: #D9CCDD; }}
  .kpi.hero .value {{ color: {WHITE}; }}

  .section {{ margin: 34px 0 12px 0; }}
  .section .eyebrow {{ color: {D_PLUM}; font-size: .74rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }}
  .section h3 {{ margin: 2px 0 2px 0; padding: 0; font-size: 1.3rem; }}
  .section p {{ color: {INK_2}; margin: 2px 0 0 0; max-width: 860px; }}

  .read {{ border-left: 3px solid {PEACH}; padding: 2px 0 2px 12px; color: {INK_2}; font-size: .9rem; line-height: 1.5;
          margin: 4px 0 6px 0; }}
  .read b {{ color: {INK}; }}

  .card {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 20px; height: 100%; }}
  .card .num {{ color: {D_PLUM}; font-weight: 700; font-size: .78rem; letter-spacing: .08em; }}
  .card .title {{ color: {INK}; font-weight: 650; font-size: 1.02rem; margin: 4px 0 6px 0; }}
  .card .body {{ color: {INK_2}; font-size: .9rem; line-height: 1.5; }}
  .card .body b {{ color: {INK}; }}
  .card.blush {{ background: {BLUSH}; border-color: #F1DCD6; }}

  .step {{ display: grid; grid-template-columns: 42px 1fr; gap: 14px; padding: 16px 0; border-bottom: 1px solid {BORDER}; }}
  .step:last-child {{ border-bottom: none; }}
  .step .n {{ width: 34px; height: 34px; border-radius: 50%; background: {BLUSH}; color: {PLUM}; font-weight: 700;
             display: grid; place-items: center; font-size: .88rem; }}
  .step .t {{ color: {INK}; font-weight: 650; font-size: 1rem; }}
  .step .d {{ color: {INK_2}; font-size: .9rem; line-height: 1.5; margin-top: 2px; }}
  .step .r {{ color: {INK}; font-size: .88rem; margin-top: 6px; }}
  .tag {{ display: inline-block; border-radius: 999px; padding: 1px 10px; font-size: .72rem; font-weight: 700;
         letter-spacing: .04em; margin-left: 8px; vertical-align: 2px; }}
  .tag.found {{ background: #FBEADF; color: #9A4A22; }}
  .tag.done {{ background: #E9F3EF; color: #1F6B5C; }}
  .tag.todo {{ background: #F1ECF3; color: {PLUM}; }}
  .tag.ok {{ background: #F3F1F2; color: {INK_2}; }}

  .flow {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }}
  .flow-step {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 12px; padding: 10px 14px; font-weight: 600;
               color: {INK}; font-size: .88rem; text-align: center; }}
  .flow-step small {{ font-weight: 400; color: {INK_2}; }}
  .flow-arrow {{ color: {MAUVE}; }}
  code {{ color: {PLUM} !important; background: {BLUSH} !important; }}
</style>
""", unsafe_allow_html=True)


# ================================================================ helpers
def style(fig, height=360, legend=True):
    fig.update_layout(
        template='plotly_white', height=height, font=dict(family='Inter, system-ui, sans-serif', size=12.5, color=INK_2),
        title=dict(font=dict(size=14.5, color=INK, family='Inter, system-ui, sans-serif'), x=0, xanchor='left'),
        paper_bgcolor=WHITE, plot_bgcolor=WHITE, showlegend=legend, bargap=.28,
        margin=dict(l=8, r=8, t=52 if fig.layout.title.text else 14, b=8),
        legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1, title_text='', font=dict(color=INK_2)),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER, font=dict(color=INK, size=12, family='Inter, sans-serif')))
    fig.update_xaxes(gridcolor='#F1ECEF', linecolor=BORDER, zeroline=False, title_font=dict(size=12, color=INK_2),
                     tickfont=dict(color=INK_2))
    fig.update_yaxes(gridcolor='#F1ECEF', linecolor=BORDER, zeroline=False, title_font=dict(size=12, color=INK_2),
                     tickfont=dict(color=INK_2))
    return fig


def chart(fig, note=None, **kw):
    """A chart in a white card, with a one-line reading underneath."""
    with st.container(border=True):
        st.plotly_chart(style(fig, **kw), use_container_width=True, config={'displaylogo': False})
        if note:
            st.markdown(f'<div class="read">{note}</div>', unsafe_allow_html=True)


def kpi(col, label, value, note='', hero=False):
    col.markdown(f'<div class="kpi{" hero" if hero else ""}"><div class="label">{label}</div>'
                 f'<div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)


def card(col, num, title, body, blush=False):
    col.markdown(f'<div class="card{" blush" if blush else ""}"><div class="num">{num}</div><div class="title">{title}</div>'
                 f'<div class="body">{body}</div></div>', unsafe_allow_html=True)


def page_header(title, sub, chips):
    chip_html = ''.join(f'<span class="chip">{k} <b>{v}</b></span>' for k, v in chips)
    st.markdown(f'<div class="crumb">AWTAD · Unit 3 &nbsp;/&nbsp; <b>{title}</b></div>'
                f'<div class="page-title">{title}</div><div class="page-sub">{sub}</div>'
                f'<div class="chips">{chip_html}</div>', unsafe_allow_html=True)


def section(eyebrow, title, sub=''):
    st.markdown(f'<div class="section"><div class="eyebrow">{eyebrow}</div><h3>{title}</h3>'
                f'{f"<p>{sub}</p>" if sub else ""}</div>', unsafe_allow_html=True)


def dots(fig, size=8):
    fig.update_traces(marker=dict(size=size, line=dict(color=WHITE, width=1.5)))
    return fig


# ================================================================ data and model (same method as the notebook)
@st.cache_data
def load_data():
    df = pd.read_csv(HERE / 'clean_data.csv')
    df['class'] = df.label.map({0: 'balanced', 1: 'imbalance'})
    return df


def is_amplitude(c):
    return any(c.startswith(a + '_') for a in AMPLITUDE)


@st.cache_data
def feature_ranking(df):
    """Notebook 6.1 and 6.3: Spearman with SI for every feature, AUC = max(auc, 1 - auc) for the class."""
    feats = [c for c in df.columns if c not in META]
    rows = []
    for c in feats:
        auc = roc_auc_score(df.label, df[c])
        rows.append({'feature': c, 'spearman': df[c].corr(df.si_rms, method='spearman'), 'AUC': max(auc, 1 - auc),
                     'group': 'amplitude (leaks the label)' if is_amplitude(c) else 'amplitude-free'})
    return pd.DataFrame(rows)


def between_subject_share(df, col):
    """Notebook 6.6."""
    g = df.groupby('subject')[col]
    between, within = g.mean().var(), g.var().mean()
    return between / (between + within)


@st.cache_resource
def final_model(df):
    """Notebook 7.6: Ridge(alpha=10) on the 3 features SelectKBest picks in all 33 folds, LOSO."""
    X, y, groups, cv = df[MODEL_FEATURES], df.si_rms, df.subject, LeaveOneGroupOut()
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    pred = cross_val_predict(model, X, y, groups=groups, cv=cv)
    dummy = cross_val_predict(DummyRegressor(), X, y, groups=groups, cv=cv)
    fitted = model.fit(X, y)
    per_subj = pd.DataFrame({'subject': groups, 'true': y, 'pred': pred}).groupby('subject').mean()
    yhat = (pred >= SI_THRESHOLD).astype(int)
    m = {'mae': mean_absolute_error(y, pred), 'mae_dummy': mean_absolute_error(y, dummy),
         'rmse': np.sqrt(np.mean((y - pred) ** 2)), 'r2': r2_score(y, pred), 'r2_dummy': r2_score(y, dummy),
         'r2_subj': r2_score(per_subj.true, per_subj.pred), 'r2_train': r2_score(y, fitted.predict(X)),
         'bal_acc': balanced_accuracy_score(df.label, yhat), 'auc': roc_auc_score(df.label, pred),
         'precision': precision_score(df.label, yhat), 'recall': recall_score(df.label, yhat),
         'f1': f1_score(df.label, yhat)}
    return fitted, pred, m


df_all = load_data()
ranking = feature_ranking(df_all)
model, loso_pred, metrics = final_model(df_all)
subjects = sorted(df_all.subject.unique())
NON_AMP = ranking[ranking.group == 'amplitude-free'].feature.tolist()
k_results = pd.read_csv(HERE / 'results_feature_count.csv')
k_results['features'] = k_results.k.astype(str).replace({'all': '64'})
learning = pd.read_csv(HERE / 'results_learning_curve.csv')

# ================================================================ sidebar: identity, description, filters
SI_MAX = float(np.ceil(df_all.si_rms.max() * 100) / 100)
DEFAULTS = {'f_subjects': [], 'f_set': 'Both sets', 'f_si': (0.0, SI_MAX), 'f_class': ['balanced', 'imbalance']}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


def reset_filters():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


st.logo(str(HERE / 'assets' / 'awtad_logo.png'), size='large')
with st.sidebar:
    st.markdown('<div class="side-note">Quadriceps sEMG · 33 participants · 314 squat reps</div>',
                unsafe_allow_html=True)
    st.divider()
    st.markdown('**Filters**', help='Apply to the EDA page. Model results always use all 33 participants.')
    st.multiselect('Participants', subjects, key='f_subjects', placeholder='All participants')
    st.selectbox('Set', ['Both sets', 'Set 1', 'Set 2'], key='f_set')
    st.slider('Symmetry Index range', 0.0, SI_MAX, step=0.01, key='f_si')
    st.multiselect('Class', ['balanced', 'imbalance'], key='f_class')
    st.button('Reset filters', icon=':material/restart_alt:', on_click=reset_filters, use_container_width=True)

chosen = st.session_state.f_subjects or subjects
df = df_all[df_all.subject.isin(chosen) & df_all.si_rms.between(*st.session_state.f_si)
            & df_all['class'].isin(st.session_state.f_class)]
if st.session_state.f_set != 'Both sets':
    df = df[df.trial == int(st.session_state.f_set[-1])]
FILTERED = len(df) < len(df_all)


def filter_chip():
    return ('Filter', f'{len(df)} of {len(df_all)} reps') if FILTERED else ('Filter', 'none · all reps')


# ================================================================ pages
def page_overview():
    page_header('Quadriceps imbalance from sEMG',
                'Which features of the muscle signal show that one leg works harder than the other during the squat? '
                'This dashboard presents the exploration, the cleaning, the models and the conclusions of the project '
                'notebook.',
                [('Dataset', 'AWTAD squat sEMG'), ('Sensors', 'Delsys Trigno · 2148 Hz'),
                 ('Target', 'SI (regression) · SI ≥ 0.15 (class)'), ('Evaluation', 'Leave-One-Subject-Out')])

    c = st.columns(4)
    kpi(c[0], 'Recordings', '314 reps', '33 participants · 63 sets')
    kpi(c[1], 'Model features', '64', 'amplitude-free (28 amplitude features excluded)')
    kpi(c[2], 'Imbalanced reps', f'{df_all.label.mean():.0%}', '185 of 314 have SI ≥ 0.15')
    kpi(c[3], 'Final model · R²', f'{metrics["r2"]:.2f}', f'Ridge, 3 features, new participants', hero=True)

    section('The target', 'How the two legs differ',
            'The Symmetry Index compares the RMS of the two legs: 0 = identical, 1 = one leg silent.')
    left, right = st.columns([1.2, 1])
    with left:
        fig = px.histogram(df_all, x='si_rms', color='class', nbins=30, color_discrete_map=CLASS_COLORS,
                           category_orders={'class': ['balanced', 'imbalance']},
                           labels={'si_rms': 'Symmetry Index (SI)', 'count': 'reps'}, title='SI per rep')
        fig.add_vline(SI_THRESHOLD, line_dash='dash', line_color=INK_2, annotation_text='threshold 0.15',
                      annotation_font_color=INK_2)
        fig.update_traces(marker_line_color=WHITE, marker_line_width=1.5)
        fig.update_yaxes(title_text='reps')
        chart(fig, '<b>59%</b> of reps are imbalanced and SI reaches 0.66. The classes are unequal but not severely.')
    with right:
        per_set = df_all.groupby(['subject', 'trial']).si_rms.mean().unstack().dropna().reset_index()
        r = np.corrcoef(per_set[1], per_set[2])[0, 1]
        fig = px.scatter(per_set, x=1, y=2, hover_name='subject', title=f'Same person, two sets (r = {r:.2f})',
                         labels={'1': 'mean SI · set 1', '2': 'mean SI · set 2'})
        dots(fig, 11).update_traces(marker_color=D_PLUM)
        fig.add_shape(type='line', x0=0, y0=0, x1=.7, y1=.7, line=dict(dash='dot', color=MAUVE))
        chart(fig, 'Imbalance is a <b>property of the person</b>, so the model is always tested on unseen participants.',
              legend=False)

    section('What predicts it', 'The strongest amplitude-free features',
            'AUC of each feature on its own: 0.5 = no separation of the two classes, 1 = perfect.')
    left, right = st.columns([1.2, 1])
    with left:
        top = ranking[ranking.group == 'amplitude-free'].nlargest(8, 'AUC').iloc[::-1]
        top['in final model'] = np.where(top.feature.isin(MODEL_FEATURES), 'in the final model', 'other')
        fig = px.bar(top, x='AUC', y='feature', orientation='h', color='in final model', text='AUC',
                     color_discrete_map={'in the final model': D_PLUM, 'other': '#D9CCDD'},
                     labels={'feature': ''}, title='Top 8 features by AUC')
        fig.update_traces(texttemplate='%{x:.2f}', textposition='outside', cliponaxis=False, marker_cornerradius=4)
        fig.update_yaxes(categoryorder='array', categoryarray=top.feature.tolist())
        fig.update_xaxes(range=[0.5, 0.8])
        chart(fig, 'The three features the final model picks in <b>all 33 folds</b> are among the strongest in the EDA.')
    with right:
        d = df_all.assign(predicted=loso_pred)
        fig = px.scatter(d, x='si_rms', y='predicted', color='class', color_discrete_map=CLASS_COLORS,
                         hover_data=['subject', 'trial', 'rep'], title='Final model: true vs predicted SI',
                         labels={'si_rms': 'true SI', 'predicted': 'predicted SI'})
        dots(fig, 7).add_shape(type='line', x0=0, y0=0, x1=.7, y1=.7, line=dict(dash='dot', color=MAUVE))
        chart(fig, f'Each rep is predicted by a model that never saw that participant. MAE <b>{metrics["mae"]:.3f}</b>.')

    section('Key findings', 'What the notebook shows')
    findings = [
        ('Imbalance is stable', 'Set 1 and set 2 SI correlate at <b>r ≈ 0.96</b>, and 86% of SI variation is between '
         'participants. The label describes the person, not a random rep.'),
        ('Amplitude features are the label', 'RMS, MAV, VAR, iEMG, WL and the envelope correlate 0.8–1.0 with SI '
         'because SI is computed from RMS. Using them would be <b>leakage</b>, so they are excluded.'),
        ('Slope sign changes carry the signal', '<code>ssc_rate_si</code> is the strongest amplitude-free feature '
         '(ρ 0.61, AUC 0.75). The weaker leg has more slope sign changes in 72% of reps.'),
        ('Fewer features generalise better', 'Ridge with 3 features beats Ridge with 64 on new participants '
         '(R² <b>0.57 vs 0.49</b>), and its train–test gap falls from 0.31 to 0.06.'),
    ]
    cols = st.columns(4)
    for i, (col, (t, b)) in enumerate(zip(cols, findings), 1):
        card(col, f'0{i}', t, b)


def page_eda():
    page_header('Exploratory data analysis',
                'The cleaned feature table: one row per rep, with each feature computed for the left leg, the right '
                'leg, and their difference (_si = relative, _diff = left − right).',
                [('Rows', f'{len(df)} reps'), ('Columns', '99'), ('Participants', df.subject.nunique()), filter_chip()])
    if df.empty:
        st.warning('No reps match the filters. Widen them in the sidebar or press Reset filters.')
        return

    tabs = st.tabs(['Dataset', 'Distributions', 'Feature vs target', 'Correlations', 'Outliers'])

    # -------------------------------------------------- dataset
    with tabs[0]:
        c = st.columns(4)
        kpi(c[0], 'Shape', f'{len(df)} × 99', 'rows × columns')
        kpi(c[1], 'Numeric columns', '96', '93 float · 3 integer')
        kpi(c[2], 'Text columns', '3', 'subject · set_name · weak_side')
        kpi(c[3], 'Missing values', '0', 'no NaN, no inf, no duplicates')
        st.write('')
        groups = pd.DataFrame([
            ('Identifiers', 'subject, trial, set_name, rep', 4, 'text / integer'),
            ('Target', 'si_rms (SI), label (SI ≥ 0.15), weak_side', 3, 'float / integer / text'),
            ('Amplitude', 'rms, mav, iemg, var, wl, env_peak, env_mean', 28, 'float · leaks the label'),
            ('Pattern', 'zc_rate, ssc_rate, hjorth_mob, hjorth_comp', 16, 'float'),
            ('Frequency', 'mnf, mdf, pkf, spec_ent', 16, 'float'),
            ('Timing', 'onset_frac, ttp_frac, burst_frac', 12, 'float'),
            ('Envelope shape', 'env_skew, env_kurt, env_cv', 12, 'float'),
            ('Both legs', 'env_xcorr_peak, env_xcorr_lag_ms, env_corr0, coherence_20_100', 4, 'float'),
            ('Fatigue and duration', 'mdf_slope_L/R/diff, duration_s', 4, 'float'),
        ], columns=['group', 'features (each × L, R, si, diff)', 'columns', 'type'])
        left, right = st.columns([1, 1.25])
        with left:
            with st.container(border=True):
                st.markdown('**Column groups**')
                st.dataframe(groups, hide_index=True, use_container_width=True, height=360)
        with right:
            fig = px.bar(groups.iloc[::-1], x='columns', y='group', orientation='h', text='columns',
                         title='Columns per group', labels={'group': ''})
            fig.update_traces(marker_color=[D_CLAY if g == 'Amplitude' else '#C9B8CE' for g in groups.group[::-1]],
                              marker_cornerradius=4, textposition='outside', cliponaxis=False)
            chart(fig, 'The 28 amplitude columns (clay) are dropped before modelling: 64 features remain.',
                  legend=False, height=380)
        with st.container(border=True):
            st.markdown('**Data preview**')
            st.dataframe(df.drop(columns='class').head(200), use_container_width=True, height=300)
        with st.container(border=True):
            st.markdown('**Summary statistics**')
            cols = st.multiselect('Columns', [c for c in df_all.select_dtypes('number').columns],
                                  default=['si_rms', 'duration_s', 'ssc_rate_L', 'ssc_rate_R', 'mnf_L', 'mnf_R',
                                           'env_cv_L', 'env_cv_R'])
            if cols:
                st.dataframe(df[cols].describe().round(3), use_container_width=True)

    # -------------------------------------------------- distributions
    with tabs[1]:
        left, right = st.columns([1, 1])
        with left:
            counts = df['class'].value_counts().reindex(['balanced', 'imbalance']).fillna(0).reset_index()
            fig = px.bar(counts, x='class', y='count', color='class', color_discrete_map=CLASS_COLORS, text='count',
                         title='Reps per class', labels={'class': '', 'count': 'reps'})
            fig.update_traces(textposition='outside', marker_cornerradius=4, cliponaxis=False)
            chart(fig, 'Because the classes are unequal, the models are scored with <b>balanced accuracy</b>.',
                  legend=False)
        with right:
            weak = df[df.label == 1].weak_side.map({'L': 'left', 'R': 'right'}).value_counts().reindex(['left', 'right'])
            weak = weak.dropna()
            if weak.sum():
                fig = go.Figure(go.Pie(labels=[f'{s} leg' for s in weak.index], values=weak.values, hole=.64, sort=False,
                                       marker=dict(colors=[LEG_COLORS[s] for s in weak.index],
                                                   line=dict(color=WHITE, width=3)),
                                       textinfo='label+percent', insidetextorientation='horizontal',
                                       hovertemplate='%{label}: %{value} reps<extra></extra>'))
                fig.update_layout(title='Weaker leg in imbalanced reps',
                                  annotations=[dict(text=f'<b>{int(weak.sum())}</b><br>imbalanced reps',
                                                    showarrow=False, font=dict(size=14, color=INK_2))])
                chart(fig, 'The <b>left leg</b> is the weaker one in most imbalanced reps. Each participant\'s dominant '
                      'leg was not recorded, so this cannot be explained yet.', legend=False)
            else:
                st.info('No imbalanced reps in the current filter.')

        per_subj = df.groupby('subject', as_index=False).si_rms.mean().sort_values('si_rms')
        per_subj['class'] = np.where(per_subj.si_rms >= SI_THRESHOLD, 'imbalance', 'balanced')
        fig = px.bar(per_subj, x='subject', y='si_rms', color='class', color_discrete_map=CLASS_COLORS,
                     labels={'si_rms': 'mean SI', 'subject': ''}, title='Mean SI per participant')
        fig.add_hline(SI_THRESHOLD, line_dash='dash', line_color=INK_2)
        fig.update_traces(marker_cornerradius=4, hovertemplate='%{x}: %{y:.3f}<extra></extra>')
        fig.update_xaxes(tickangle=-45)
        chart(fig, '20 of the 33 participants have a mean SI above 0.15. A few (Wintana, Afaf, Nora) exceed 0.5.',
              height=380)

        left, right = st.columns([1, 1])
        with left:
            feature = st.selectbox('Feature to plot', NON_AMP, index=NON_AMP.index('ssc_rate_si'), key='dist_feature')
            fig = px.histogram(df, x=feature, color='class', nbins=30, color_discrete_map=CLASS_COLORS, barmode='overlay',
                               opacity=.75, category_orders={'class': ['balanced', 'imbalance']},
                               title=f'Distribution of {feature}')
            fig.update_traces(marker_line_color=WHITE, marker_line_width=1)
            fig.update_yaxes(title_text='reps')
            chart(fig, 'Overlapping histograms: where the two colours separate, the feature can tell the classes apart.')
        with right:
            st.write('')
            fat = df.groupby('rep')[['mdf_L', 'mdf_R']].mean().reset_index().melt('rep', var_name='leg', value_name='mdf')
            fat['leg'] = fat.leg.map({'mdf_L': 'left', 'mdf_R': 'right'})
            fig = px.line(fat, x='rep', y='mdf', color='leg', markers=True, color_discrete_map=LEG_COLORS,
                          title='Fatigue: median frequency per rep', labels={'mdf': 'median frequency (Hz)'})
            fig.update_traces(line_width=2, marker_size=9)
            fig.update_xaxes(dtick=1)
            chart(fig, 'Median frequency drops ~3 Hz from rep 1 to rep 5 in both legs (fatigue), but the difference in '
                  'fatigue slope is unrelated to SI (ρ = −0.02).')

    # -------------------------------------------------- feature vs target
    with tabs[2]:
        c1, c2 = st.columns([1, 2.4])
        with_amp = c1.toggle('Include amplitude features', value=False)
        top_n = c2.slider('Features to show', 5, 30, 15)
        view = ranking if with_amp else ranking[ranking.group == 'amplitude-free']
        top = view.reindex(view.spearman.abs().sort_values(ascending=False).index).head(top_n).iloc[::-1]
        fig = px.bar(top, x='spearman', y='feature', orientation='h', color='group',
                     color_discrete_map={'amplitude (leaks the label)': D_CLAY, 'amplitude-free': D_PLUM},
                     hover_data={'AUC': ':.3f', 'spearman': ':.3f'},
                     labels={'spearman': 'Spearman correlation with SI', 'feature': ''},
                     title='Features ranked by correlation with SI (all 314 reps)')
        fig.update_traces(marker_cornerradius=4)
        fig.update_yaxes(categoryorder='array', categoryarray=top.feature.tolist())
        chart(fig, 'Amplitude features correlate almost perfectly with SI because SI is built from them. Without them, '
              '<code>ssc_rate_si</code> leads (ρ 0.61).', height=24 * top_n + 110)

        feature = st.selectbox('Explore one feature', ranking[ranking.group == 'amplitude-free'].nlargest(20, 'AUC').feature,
                               key='rel_feature')
        auc = ranking.set_index('feature').loc[feature, 'AUC']
        left, right = st.columns(2)
        with left:
            fig = px.box(df, x='class', y=feature, color='class', color_discrete_map=CLASS_COLORS, points='all',
                         category_orders={'class': ['balanced', 'imbalance']}, title=f'{feature} by class (AUC {auc:.2f})',
                         labels={'class': ''}, hover_data=['subject', 'rep'])
            fig.update_traces(marker=dict(size=5, opacity=.55), jitter=.35)
            chart(fig, 'Box = middle 50% of reps, line = median. The boxes overlap: one feature alone separates the '
                  'classes only partly.', legend=False, height=400)
        with right:
            fig = px.scatter(df, x=feature, y='si_rms', color='class', color_discrete_map=CLASS_COLORS,
                             hover_data=['subject', 'trial', 'rep'], labels={'si_rms': 'SI'}, title=f'{feature} vs SI')
            dots(fig, 7).add_hline(SI_THRESHOLD, line_dash='dash', line_color=INK_2)
            chart(fig, 'Each dot is a rep. An upward cloud means SI rises with the feature.', height=400)

        base = feature.rsplit('_', 1)[0]
        left, right = st.columns(2)
        with left:
            if f'{base}_L' in df.columns:
                fig = px.scatter(df, x=f'{base}_L', y=f'{base}_R', color='class', color_discrete_map=CLASS_COLORS,
                                 hover_data=['subject', 'rep'], title=f'{base}: left leg vs right leg',
                                 labels={f'{base}_L': 'left leg', f'{base}_R': 'right leg'})
                dots(fig, 7)
                lo = min(df[f'{base}_L'].min(), df[f'{base}_R'].min())
                hi = max(df[f'{base}_L'].max(), df[f'{base}_R'].max())
                fig.add_shape(type='line', x0=lo, y0=lo, x1=hi, y1=hi, line=dict(dash='dot', color=MAUVE))
                chart(fig, 'On the dotted line both legs are equal; distance from it is a left/right difference.',
                      height=400)
        with right:
            cols = ['si_rms'] + ranking[ranking.group == 'amplitude-free'].nlargest(8, 'AUC').feature.tolist() + \
                   ['ttp_frac_diff', 'env_corr0']
            share = pd.Series({c: between_subject_share(df_all, c) for c in cols}).sort_values().reset_index()
            share.columns = ['feature', 'share']
            fig = px.bar(share, x='share', y='feature', orientation='h', title='Is the feature a property of the person?',
                         labels={'share': 'share of variation between participants (all reps)', 'feature': ''})
            fig.update_traces(marker_color=[D_PLUM if f == 'si_rms' else '#C9B8CE' for f in share.feature],
                              marker_cornerradius=4, hovertemplate='%{y}: %{x:.0%}<extra></extra>')
            fig.add_vline(.5, line_dash='dot', line_color=MAUVE)
            fig.update_xaxes(range=[0, 1], tickformat='.0%')
            chart(fig, 'SI (plum) and the strongest features vary mostly <b>between</b> people; timing and coordination '
                  'vary from rep to rep.', legend=False, height=400)

    # -------------------------------------------------- correlations
    with tabs[3]:
        default = ['si_rms'] + ranking[ranking.group == 'amplitude-free'].nlargest(11, 'AUC').feature.tolist()
        chosen_cols = st.multiselect('Features in the heatmap', ['si_rms'] + ranking.feature.tolist(), default=default)
        if len(chosen_cols) < 2:
            st.info('Pick at least two features.')
        else:
            corr = df[chosen_cols].corr(method='spearman')
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale=DIVERGING, zmin=-1, zmax=1, aspect='auto',
                            title='Spearman correlation')
            fig.update_traces(hovertemplate='%{x} · %{y}: %{z:.2f}<extra></extra>')
            chart(fig, 'Plum = move together, clay = move in opposite directions, pale = unrelated.',
                  height=150 + 40 * len(chosen_cols), legend=False)
            pairs = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack().reset_index()
            pairs.columns = ['feature A', 'feature B', 'Spearman r']
            pairs = pairs[pairs['Spearman r'].abs() > .9].sort_values('Spearman r', key=abs, ascending=False)
            with st.container(border=True):
                st.markdown('**Redundant pairs (|r| > 0.9)** – the two features carry almost the same information')
                if pairs.empty:
                    st.caption('None among the selected features. Add mnf_L and hjorth_mob_L to see one.')
                else:
                    st.dataframe(pairs.round(3), use_container_width=True, hide_index=True)

    # -------------------------------------------------- outliers
    with tabs[4]:
        st.markdown('<div class="read">The notebook <b>keeps</b> outliers: with 33 people an extreme value is usually a '
                    'real participant (for example SI > 0.5), not a recording error. The counts below are a '
                    '<b>dashboard check</b> using the 1.5 × IQR rule; nothing was removed.</div>', unsafe_allow_html=True)
        cols = ['si_rms'] + MODEL_FEATURES + ['env_cv_R', 'env_cv_L', 'zc_rate_diff', 'mnf_diff', 'duration_s']
        rows = []
        for c in cols:
            q1, q3 = df[c].quantile([.25, .75])
            out = df[(df[c] < q1 - 1.5 * (q3 - q1)) | (df[c] > q3 + 1.5 * (q3 - q1))]
            rows.append({'feature': c, 'outlier reps': len(out), 'share': len(out) / len(df),
                         'participants': ', '.join(out.subject.value_counts().index[:4])})
        out_tab = pd.DataFrame(rows)
        left, right = st.columns([1.3, 1])
        with left:
            feature = st.selectbox('Feature', cols, key='out_feature')
            fig = px.box(df, x='class', y=feature, color='class', color_discrete_map=CLASS_COLORS, points='outliers',
                         category_orders={'class': ['balanced', 'imbalance']}, hover_data=['subject', 'trial', 'rep'],
                         title=f'{feature}: outliers per class', labels={'class': ''})
            chart(fig, 'Dots outside the whiskers are beyond 1.5 × IQR. Hover to see which participant they belong to.',
                  legend=False, height=420)
        with right:
            st.write('')
            with st.container(border=True):
                st.markdown('**Reps beyond 1.5 × IQR**')
                st.dataframe(out_tab, hide_index=True, use_container_width=True, height=420,
                             column_config={'share': st.column_config.ProgressColumn('share', format='percent',
                                                                                     min_value=0, max_value=.2)})


def page_quality():
    page_header('Data quality & preprocessing',
                'How the raw recordings became the cleaned table: what was checked, what was changed, and what is left '
                'to improve.',
                [('Raw files', '63 Trigno CSVs'), ('Result', 'clean_data.csv · 314 × 99'),
                 ('Signal chain', 'DC · 20–450 Hz · 60 Hz notch')])

    section('Recording', 'How the data was collected',
            'The barbell back squat, with one wireless EMG sensor on the front of each thigh (quadriceps). Press play '
            'to see each quadriceps light up with the recorded EMG of the chosen set.')
    exercise_viewer()

    sig = pd.read_csv(HERE / 'signal_example.csv')
    marks = pd.read_csv(HERE / 'signal_example_markers.csv').marker_s.tolist()
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=.07)
    for row, side, color, light in [(1, 'L', D_PLUM, '#DCCFE3'), (2, 'R', D_CLAY, '#F2D3C3')]:
        fig.add_trace(go.Scatter(x=sig.t, y=sig[f'raw_{side}'], line=dict(color=light, width=1), name='filtered EMG',
                                 showlegend=row == 1, legendgroup='raw', hoverinfo='skip'), row=row, col=1)
        fig.add_trace(go.Scatter(x=sig.t, y=sig[f'env_{side}'], line=dict(color=color, width=2.4),
                                 name='envelope (left)' if side == 'L' else 'envelope (right)',
                                 hovertemplate='%{x:.1f} s · %{y:.0f} µV<extra></extra>'), row=row, col=1)
        for m in marks:
            fig.add_vline(m, line_dash='dash', line_color=MAUVE, line_width=1, row=row, col=1)
    for i in range(len(marks) - 1):
        fig.add_annotation(x=(marks[i] + marks[i + 1]) / 2, y=1.0, yref='paper', yanchor='bottom', showarrow=False,
                           text=f'rep {i + 1}', font=dict(size=12, color=INK_2))
    fig.update_yaxes(title_text='left leg (µV)', row=1, col=1)
    fig.update_yaxes(title_text='right leg (µV)', row=2, col=1)
    fig.update_xaxes(title_text='time (s)', row=2, col=1)
    fig.update_layout(title='One real set after filtering (Wintana, set 1)')
    style(fig, height=470)
    fig.update_layout(legend=dict(y=-0.14, yanchor='top', x=0, xanchor='left'), margin=dict(t=70, b=40))
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False})
        st.markdown('<div class="read">Dashed lines are the rep markers (6 markers = 5 reps). The right leg is more '
                    'active in every rep: this participant has a large imbalance (SI ≈ 0.6).</div>',
                    unsafe_allow_html=True)

    section('Pipeline', 'Step by step',
            '<span class="tag found">FOUND</span> an issue seen in the data · <span class="tag done">DONE</span> a change '
            'applied in the notebook · <span class="tag ok">CHECKED</span> examined, nothing to change')
    steps = [
        ('Loading the Trigno files', 'found',
         'Each file has 7 metadata lines and the marker table in extra columns, so a small parser reads the left EMG '
         '(column 2), the right EMG (column 4) and the marker times.',
         '33 participants · 63 sets · 314 reps. Abeer, Lulu and SaraA have only one set; Khawla\'s first set has 4 reps. '
         'All are <b>kept</b>.'),
        ('Frequency check (PSD)', 'found',
         'The power spectrum of every recording is used to choose the filters from the data.',
         'Power above 450 Hz: max 0.001 (nothing to remove). Below 20 Hz: median 0.009, max 0.177 (movement artefact '
         'in a few files). A small 60 Hz mains peak in a few files.'),
        ('Signal filtering', 'done',
         'DC offset removed, 4th-order Butterworth band-pass 20–450 Hz, 60 Hz notch (Q = 30). filtfilt runs forward and '
         'backward so there is no time shift between the legs.',
         'Applied to all 126 channels (63 sets × 2 legs).'),
        ('Envelope and rep segmentation', 'done',
         'Linear envelope: absolute value, then a 6 Hz low-pass. Each set is cut into reps at the markers.',
         '6 markers → 5 reps per set.'),
        ('Feature engineering', 'done',
         '21 features per leg (amplitude, pattern, frequency, timing, envelope shape), plus _si and _diff for every pair, '
         '4 two-leg coordination features, MDF fatigue slopes and rep duration.',
         '92 features + 7 identifier/target columns = <b>99 columns</b>.'),
        ('Missing, infinite and duplicate values', 'ok', 'isnull(), isinf() and duplicated() on the feature table.',
         '<b>0</b> missing · <b>0</b> infinite · <b>0</b> duplicates. Nothing to fill or drop.'),
        ('Types and names', 'done',
         'trial, rep and label cast to integers; subject and weak_side stay text. Lower-case names with _L / _R / _si / '
         '_diff suffixes.', 'Saved as clean_data.csv (314 × 99).'),
        ('Outliers', 'ok',
         'Extreme values are kept: with 33 people they usually belong to real participants (e.g. SI > 0.5), not to errors.',
         'No rows removed. The EDA page has a 1.5 × IQR count as a dashboard check.'),
        ('Leakage removal', 'done',
         'SI is computed from RMS, so the 28 amplitude features (RMS, MAV, iEMG, VAR, WL, envelope peak and mean) copy '
         'the label.', '28 dropped → <b>64</b> amplitude-free model features.'),
        ('Scaling', 'done', 'StandardScaler inside each model pipeline (Ridge, LogReg, SVM/SVR, kNN). Trees use raw values.',
         'Fitted on the training participants of each fold only.'),
        ('Feature selection', 'done',
         'SelectKBest inside the pipeline with k = 3, 5, 10 or all, so it never sees the test participant.',
         'With k = 3 the same features are picked in <b>33 of 33</b> folds: ssc_rate_si, zc_rate_si, ssc_rate_diff.'),
        ('Train / test split', 'done',
         'Leave-One-Subject-Out: each fold holds out one whole participant, 33 times. A participant\'s reps look alike, '
         'so a random split would let the model recognise the person.', '33 folds · every rep is predicted once.'),
    ]
    labels = {'found': 'FOUND', 'done': 'DONE', 'ok': 'CHECKED'}
    html = ''.join(f'<div class="step"><div class="n">{i}</div><div><div class="t">{t}<span class="tag {s}">'
                   f'{labels[s]}</span></div><div class="d">{d}</div><div class="r">{r}</div></div></div>'
                   for i, (t, s, d, r) in enumerate(steps, 1))
    with st.container(border=True):
        st.markdown(html, unsafe_allow_html=True)

    section('Not done yet', 'Possible improvements', 'Suggestions only – none of these is in the notebook.')
    todo = [('Normalise amplitude', 'Record a maximal voluntary contraction (MVC) per leg, so SI does not depend on '
             'electrode placement, skin or fat.'),
            ('Flag low-quality recordings', 'Use the below-20 Hz power from the PSD check to flag files with movement '
             'artefact automatically.'),
            ('Record the dominant leg', 'Needed to explain why the left leg is usually the weaker one.')]
    cols = st.columns(3)
    for col, (t, b) in zip(cols, todo):
        card(col, 'SUGGESTION', t, b, blush=True)


def page_model():
    page_header('Model performance',
                'Two tasks on the same 314 reps: <b>regression</b> predicts the SI value, <b>classification</b> predicts '
                'imbalance (SI ≥ 0.15). Every score is on participants the model never saw (Leave-One-Subject-Out).',
                [('Models', 'Ridge · LogReg · RF · SVM/SVR · kNN · Dummy'), ('Features', '3 / 5 / 10 / 64'),
                 ('Validation', 'LOSO · 33 folds')])

    c = st.columns(4)
    kpi(c[0], 'R² · new participants', f'{metrics["r2"]:.3f}', f'Ridge, 3 features · Dummy {metrics["r2_dummy"]:.2f}',
        hero=True)
    kpi(c[1], 'Mean absolute error', f'{metrics["mae"]:.3f}', f'SI units · Dummy {metrics["mae_dummy"]:.3f}')
    kpi(c[2], 'R² · per participant', f'{metrics["r2_subj"]:.3f}', 'mean SI of each person')
    kpi(c[3], 'R² · training data', f'{metrics["r2_train"]:.3f}', f'gap to test {metrics["r2_train"] - metrics["r2"]:.2f}')
    st.markdown('<div class="read" style="margin-top:14px">Lower MAE and RMSE = smaller errors. R² is the share of SI '
                'variation the model explains compared with always predicting the mean (0 = no better, 1 = perfect). '
                'It is <b>not</b> an accuracy percentage.</div>', unsafe_allow_html=True)

    task = st.segmented_control('Task', ['Regression', 'Classification'], default='Regression', key='task')
    task = task or 'Regression'
    task_key = 'regression (R2)' if task == 'Regression' else 'classification (bal. acc)'
    metric_name = 'R²' if task == 'Regression' else 'balanced accuracy'
    k = k_results[k_results.task == task_key]
    models = sorted(k.model.unique())

    section('Comparison', f'{task}: models and number of features',
            'From notebook 7.5. SelectKBest picks the features inside each fold.')
    shown = st.multiselect('Models', models, default=models, key=f'models_{task}')
    k = k[k.model.isin(shown)]
    if k.empty:
        st.info('Pick at least one model.')
    else:
        left, right = st.columns([1.15, 1])
        with left:
            fig = px.line(k, x='features', y='test', color='model', markers=True, color_discrete_map=MODEL_COLORS,
                          category_orders={'features': ['3', '5', '10', '64']}, title=f'{metric_name} on new participants',
                          labels={'test': metric_name, 'features': 'number of features'})
            fig.update_traces(line_width=2, marker=dict(size=9, line=dict(color=WHITE, width=1.5)))
            for m in shown[:4]:   # direct label above the first point (index 0 = '3' on the category axis)
                first = k[(k.model == m) & (k.features == '3')]
                if len(first):
                    fig.add_annotation(x=0, y=first.test.iloc[0], text=m, showarrow=False, xanchor='left', xshift=6,
                                       yshift=10, font=dict(size=11, color=INK_2))
            chart(fig, 'Every model scores better with 3–5 features than with all 64.')
        with right:
            g = k[k.features.isin(['3', '64'])].copy()
            g['setup'] = g.model + ' · ' + g.features
            g = g.sort_values(['model', 'features'], ascending=[False, False])
            fig = go.Figure()
            for _, r in g.iterrows():
                fig.add_trace(go.Scatter(x=[r.test, r.train], y=[r.setup] * 2, mode='lines', showlegend=False,
                                         line=dict(color=BORDER, width=6), hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=g.test, y=g.setup, mode='markers', name='new participants',
                                     marker=dict(size=12, color=D_PLUM, line=dict(color=WHITE, width=2)),
                                     hovertemplate='%{y} features<br>test %{x:.3f}<extra></extra>'))
            fig.add_trace(go.Scatter(x=g.train, y=g.setup, mode='markers', name='training data',
                                     marker=dict(size=12, color=D_GOLD, line=dict(color=WHITE, width=2)),
                                     hovertemplate='%{y} features<br>train %{x:.3f}<extra></extra>'))
            fig.update_layout(title='Overfitting check: train vs test', xaxis_title=metric_name)
            fig.update_yaxes(title_text='model · features')
            chart(fig, 'A long bar means the model memorises its training participants. Only the linear models stay '
                  'short with 3 features.')

        table = k.pivot_table(index='model', columns='features', values='test')[['3', '5', '10', '64']]
        table.columns = [f'{c} features' for c in table.columns]
        gaps = k[k.features == '3'].set_index('model').gap
        table['gap (3 features)'] = gaps
        best = table['3 features'].idxmax()
        with st.container(border=True):
            st.markdown(f'**{metric_name} on new participants** · highlighted: best with 3 features')
            st.dataframe(table.round(3).style.format('{:.3f}').apply(
                lambda row: [f'background-color: {BLUSH}; color: {PLUM}; font-weight: 600' if row.name == best else ''
                             for _ in row], axis=1), use_container_width=True)
        if task == 'Regression':
            st.markdown('<div class="read"><b>Best: Ridge with 3 features</b> (R² 0.568) – the highest test score and '
                        'the smallest train–test gap (0.06).</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="read">With 3 features LogReg (0.655) and Random Forest (0.656) tie, but Random '
                        'Forest\'s gap is 0.18 against 0.07. <b>LogReg</b> is the more defensible choice. When linear, '
                        'tree and neighbour models all stop near 0.65, the limit is the information in the features, '
                        'not the model.</div>', unsafe_allow_html=True)

    if task == 'Regression':
        section('Diagnostics', 'Final model: Ridge on 3 features',
                f'RMSE {metrics["rmse"]:.3f} · residuals = true − predicted, from the LOSO predictions.')
        d = df_all.assign(predicted=loso_pred, residual=df_all.si_rms - loso_pred)
        view = st.segmented_control('Level', ['per rep', 'per participant'], default='per rep', key='level') or 'per rep'
        if view == 'per participant':
            d = d.groupby('subject', as_index=False)[['si_rms', 'predicted', 'residual']].mean()
            d['class'] = np.where(d.si_rms >= SI_THRESHOLD, 'imbalance', 'balanced')
        left, right = st.columns(2)
        with left:
            fig = px.scatter(d, x='si_rms', y='predicted', color='class', color_discrete_map=CLASS_COLORS,
                             hover_data=['subject'], title=f'Actual vs predicted SI ({view})',
                             labels={'si_rms': 'true SI', 'predicted': 'predicted SI'})
            dots(fig, 8 if view == 'per rep' else 12)
            fig.add_shape(type='line', x0=0, y0=0, x1=.7, y1=.7, line=dict(dash='dot', color=MAUVE))
            fig.add_hline(SI_THRESHOLD, line_width=1, line_color='#D9CCDD')
            fig.add_vline(SI_THRESHOLD, line_width=1, line_color='#D9CCDD')
            chart(fig, 'Perfect predictions sit on the dotted line. High SI is under-predicted: the model is '
                  'cautious at the extremes.', height=400)
        with right:
            fig = px.scatter(d, x='predicted', y='residual', color='class', color_discrete_map=CLASS_COLORS,
                             hover_data=['subject'], title='Residuals vs predicted SI',
                             labels={'predicted': 'predicted SI', 'residual': 'residual (true − predicted)'})
            dots(fig, 8 if view == 'per rep' else 12).add_hline(0, line_dash='dot', line_color=MAUVE)
            chart(fig, 'Residuals grow with SI: large imbalances are underestimated.', height=400)
        left, right = st.columns(2)
        with left:
            fig = px.histogram(d, x='residual', nbins=30, title='Residual distribution', labels={'residual': 'residual'})
            fig.update_traces(marker_color='#C9B8CE', marker_line_color=WHITE, marker_line_width=1.5)
            fig.add_vline(0, line_dash='dot', line_color=INK_2)
            fig.update_yaxes(title_text='count')
            chart(fig, f'Centred near zero (mean {d.residual.mean():+.3f}), with a longer tail of under-predictions.',
                  legend=False, height=360)
        with right:
            w = pd.Series(model[-1].coef_, index=MODEL_FEATURES).sort_values()
            fig = px.bar(x=w.values, y=w.index, orientation='h', title='Ridge weights (standardised features)',
                         labels={'x': 'weight', 'y': ''})
            fig.update_traces(marker_color=D_PLUM, marker_cornerradius=4, texttemplate='%{x:.4f}', textposition='outside',
                              cliponaxis=False)
            fig.update_xaxes(range=[0, w.max() * 1.3])
            chart(fig, 'All weights are positive: the bigger the left/right difference in slope sign changes and zero '
                  'crossings, the higher the predicted SI.', legend=False, height=360)

        left, right = st.columns(2)
        with left:
            nb = NOTEBOOK_REGRESSORS.copy()
            nb['kind'] = np.where(nb.model.str.contains('no ZC'), 'without ZC/SSC',
                                  np.where(nb.model.str.startswith('Dummy'), 'baseline', 'all 64 features'))
            fig = px.bar(nb, x='R²', y='model', orientation='h', color='kind', text='R²',
                         color_discrete_map={'baseline': MAUVE, 'all 64 features': D_PLUM, 'without ZC/SSC': D_CLAY},
                         title='Notebook 7.2 and 7.4: 64 features, with and without ZC/SSC', labels={'model': ''})
            fig.update_traces(texttemplate='%{x:.3f}', textposition='outside', cliponaxis=False, marker_cornerradius=4)
            fig.update_yaxes(categoryorder='array', categoryarray=nb.model.tolist()[::-1])
            chart(fig, 'Removing the ZC and SSC features drops R² from 0.49 to 0.28: they carry most of the signal.',
                  height=380)
        with right:
            fig = px.line(learning, x='participants', y='median_R2', markers=True, title='Would more participants help?',
                          labels={'participants': 'participants in training', 'median_R2': 'median R² on 4 new participants'})
            fig.update_traces(line=dict(color=D_PLUM, width=2), marker=dict(size=9, line=dict(color=WHITE, width=1.5)))
            fig.add_hline(0, line_dash='dot', line_color=MAUVE)
            chart(fig, 'The score levels off after ~18 participants. The test set here is only 4 people, so the shape '
                  'matters more than the level.', legend=False, height=380)

    else:
        section('Diagnostics', 'Final Ridge used as a classifier',
                'Predicted SI ≥ 0.15 → imbalance. Positive class = imbalance. From the same LOSO predictions.')
        c = st.columns(5)
        kpi(c[0], 'Balanced accuracy', f'{metrics["bal_acc"]:.3f}')
        kpi(c[1], 'AUC', f'{metrics["auc"]:.3f}')
        kpi(c[2], 'Precision', f'{metrics["precision"]:.3f}', 'dashboard calc.')
        kpi(c[3], 'Recall', f'{metrics["recall"]:.3f}', 'dashboard calc.')
        kpi(c[4], 'F1', f'{metrics["f1"]:.3f}', 'dashboard calc.')
        st.write('')
        left, right = st.columns(2)
        with left:
            cm = confusion_matrix(df_all.label, loso_pred >= SI_THRESHOLD)
            names = ['balanced', 'imbalance']
            fig = go.Figure(go.Heatmap(z=cm, x=names, y=names, text=cm, texttemplate='%{text}',
                                       textfont=dict(size=22, family='Inter'), showscale=False,
                                       colorscale=[[0, BLUSH], [1, D_PLUM]],
                                       hovertemplate='true %{y}<br>predicted %{x}<br>%{z} reps<extra></extra>'))
            fig.update_layout(title='Confusion matrix (314 reps)', xaxis_title='predicted', yaxis_title='true')
            fig.update_yaxes(autorange='reversed')
            chart(fig, 'Most errors are reps close to the 0.15 threshold that land on the wrong side.', legend=False,
                  height=400)
        with right:
            fpr, tpr, _ = roc_curve(df_all.label, loso_pred)
            fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(color=D_PLUM, width=2.4),
                                       fill='tozeroy', fillcolor='rgba(122,85,150,.08)', name='Ridge',
                                       hovertemplate='false positive rate %{x:.2f}<br>true positive rate %{y:.2f}'
                                                     '<extra></extra>'))
            fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dot', color=MAUVE))
            fig.update_layout(title=f'ROC curve (AUC {metrics["auc"]:.3f})', xaxis_title='false positive rate',
                              yaxis_title='true positive rate')
            chart(fig, 'The dotted diagonal is random guessing (AUC 0.5). The curve sweeps every threshold, not only '
                  '0.15.', legend=False, height=400)
        nb = NOTEBOOK_CLASSIFIERS.melt('model', var_name='metric', value_name='score')
        fig = px.bar(nb, x='model', y='score', color='metric', barmode='group', text='score',
                     color_discrete_map={'balanced accuracy': D_PLUM, 'F1': D_CLAY, 'AUC': D_TEAL},
                     title='Notebook 7.1: classifiers on all 64 features (threshold 0.5)', labels={'model': ''})
        fig.update_traces(texttemplate='%{y:.2f}', textposition='outside', cliponaxis=False, marker_cornerradius=4)
        fig.add_hline(.5, line_dash='dot', line_color=MAUVE)
        chart(fig, 'With all 64 features only LogReg clearly beats the Dummy. Reducing to 3 features lifts every model '
              '(see the comparison above).', height=380)

    section('Try it', 'Predictions for one participant',
            'The bars come from the LOSO run: the model that produced them was trained without this participant.')
    who = st.selectbox('Participant', subjects, index=subjects.index('Wintana') if 'Wintana' in subjects else 0)
    mask = (df_all.subject == who).values
    view = df_all.loc[mask, ['trial', 'rep', 'si_rms']].assign(predicted=loso_pred[mask])
    view['rep_id'] = 'set ' + view.trial.astype(str) + ' · rep ' + view.rep.astype(str)
    true_m, pred_m = view.si_rms.mean(), view.predicted.mean()
    correct = (pred_m >= SI_THRESHOLD) == (true_m >= SI_THRESHOLD)
    c = st.columns(3)
    kpi(c[0], 'Predicted mean SI', f'{pred_m:.3f}', 'imbalance' if pred_m >= SI_THRESHOLD else 'balanced', hero=True)
    kpi(c[1], 'True mean SI', f'{true_m:.3f}', 'imbalance' if true_m >= SI_THRESHOLD else 'balanced')
    kpi(c[2], 'Participant verdict', 'correct' if correct else 'wrong', 'at the 0.15 threshold')
    st.write('')
    fig = go.Figure([go.Bar(x=view.rep_id, y=view.si_rms, name='true SI', marker_color='#D9CCDD', marker_cornerradius=4),
                     go.Bar(x=view.rep_id, y=view.predicted, name='predicted SI', marker_color=D_PLUM,
                            marker_cornerradius=4)])
    fig.add_hline(SI_THRESHOLD, line_dash='dash', line_color=INK_2, annotation_text='0.15')
    fig.update_layout(barmode='group', title=f'{who}: true vs predicted SI per rep', yaxis_title='SI')
    chart(fig, height=360)


def page_insights():
    page_header('Insights & conclusions',
                'What the analysis supports, what limits it, and what to do next.',
                [('Based on', 'notebook sections 5–8'), ('Final model', 'Ridge · 3 features · R² 0.57')])

    groups = [
        ('EDA insights', [
            ('The data is clean', 'No missing, infinite or duplicate values. Above 450 Hz there is almost no energy, so a '
             '20–450 Hz band-pass and a 60 Hz notch are enough.'),
            ('Imbalance is common and stable', '59% of reps and 20 of 33 participants are imbalanced. Set 1 and set 2 '
             'agree at r ≈ 0.96.'),
            ('The left leg is usually weaker', '68% of imbalanced reps. Without each participant\'s dominant leg this '
             'stays a description, not an explanation.'),
        ]),
        ('Feature insights', [
            ('Amplitude = label', 'Amplitude features correlate 0.8–1.0 with SI because SI is computed from RMS. They '
             'were excluded to avoid leakage.'),
            ('SSC and ZC lead – indirectly', '<code>ssc_rate_si</code> (ρ 0.61, AUC 0.75) and <code>zc_rate_si</code> '
             'are associated with SI. The weaker leg\'s signal carries relatively more noise, so the link runs '
             'through amplitude: correlation, not an independent cause.'),
            ('Some features repeat', 'MNF ≈ Hjorth mobility ≈ spectral entropy (r 0.90–0.98), and env_xcorr_peak ≈ '
             'env_corr0 (0.97). One of each group is enough.'),
        ]),
        ('Model findings', [
            ('3 features beat 64', 'Ridge: R² 0.57 vs 0.49 on new participants, with the train–test gap falling from '
             '0.31 to 0.06. Flexible models (RF, SVR) overfit with 64 features.'),
            ('A shared ceiling', 'With 3–5 features LogReg, RF, SVM and kNN all reach balanced accuracy ≈ 0.59–0.66. '
             'The limit is the information in the features, not the algorithm.'),
            ('Useful at the person level', 'Per participant R² is 0.63; most rep-level errors sit near the 0.15 '
             'threshold.'),
        ]),
    ]
    for i, (title, items) in enumerate(groups):
        section(f'0{i + 1}', title)
        cols = st.columns(3)
        for col, (t, b) in zip(cols, items):
            card(col, title.split()[0].upper(), t, b)

    section('04', 'Limitations')
    lims = [('Small, specific sample', '33 participants, all female. The learning curve levels off after ~18 people.'),
            ('The label depends on amplitude', 'SI comes from RMS, which also depends on electrode placement, skin and '
             'fat. No amplitude-free feature can see those differences.'),
            ('Short sets', 'Only 5 reps per set: too short to link fatigue to imbalance.')]
    cols = st.columns(3)
    for col, (t, b) in zip(cols, lims):
        card(col, 'LIMITATION', t, b)

    section('05', 'Next steps', 'Recommendations – not yet done.')
    nxt = [('Normalise with MVC', 'Record a maximal voluntary contraction per leg to remove electrode and tissue '
            'effects from amplitude.'),
           ('A label from outside the EMG', 'Use a force plate or a single-leg strength test as the reference for '
            'imbalance.'),
           ('Record dominance and more people', 'Add each participant\'s dominant leg and more varied participants.')]
    cols = st.columns(3)
    for col, (t, b) in zip(cols, nxt):
        card(col, 'NEXT STEP', t, b, blush=True)


@st.cache_data
def load_envelopes():
    return json.loads((HERE / 'envelopes.json').read_text())


def exercise_viewer():
    """3D squat with the quadriceps coloured by the recorded envelope of one set (viewer3d.html)."""
    env = load_envelopes()
    sets = df_all.groupby('set_name').agg(subject=('subject', 'first'), trial=('trial', 'first'), si=('si_rms', 'mean'))
    sets = sets[sets.index.isin(env)].sort_values('si', ascending=False)

    def name(s):
        r = sets.loc[s]
        return f'{r.subject} · set {r.trial}  (mean SI {r.si:.2f}, {"imbalance" if r.si >= SI_THRESHOLD else "balanced"})'

    options = sets.index.tolist()
    c1, c2 = st.columns([2, 3])
    choice = c1.selectbox('Recording', options, index=options.index('Wintana_1') if 'Wintana_1' in options else 0,
                          format_func=name)
    c2.markdown('<div class="read" style="margin-top:28px">Try <b>Wintana · set 1</b> (large imbalance) and '
                '<b>Angie · set 1</b> (balanced) to see the difference.</div>', unsafe_allow_html=True)

    reps = df_all[df_all.set_name == choice][['rep', 'si_rms', 'label', 'weak_side']]
    data = dict(env[choice], title=f'{sets.loc[choice].subject} · set {sets.loc[choice].trial}',
                reps=[{'rep': int(r.rep), 'si': float(r.si_rms), 'label': int(r.label), 'weak': r.weak_side}
                      for r in reps.itertuples()])
    html = (HERE / 'viewer3d.html').read_text(encoding='utf-8').replace('__DATA__', json.dumps(data))
    components.html(html, height=770, scrolling=False)

    cols = st.columns(3)
    card(cols[0], 'RECORDED', 'Activation and SI',
         'The colour of each quadriceps and the left/right bars follow the <b>real EMG envelope</b> of this set '
         '(same filters as the notebook). The SI of each rep is the value in the cleaned dataset.')
    card(cols[1], 'ILLUSTRATIVE', 'The movement',
         'There was no motion capture. Reps come from the <b>real rep markers</b>, and the down-and-up of each rep is '
         'placed around its EMG burst (bottom = burst peak). The phases are an estimate, not a measurement.')
    card(cols[2], 'MODEL', 'Figure and muscles',
         'A rigged character (Mixamo "Michelle", from the three.js examples), posed for the squat. The coloured area '
         'marks where the quadriceps lie; it is not a muscle simulation. One sensor covers the whole muscle on each leg.',
         blush=True)


pages = [
    st.Page(page_overview, title='Overview', icon=':material/space_dashboard:', default=True),
    st.Page(page_eda, title='EDA', icon=':material/query_stats:', url_path='eda'),
    st.Page(page_quality, title='Data quality & preprocessing', icon=':material/tune:', url_path='data-quality'),
    st.Page(page_model, title='Model performance', icon=':material/model_training:', url_path='model'),
    st.Page(page_insights, title='Insights & conclusions', icon=':material/lightbulb:', url_path='insights'),
]
st.navigation(pages, position='sidebar').run()
