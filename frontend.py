
# app.py
import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
import os

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="NFL Predictor",
    page_icon="🏟️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CSS to mimic the look/format
# -----------------------------
CSS = """
<style>
/* Hide Streamlit chrome */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

/* App background (purple space vibe) */
.stApp {
  background: radial-gradient(circle at 50% 20%, rgba(124, 77, 255, 0.35), rgba(0,0,0,0) 45%),
              radial-gradient(circle at 80% 10%, rgba(255, 99, 132, 0.18), rgba(0,0,0,0) 40%),
              linear-gradient(180deg, #0b0620 0%, #09041b 40%, #070315 100%);
}

/* Subtle network overlay */
.bg-net {
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: 0.25;
  background-image:
    radial-gradient(circle at 10% 70%, rgba(255,255,255,0.08) 0 2px, transparent 3px),
    radial-gradient(circle at 30% 85%, rgba(255,255,255,0.08) 0 2px, transparent 3px),
    radial-gradient(circle at 55% 75%, rgba(255,255,255,0.08) 0 2px, transparent 3px),
    radial-gradient(circle at 70% 90%, rgba(255,255,255,0.08) 0 2px, transparent 3px),
    radial-gradient(circle at 85% 80%, rgba(255,255,255,0.08) 0 2px, transparent 3px);
  filter: blur(0.2px);
}
.bg-net:before {
  content: "";
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(120deg, rgba(158, 114, 255, 0.18) 1px, transparent 1px),
    linear-gradient(60deg, rgba(158, 114, 255, 0.12) 1px, transparent 1px);
  background-size: 220px 220px;
  mix-blend-mode: screen;
  opacity: 0.35;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
  background: rgba(10, 6, 28, 0.65);
  border-right: 1px solid rgba(255,255,255,0.06);
  backdrop-filter: blur(8px);
}
.sidebar-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 18px;
  padding: 16px;
}

/* Main hero */
.hero-title {
  text-align: center;
  font-size: 56px;
  font-weight: 800;
  letter-spacing: 0.5px;
  color: rgba(255,255,255,0.92);
  margin-top: 8px;
  margin-bottom: 2px;
}
.hero-subtitle {
  text-align: center;
  font-size: 16px;
  color: rgba(255,255,255,0.72);
  margin-top: 0px;
  margin-bottom: 26px;
}

/* Center card */
.center-card {
  max-width: 980px;
  margin: 0 auto;
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 20px;
  padding: 22px 22px 18px 22px;
  backdrop-filter: blur(10px);
  box-shadow: 0 16px 60px rgba(0,0,0,0.35);
}

.card-title {
  font-size: 28px;
  font-weight: 750;
  color: rgba(255,255,255,0.92);
  margin-bottom: 10px;
}

.small-label {
  color: rgba(255,255,255,0.75);
  font-size: 12px;
  margin-bottom: -6px;
}

/* Make inputs darker */
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] div,
div[data-testid="stTextInput"] input {
  background: rgba(6, 4, 20, 0.55) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
}

/* Predict button wide */
div.stButton > button {
  width: 100%;
  border-radius: 12px;
  padding: 14px 16px;
  font-weight: 700;
  border: 1px solid rgba(255,255,255,0.16);
  background: rgba(122, 104, 255, 0.78);
}
div.stButton > button:hover {
  background: rgba(122, 104, 255, 0.88);
  border: 1px solid rgba(255,255,255,0.22);
}

/* Result pill */
.result-pill {
  margin-top: 14px;
  padding: 14px 16px;
  border-radius: 14px;
  border: 1px solid rgba(255,255,255,0.14);
  background: rgba(0,0,0,0.25);
}
</style>
<div class="bg-net"></div>
"""
st.markdown(CSS, unsafe_allow_html=True)

# -----------------------------
# Model integration using prediction_ready.csv
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), "prediction_ready.csv")

FEATURE_COLS = [
    "HomeWins", "HomeLosses", "AwayWins", "AwayLosses",
    "HomeRestDays", "AwayRestDays", "H2H_HomeWinPct",
    "HomeRollingScore", "HomeRollingDefense",
    "AwayRollingScore", "AwayRollingDefense",
    "Home_AvgPointsScored", "Home_AvgPointsAllowed",
    "Away_AvgPointsScored", "Away_AvgPointsAllowed",
    "HomeWonLastH2H", "PostSeason",
]

@dataclass
class Prediction:
    winner: str
    probability: float  # 0..1

@st.cache_resource
def load_data_and_model():
    df = pd.read_csv(DATA_PATH)
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + ["HomeWin"])

    X = df[FEATURE_COLS].values
    y = df["HomeWin"].astype(int).values

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X, y)

    teams = sorted(set(df["HomeTeam"].unique()) | set(df["AwayTeam"].unique()))

    # Build per-team average stats for lookup at prediction time
    home_stats = df.groupby("HomeTeam")[FEATURE_COLS].mean()
    away_stats = df.groupby("AwayTeam")[FEATURE_COLS].mean()

    return clf, teams, df, home_stats, away_stats

clf, TEAMS, df_all, home_stats, away_stats = load_data_and_model()

def predict_match(home_team: str, away_team: str, postseason: int) -> Prediction:
    # Build feature vector from historical averages for the two teams
    features = {}
    for col in FEATURE_COLS:
        if col.startswith("Home"):
            features[col] = home_stats.loc[home_team, col] if home_team in home_stats.index else 0.0
        elif col.startswith("Away"):
            features[col] = away_stats.loc[away_team, col] if away_team in away_stats.index else 0.0
        elif col == "H2H_HomeWinPct":
            h2h = df_all[
                (df_all["HomeTeam"] == home_team) & (df_all["AwayTeam"] == away_team)
            ]["H2H_HomeWinPct"]
            features[col] = h2h.mean() if len(h2h) > 0 else 0.5
        elif col == "HomeWonLastH2H":
            h2h = df_all[
                (df_all["HomeTeam"] == home_team) & (df_all["AwayTeam"] == away_team)
            ]["HomeWonLastH2H"]
            features[col] = h2h.iloc[-1] if len(h2h) > 0 else 0.0
        elif col == "PostSeason":
            features[col] = postseason
        else:
            features[col] = 0.0

    X_input = np.array([[features[c] for c in FEATURE_COLS]])
    proba = clf.predict_proba(X_input)[0]  # [P(away_win), P(home_win)]
    class_labels = list(clf.classes_)
    p_home = float(proba[class_labels.index(1)])

    if p_home >= 0.5:
        return Prediction(winner=home_team, probability=p_home)
    else:
        return Prediction(winner=away_team, probability=1.0 - p_home)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center; margin-top:10px; color:rgba(255,255,255,0.8); font-weight:700; font-size:18px;">
          NFL Predictor
        </div>
        <div style="text-align:center; margin-top:14px; color:rgba(255,255,255,0.65); font-size:12px; line-height:1.55;">
          <b>Tech Lead:</b> Eric<br/>
          <b>Team:</b> Brandon, Sharrav, Soren<br/><br/>
          MLSN Winter Cohort 2026
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    with st.expander("⚙️  Model Information", expanded=False):
        st.write("**Task:** Predict match winner (binary classification).")
        st.write("**Inputs:** Home team, Away team, Game type.")
        st.write("**Output:** Predicted winner + probability (confidence).")
        st.write(f"**Model:** Random Forest (200 trees, max depth 10).")
        st.write(f"**Training data:** {len(df_all):,} games from prediction_ready.csv.")

    st.markdown(
        """
        <div style="margin-top:18px; text-align:center; color:rgba(255,255,255,0.55); font-size:12px;">
          © Machine Learning Student Network – 2026
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Main hero
# -----------------------------
st.markdown('<div class="hero-title">NFL Predictor</div>', unsafe_allow_html=True)

# -----------------------------
# Center card content
# -----------------------------
st.markdown('<div class="center-card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🏈 Match Details</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2, gap="large")

with c1:
    st.markdown('<div class="small-label">🏠 Home Team</div>', unsafe_allow_html=True)
    home_team = st.selectbox("Home Team", TEAMS, label_visibility="collapsed")

with c2:
    st.markdown('<div class="small-label">🚌 Away Team</div>', unsafe_allow_html=True)
    away_team = st.selectbox("Away Team", TEAMS, index=1 if len(TEAMS) > 1 else 0, label_visibility="collapsed")

st.markdown('<div class="small-label">🧠 Game Type</div>', unsafe_allow_html=True)
game_type = st.selectbox("Game Type", ["Regular Season", "Playoffs"], label_visibility="collapsed")
postseason = 1 if game_type == "Playoffs" else 0

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

predict = st.button("🔮  Predict Winner", use_container_width=True)

if predict:
    if home_team == away_team:
        st.warning("Pick two different teams.")
    else:
        pred = predict_match(home_team, away_team, postseason)
        pct = pred.probability * 100.0

        st.markdown(
            f"""
            <div class="result-pill">
              <div style="font-size:18px; font-weight:750; color:rgba(255,255,255,0.92);">
                Predicted Winner: <span style="color:rgba(160, 140, 255, 0.95);">{pred.winner}</span>
              </div>
              <div style="margin-top:6px; color:rgba(255,255,255,0.75);">
                Probability correct: <b>{pct:.1f}%</b>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("</div>", unsafe_allow_html=True)
