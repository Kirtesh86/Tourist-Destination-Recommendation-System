"""KNN-powered travel destination recommendations in a beginner-friendly UI."""

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Discover India | Travel finder",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DATA_PATH = Path(__file__).with_name("tourist_destinations.csv")
FEATURES = ["Budget", "Days", "Nature", "Beach", "Adventure", "Culture"]
INTERESTS = ["Nature", "Beach", "Adventure", "Culture"]
RATING_DEFAULTS = {
    "nature_interest": 7,
    "beach_interest": 4,
    "adventure_interest": 6,
    "culture_interest": 7,
}

# Editorial destination photography from Unsplash. The CSS color remains visible
# behind each image if a visitor is offline or the photo host is unavailable.
PHOTOS = {
    "hero": "https://images.unsplash.com/photo-1524492412937-b28074a5d7da?auto=format&fit=crop&w=1800&q=85",
    "Nature": "https://images.unsplash.com/photo-1448375240586-882707db888b?auto=format&fit=crop&w=1000&q=82",
    "Beach": "https://images.unsplash.com/photo-1500375592092-40eb2168fd21?auto=format&fit=crop&w=1000&q=82",
    "Adventure": "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?auto=format&fit=crop&w=1000&q=82",
    "Culture": "https://images.unsplash.com/photo-1548013146-72479768bada?auto=format&fit=crop&w=1000&q=82",
    "Desert": "https://images.unsplash.com/photo-1509316785289-025f5b846b35?auto=format&fit=crop&w=1000&q=82",
    "Lake": "https://images.unsplash.com/photo-1501785888041-af3ef285b470?auto=format&fit=crop&w=1000&q=82",
}

# Curated imagery for recognizable destinations; other places use the photo
# that best represents their strongest destination interest score.
DESTINATION_PHOTOS = {
    "Goa": PHOTOS["Beach"], "Manali": PHOTOS["Adventure"], "Jaipur": PHOTOS["Culture"],
    "Kerala": PHOTOS["Nature"], "Rishikesh": PHOTOS["Adventure"], "Mumbai": PHOTOS["Culture"],
    "Udaipur": PHOTOS["Lake"], "Andaman Islands": PHOTOS["Beach"], "Shimla": PHOTOS["Adventure"],
    "Agra": PHOTOS["hero"], "Ladakh": PHOTOS["Lake"], "Darjeeling": PHOTOS["Nature"],
    "Mysuru": PHOTOS["Culture"], "Ooty": PHOTOS["Nature"], "Jaisalmer": PHOTOS["Desert"],
    "Varanasi": PHOTOS["Culture"], "Amritsar": PHOTOS["Culture"], "Kashmir": PHOTOS["Lake"],
    "Coorg": PHOTOS["Nature"], "Puducherry": PHOTOS["Beach"], "Hampi": PHOTOS["Culture"],
    "Ranthambore": PHOTOS["Nature"], "Sundarbans": PHOTOS["Nature"], "Lakshadweep": PHOTOS["Beach"],
    "Kutch": PHOTOS["Desert"], "Mahabalipuram": PHOTOS["Culture"], "Shillong": PHOTOS["Nature"],
    "Tawang": PHOTOS["Adventure"], "Rann of Kutch": PHOTOS["Desert"], "Kanyakumari": PHOTOS["Beach"],
    "Ajanta and Ellora": PHOTOS["Culture"], "Chikmagalur": PHOTOS["Nature"], "Kochi": PHOTOS["Beach"],
    "Alleppey": PHOTOS["Lake"], "Spiti Valley": PHOTOS["Adventure"],
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
    :root { --ink:#18382e; --green:#245b48; --muted:#68766e; --paper:#f6f5f0; --coral:#e9784d; }
    .stApp { background:var(--paper); color:var(--ink); }
    .block-container { max-width:1180px; padding:2rem 2rem 4rem; }
    h1,h2,h3,h4 { color:var(--ink); }
    h1,h2 { font-family:'Playfair Display',Georgia,serif; letter-spacing:-.6px; }
    p,label,span,div { font-family:'DM Sans',sans-serif; }
    .hero { min-height:290px; display:flex; align-items:center; justify-content:space-between; gap:2rem; padding:2.7rem 3rem; border-radius:24px; color:#fff; margin:0 0 2.2rem; background-color:#173b30; background-position:center 45%; background-size:cover; box-shadow:0 18px 45px #173b3020; }
    .hero-copy { max-width:650px; }
    .hero .eyebrow { color:#b9ddc8; text-transform:uppercase; letter-spacing:2.3px; font-size:.72rem; font-weight:700; }
    .hero h1 { color:white; font-size:clamp(2.3rem,5vw,3.65rem); line-height:1.08; margin:.65rem 0 .7rem; }
    .hero p { color:#e0ebe4; font-size:1.02rem; line-height:1.6; max-width:560px; margin:0; }
    .hero-stats { flex:none; border:1px solid #ffffff55; background:#ffffff18; backdrop-filter:blur(8px); border-radius:18px; padding:1rem 1.25rem; text-align:center; }
    .hero-stats strong { display:block; font-family:'Playfair Display',Georgia,serif; font-size:2rem; line-height:1.1; }
    .hero-stats span { display:block; color:#e0ebe4; font-size:.77rem; margin-top:.25rem; }
    .section-kicker { color:var(--coral); text-transform:uppercase; letter-spacing:1.8px; font-weight:700; font-size:.72rem; margin-bottom:.25rem; }
    .section-intro { color:var(--muted); margin-top:-.45rem; margin-bottom:1.1rem; }
    div[data-testid="stForm"] { background:#fff; border:1px solid #e7e8e1; border-radius:20px; padding:1.35rem 1.5rem 1.2rem; box-shadow:0 12px 35px #173b300b; }
    .form-panel { min-height:100%; background:#f8f8f4; border:1px solid #eceee7; border-radius:15px; padding:1.1rem 1.15rem .7rem; }
    .panel-title { color:var(--ink); font-weight:700; font-size:1.02rem; margin:0 0 .15rem; }
    .panel-note { color:var(--muted); font-size:.83rem; margin:0 0 .85rem; line-height:1.45; }
    .field-label { color:#53665c; font-size:.84rem; font-weight:600; }
    .rating-pill { text-align:right; color:var(--green); background:#e8f1eb; border-radius:20px; padding:.16rem .58rem; font-weight:700; font-size:.76rem; white-space:nowrap; }
    .stSlider [data-baseweb="slider"] div[role="slider"] { background:var(--green); border-color:var(--green); }
    .stSlider [data-testid="stTickBar"] { display:none; }
    .stButton > button, div[data-testid="stFormSubmitButton"] button { min-height:3rem; background:var(--coral); color:#fff; border:0; border-radius:11px; font-weight:700; transition:transform .15s ease,background .15s ease; }
    .stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover { background:#d86740; color:white; border:0; transform:translateY(-1px); }
    .results-heading { padding-top:1.1rem; }
    .result-card { overflow:hidden; background:#fff; border:1px solid #e5e8e2; border-radius:17px; margin:.3rem 0 .8rem; box-shadow:0 8px 24px #173b300c; }
    .result-photo { height:150px; position:relative; background:linear-gradient(135deg,#aac6a9,#365e4d); background-size:cover; background-position:center; }
    .match-rank { position:absolute; top:12px; left:13px; border-radius:20px; padding:.34rem .62rem; color:#18382e; background:#ffffffed; text-transform:uppercase; letter-spacing:1px; font-size:.67rem; font-weight:700; }
    .photo-tag { position:absolute; bottom:12px; left:13px; border-radius:20px; padding:.33rem .62rem; color:#fff; background:#173b30cf; font-size:.72rem; font-weight:600; }
    .result-body { padding:1rem 1.1rem 1.15rem; }
    .place { font-family:'Playfair Display',Georgia,serif; font-size:1.42rem; font-weight:700; color:var(--ink); line-height:1.2; margin:.1rem 0 .8rem; }
    .metric-grid { display:grid; grid-template-columns:1fr 1fr; gap:.55rem; }
    .metric { background:#f7f8f4; padding:.57rem .65rem; border-radius:10px; }
    .metric span { display:block; color:var(--muted); font-size:.68rem; text-transform:uppercase; letter-spacing:.7px; }
    .metric strong { display:block; color:var(--ink); font-size:.9rem; margin-top:.12rem; }
    .budget-fit { margin-top:.68rem; border-radius:9px; padding:.52rem .65rem; font-size:.78rem; font-weight:600; }
    .fit-ok { color:#246047; background:#eaf4ec; }
    .fit-near { color:#8a5a14; background:#fff4df; }
    .fit-over { color:#8d4b37; background:#fff0ea; }
    .location-line { color:var(--muted); font-size:.78rem; margin:-.45rem 0 .75rem; }
    div[data-testid="stExpander"] { border:1px solid #e5e8e2; border-radius:12px; background:#fff; }
    .compare-legend { display:flex; gap:1rem; color:var(--muted); font-size:.74rem; margin:.25rem 0 .7rem; }
    .legend-destination::before,.legend-you::before { content:""; display:inline-block; width:9px; height:9px; border-radius:3px; margin-right:.35rem; }
    .legend-destination::before { background:#9dbca4; }
    .legend-you::before { background:var(--coral); }
    .compare-label { display:flex; justify-content:space-between; color:#53665c; font-size:.78rem; margin:.4rem 0 .17rem; }
    .compare-track { height:8px; background:#edf0eb; border-radius:8px; position:relative; margin-bottom:.55rem; }
    .destination-fill { height:100%; background:#9dbca4; border-radius:8px; }
    .you-marker { position:absolute; width:3px; height:14px; top:-3px; background:var(--coral); border-radius:2px; transform:translateX(-1px); }
    .fine-print { color:#778178; font-size:.78rem; text-align:center; padding-top:.9rem; }
    @media (max-width:700px) {
      .block-container { padding:1rem 1rem 3rem; }
      .hero { min-height:260px; padding:1.7rem 1.4rem; border-radius:18px; }
      .hero-stats { display:none; }
      div[data-testid="stForm"] { padding:.8rem; }
      .form-panel { padding:.9rem .85rem .55rem; }
      .result-photo { height:125px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <section class="hero" style="background-image:linear-gradient(90deg,rgba(13,42,32,.90) 0%,rgba(13,42,32,.67) 56%,rgba(13,42,32,.20) 100%),url('{PHOTOS['hero']}')">
      <div class="hero-copy">
        <div class="eyebrow">A little inspiration for your next trip</div>
        <h1>Find your kind of India.</h1>
        <p>From quiet mountain mornings to a little more adventure, tell us what you love and find a place that feels like you.</p>
      </div>
      <div class="hero-stats"><strong>35</strong><span>places to explore</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(f"Could not find the destination dataset: {DATA_PATH.name}")
    st.stop()

try:
    destinations = pd.read_csv(DATA_PATH)
    missing = set(["Destination", "State", "Region", *FEATURES]) - set(destinations.columns)
    if missing:
        st.error(f"The dataset is missing required columns: {', '.join(sorted(missing))}")
        st.stop()
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not read the destination dataset: {error}")
    st.stop()

st.markdown('<div class="section-kicker">Start with your travel style</div>', unsafe_allow_html=True)
st.markdown("## What would make this trip yours?")
st.markdown('<div class="section-intro">A few quick details help us find destinations that fit your budget, time, and interests.</div>', unsafe_allow_html=True)

for rating_key, default_rating in RATING_DEFAULTS.items():
    if rating_key not in st.session_state:
        st.session_state[rating_key] = default_rating

with st.form("travel_preferences"):
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown(
            '<div class="form-panel"><div class="panel-title">🧳 Budget & time</div><div class="panel-note">Choose a comfortable spend and how long you can get away.</div></div>',
            unsafe_allow_html=True,
        )
        budget = st.number_input("Trip budget (₹)", min_value=5_000, max_value=200_000, value=30_000, step=1_000)
        days = st.number_input("Days away", min_value=1, max_value=30, value=5, step=1)
        regions = ["All India", *sorted(destinations["Region"].dropna().unique())]
        selected_region = st.selectbox("Explore a region", regions, help="Filter destinations by region before KNN ranks your matches.")
        st.caption("Sample trip budgets are illustrative estimates.")

    with right:
        st.markdown(
            '<div class="form-panel"><div class="panel-title">✦ What you love</div><div class="panel-note">Rate each from 1 (not for me) to 10 (love it).</div></div>',
            unsafe_allow_html=True,
        )

        def interest_slider(label: str, icon: str, key: str) -> int:
            heading, value_column = st.columns([4, 1])
            current_value = int(st.session_state[key])
            with heading:
                st.markdown(f'<div class="field-label">{icon} &nbsp;{label}</div>', unsafe_allow_html=True)
            with value_column:
                st.markdown(f'<div class="rating-pill">{current_value} / 10</div>', unsafe_allow_html=True)
            value = st.slider(label, min_value=1, max_value=10, key=key, label_visibility="collapsed")
            return value

        nature = interest_slider("Nature & scenery", "🌿", "nature_interest")
        beach = interest_slider("Beaches", "🏖️", "beach_interest")
        adventure = interest_slider("Adventure", "⛰️", "adventure_interest")
        culture = interest_slider("Culture & history", "🏛️", "culture_interest")

    submitted = st.form_submit_button("Find my destinations  →", use_container_width=True)

if submitted:
    # Scale all destination attributes together so rupee values do not overwhelm
    # the 1–10 ratings when KNN calculates Euclidean distances.
    scaler = StandardScaler()
    destination_features = destinations[FEATURES].astype(float)
    scaled_destinations = scaler.fit_transform(destination_features)
    if selected_region == "All India":
        candidate_positions = np.arange(len(destinations))
    else:
        candidate_positions = np.flatnonzero(destinations["Region"].to_numpy() == selected_region)

    # Keep the scaler fitted on all destinations so distances stay comparable;
    # the region filter only limits which destinations KNN may return.
    # Transform the traveler with that scaler, then retrieve up to five matches.
    neighbor_count = min(5, len(candidate_positions))
    knn = NearestNeighbors(n_neighbors=neighbor_count, metric="euclidean")
    knn.fit(scaled_destinations[candidate_positions])
    user_profile = np.array([[budget, days, nature, beach, adventure, culture]], dtype=float)
    distances, local_indices = knn.kneighbors(scaler.transform(user_profile))
    matched_positions = candidate_positions[local_indices[0]]

    st.markdown('<div class="results-heading"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Your shortlist</div>', unsafe_allow_html=True)
    st.markdown("## Places with your kind of vibe")
    st.markdown(f'<div class="section-intro">{escape(selected_region)} · Ranked by KNN distance. Lower distance means a closer match to your preferences.</div>', unsafe_allow_html=True)

    scored = list(zip(matched_positions, distances[0]))
    result_columns = st.columns(2, gap="large")
    for rank, (index, distance) in enumerate(scored, start=1):
        match = destinations.iloc[index]
        profile = {interest: int(match[interest]) for interest in INTERESTS}
        strongest = max(profile, key=profile.get)
        photo = DESTINATION_PHOTOS.get(str(match["Destination"]), PHOTOS[strongest])
        place = escape(str(match["Destination"]))
        state = escape(str(match["State"]))
        region = escape(str(match["Region"]))
        trip_budget = float(match["Budget"])
        trip_days = int(match["Days"])
        difference = trip_budget - float(budget)

        if difference <= 0:
            budget_class = "fit-ok"
            budget_text = f"✓ Within budget · ₹{abs(difference):,.0f} estimated room"
        elif difference / max(float(budget), 1) <= 0.15:
            budget_class = "fit-near"
            budget_text = f"Near your budget · ₹{difference:,.0f} above your estimate"
        else:
            budget_class = "fit-over"
            budget_text = f"Above your budget · ₹{difference:,.0f} above your estimate"

        with result_columns[(rank - 1) % len(result_columns)]:
            st.markdown(
                f"""
                <article class="result-card">
                  <div class="result-photo" style="background-image:linear-gradient(0deg,rgba(16,42,32,.27),transparent 70%),url('{photo}')">
                    <span class="match-rank">Match {rank:02d}</span>
                    <span class="photo-tag">{strongest} escape</span>
                  </div>
                  <div class="result-body">
                    <div class="place">{place}</div>
                    <div class="location-line">{state} &nbsp;·&nbsp; {region}</div>
                    <div class="metric-grid">
                      <div class="metric"><span>Est. trip budget</span><strong>₹{trip_budget:,.0f}</strong></div>
                      <div class="metric"><span>Suggested stay</span><strong>{trip_days} days</strong></div>
                      <div class="metric"><span>Match distance</span><strong>{distance:.2f}</strong></div>
                      <div class="metric"><span>Top interest</span><strong>{strongest}</strong></div>
                    </div>
                    <div class="budget-fit {budget_class}">{budget_text}</div>
                  </div>
                </article>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"Why {place} made your list"):
                st.markdown("**Your interest ratings compared with this destination**")
                st.markdown(
                    '<div class="compare-legend"><span class="legend-destination">Destination score</span><span class="legend-you">Your score</span></div>',
                    unsafe_allow_html=True,
                )
                comparison_rows = []
                for interest in INTERESTS:
                    user_score = {"Nature": nature, "Beach": beach, "Adventure": adventure, "Culture": culture}[interest]
                    destination_score = profile[interest]
                    comparison_rows.append(
                        f'<div class="compare-label"><span>{interest}</span><span>You {user_score}/10 · Place {destination_score}/10</span></div>'
                        f'<div class="compare-track"><div class="destination-fill" style="width:{destination_score * 10}%"></div>'
                        f'<span class="you-marker" style="left:{user_score * 10}%"></span></div>'
                    )
                st.markdown("".join(comparison_rows), unsafe_allow_html=True)
                st.caption("Project-assigned interest scores help explain the profile; they are not visitor ratings.")

st.markdown(
    '<div class="fine-print">Budgets and destination interest scores are illustrative project data, not live travel quotes.</div>',
    unsafe_allow_html=True,
)
