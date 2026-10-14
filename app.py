"""Wayfinder: a compact KNN travel destination planner."""

from __future__ import annotations

import html
import json
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


st.set_page_config(page_title="Wayfinder | Travel planner", page_icon="✦", layout="wide", initial_sidebar_state="collapsed")

CITY_PATH = Path(__file__).with_name("worldwide_travel_cities.csv")
INTERESTS = ["culture", "adventure", "nature", "beaches", "nightlife", "cuisine", "wellness", "urban", "seclusion"]
TRAVEL_STYLES = {
    "Cultural Immersion": {"culture": 1.0, "cuisine": 0.25, "urban": 0.2},
    "Nature and Wildlife": {"nature": 1.0, "seclusion": 0.35, "wellness": 0.15},
    "Adventure and Thrill Sports": {"adventure": 1.0, "nature": 0.3},
    "Food and Culinary Tourism": {"cuisine": 1.0, "culture": 0.25, "urban": 0.2},
    "Urban Exploration": {"urban": 1.0, "nightlife": 0.45, "culture": 0.2},
    "Wellness and Slow Travel": {"wellness": 1.0, "seclusion": 0.7, "nature": 0.2},
}
FALLBACK_USD_PER_CURRENCY = {"USD": 1.0, "INR": 0.012, "EUR": 1.09, "GBP": 1.28, "AED": 0.27, "AUD": 0.66, "CAD": 0.74, "JPY": 0.0067}
CURRENCY_SYMBOLS = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "AED": "AED ", "AUD": "A$", "CAD": "C$", "JPY": "¥"}
REGION_LABELS = {"africa": "Africa", "asia": "Asia", "europe": "Europe", "middle_east": "Middle East", "north_america": "North America", "oceania": "Oceania", "south_america": "South America"}
DURATION_LABELS = {"Day trip": "1 day", "Weekend": "2–3 days", "Short trip": "3–5 days", "One week": "7 days", "Long trip": "10+ days"}
DURATION_DAYS = {"Day trip": 1.0, "Weekend": 2.5, "Short trip": 4.0, "One week": 7.0, "Long trip": 12.0}
DAILY_USD_BY_TIER = {"Budget": 85.0, "Mid-range": 180.0, "Luxury": 360.0}
THEME_IMAGES = {
    "culture": "https://images.unsplash.com/photo-1530789253388-582c481c54b0?auto=format&fit=crop&w=900&q=85",
    "adventure": "https://images.unsplash.com/photo-1464822759023-fed622ff2c3b?auto=format&fit=crop&w=900&q=85",
    "nature": "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee?auto=format&fit=crop&w=900&q=85",
    "beaches": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=900&q=85",
    "nightlife": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?auto=format&fit=crop&w=900&q=85",
    "cuisine": "https://images.unsplash.com/photo-1504674900247-0877df9cc836?auto=format&fit=crop&w=900&q=85",
    "wellness": "https://images.unsplash.com/photo-1470770841072-f978cf4d019e?auto=format&fit=crop&w=900&q=85",
    "urban": "https://images.unsplash.com/photo-1519501025264-65ba15a82390?auto=format&fit=crop&w=900&q=85",
    "seclusion": "https://images.unsplash.com/photo-1469474968028-56623f02e42e?auto=format&fit=crop&w=900&q=85",
}
WEATHER_LABELS = {0: "Clear", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Foggy", 48: "Rime fog", 51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain", 71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers", 81: "Rain showers", 82: "Heavy showers", 95: "Thunderstorm"}


@st.cache_data(show_spinner=False)
def load_city_data() -> pd.DataFrame:
    """Load and validate the city profiles used by the recommender."""
    data = pd.read_csv(CITY_PATH)
    required = {"id", "city", "country", "region", "short_description", "latitude", "longitude", "ideal_durations", "budget_level", *INTERESTS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"City dataset is missing columns: {', '.join(sorted(missing))}")
    data[INTERESTS] = data[INTERESTS].apply(pd.to_numeric, errors="coerce")
    return data.dropna(subset=[*INTERESTS, "latitude", "longitude"]).assign(id=lambda frame: frame["id"].astype(str)).reset_index(drop=True)


def parse_duration_tags(value: object) -> list[str]:
    try:
        tags = json.loads(str(value))
        if isinstance(tags, list):
            return [str(tag) for tag in tags]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return [tag.strip() for tag in str(value).split(",") if tag.strip()]


@st.cache_data(ttl=3600, show_spinner=False)
def get_live_usd_rates() -> tuple[dict[str, float], str, bool]:
    """Get USD conversion factors from Frankfurter; retain a safe offline fallback."""
    url = "https://api.frankfurter.dev/v1/latest?base=USD&symbols=INR,EUR,GBP,AUD,CAD,JPY"
    try:
        request = Request(url, headers={"User-Agent": "Wayfinder/1.0 Streamlit travel planner"})
        with urlopen(request, timeout=7) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rates = dict(FALLBACK_USD_PER_CURRENCY)
        for currency, per_usd in payload["rates"].items():
            rates[currency] = 1 / float(per_usd)
        return rates, str(payload.get("date", "today")), True
    except (HTTPError, URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return dict(FALLBACK_USD_PER_CURRENCY), "offline fallback", False


@st.cache_data(ttl=900, show_spinner=False)
def get_weather(latitude: float, longitude: float) -> dict[str, object] | None:
    """Fetch a small live forecast from Open-Meteo without storing user data."""
    query = (
        "https://api.open-meteo.com/v1/forecast?latitude="
        f"{latitude:.4f}&longitude={longitude:.4f}"
        "&current=temperature_2m,apparent_temperature,weather_code"
        "&daily=temperature_2m_max,temperature_2m_min,weather_code&forecast_days=3&timezone=auto"
    )
    try:
        request = Request(query, headers={"User-Agent": "Wayfinder/1.0 Streamlit travel planner"})
        with urlopen(request, timeout=7) as response:
            return json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
        return None


def format_money(amount: float, currency: str) -> str:
    decimals = 0 if currency in {"INR", "JPY"} else 2
    return f"{CURRENCY_SYMBOLS[currency]}{amount:,.{decimals}f} {currency}"


def budget_tier(amount: float, currency: str, rates: dict[str, float]) -> tuple[str, float]:
    """Map a user-entered total to the three categories included in the source dataset."""
    budget_in_usd = amount * rates[currency]
    return ("Budget" if budget_in_usd < 750 else "Mid-range" if budget_in_usd < 2500 else "Luxury"), budget_in_usd


def city_days(row: pd.Series, requested_duration: str) -> float:
    if requested_duration in DURATION_DAYS:
        return DURATION_DAYS[requested_duration]
    choices = [DURATION_DAYS[tag] for tag in parse_duration_tags(row["ideal_durations"]) if tag in DURATION_DAYS]
    return sum(choices) / len(choices) if choices else 4.0


def estimated_cost(row: pd.Series, requested_duration: str, currency: str, rates: dict[str, float]) -> tuple[float, float, float]:
    """Create a transparent planning estimate; the source does not provide live prices."""
    days = city_days(row, requested_duration)
    urban_adjustment = 1 + (float(row["urban"]) + float(row["nightlife"]) - 5) * 0.025
    total_usd = DAILY_USD_BY_TIER.get(str(row["budget_level"]), 180.0) * days * max(0.85, urban_adjustment)
    return total_usd / rates[currency], total_usd, days


def expand_travel_styles(style_preferences: dict[str, int]) -> dict[str, float]:
    """Convert the six traveller controls into the dataset's nine theme scores."""
    scores, weights = ({theme: 0.0 for theme in INTERESTS} for _ in range(2))
    for style, score in style_preferences.items():
        for theme, weight in TRAVEL_STYLES[style].items():
            scores[theme] += score * weight
            weights[theme] += weight
    return {theme: scores[theme] / weights[theme] if weights[theme] else 0.0 for theme in INTERESTS}


def run_recommendations(data: pd.DataFrame, preferences: dict[str, float], budget: str, duration: str, region: str, country: str) -> pd.DataFrame:
    """Filter choices, scale interest columns, then query a Euclidean KNN model."""
    candidates = data.copy()
    if budget != "Any budget":
        candidates = candidates[candidates["budget_level"] == budget]
    if duration != "Any number of days":
        candidates = candidates[candidates["ideal_durations"].map(lambda tags: duration in parse_duration_tags(tags))]
    if region != "Any region":
        candidates = candidates[candidates["region"] == region]
    if country != "Anywhere":
        candidates = candidates[candidates["country"] == country]
    if candidates.empty:
        return candidates.assign(distance=pd.Series(dtype=float))

    # StandardScaler makes every interest dimension comparable before KNN measures distance.
    scaler = StandardScaler()
    matrix = scaler.fit_transform(candidates[INTERESTS])
    traveller = scaler.transform(pd.DataFrame([[preferences[theme] for theme in INTERESTS]], columns=INTERESTS))
    model = NearestNeighbors(n_neighbors=min(5, len(candidates)), metric="euclidean").fit(matrix)
    distances, positions = model.kneighbors(traveller)
    ranked = candidates.iloc[positions[0]].copy()
    ranked["distance"] = distances[0]
    return ranked.reset_index(drop=True)


def favourite_rows(cities: pd.DataFrame) -> pd.DataFrame:
    ids = st.session_state.get("favourite_ids", [])
    if not ids:
        return cities.iloc[0:0].copy()
    frame = cities[cities["id"].isin(ids)].copy()
    frame["_order"] = frame["id"].map({key: pos for pos, key in enumerate(ids)})
    return frame.sort_values("_order").drop(columns="_order")


def build_favourites_pdf(favourites: pd.DataFrame, currency: str, rates: dict[str, float], duration: str) -> bytes:
    """Build a concise PDF that the browser can download when ReportLab is installed."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm)
    styles = getSampleStyleSheet()
    story = [Paragraph("Wayfinder saved destinations", styles["Title"]), Spacer(1, 5 * mm)]
    rows = [["Destination", "Suggested stay", "Planning estimate"]]
    for _, place in favourites.iterrows():
        estimate, _, _ = estimated_cost(place, duration, currency, rates)
        stays = ", ".join(DURATION_LABELS.get(tag, tag) for tag in parse_duration_tags(place["ideal_durations"]))
        rows.append([f"{place['city']}, {place['country']}", stays, format_money(estimate, currency)])
    table = Table(rows, colWidths=[72 * mm, 50 * mm, 45 * mm])
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#176256")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#dce9e1")), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("PADDING", (0, 0), (-1, -1), 7)]))
    story.extend([table, Spacer(1, 5 * mm), Paragraph("Planning estimates use the destination budget category and suggested duration. They are not a live quote or booking price.", styles["BodyText"])])
    document.build(story)
    return buffer.getvalue()


def render_result(row: pd.Series, rank: int, currency: str, rates: dict[str, float], duration: str) -> None:
    city, country = html.escape(str(row["city"])), html.escape(str(row["country"]))
    description = html.escape(str(row["short_description"]))
    region = html.escape(REGION_LABELS.get(str(row["region"]), str(row["region"]).title()))
    stays = " · ".join(DURATION_LABELS.get(tag, tag) for tag in parse_duration_tags(row["ideal_durations"]))
    coordinates = f"{float(row['latitude']):.6f},{float(row['longitude']):.6f}"
    maps_url = f"https://www.google.com/maps/search/?api=1&query={quote_plus(coordinates)}"
    details_url = f"https://www.google.com/search?q={quote_plus(f'{row['city']} {row['country']} travel')}"
    estimate, estimate_usd, days = estimated_cost(row, duration, currency, rates)
    saved = str(row["id"]) in st.session_state.favourite_ids

    with st.container(border=True):
        image_column, content_column = st.columns([1.15, 4.85])
        with image_column:
            strongest_theme = max(INTERESTS, key=lambda theme: float(row[theme]))
            st.markdown(f"<div class='destination-photo' role='img' aria-label='{city}' style=\"background-image:url('{THEME_IMAGES[strongest_theme]}')\"></div>", unsafe_allow_html=True)
        with content_column:
            st.markdown(f"<div class='result-kicker'>MATCH {rank:02d} · {region}</div><div class='destination-name'>{city}</div><div class='destination-country'>{country}</div>", unsafe_allow_html=True)
            st.write(description)

        cost_column, days_column, distance_column = st.columns(3)
        cost_column.metric("Planning estimate", format_money(estimate, currency), help="A planning estimate from the source budget category and suggested stay; it is not a live price.")
        days_column.metric("Suggested days", f"{days:.0f} days")
        distance_column.metric("Match distance", f"{float(row['distance']):.2f}", help="A smaller distance means a closer match to your selected interests.")
        map_column, detail_column, favourite_column, _ = st.columns([1.2, 1.15, 1.05, 2.1])
        map_column.link_button("Google Maps", maps_url, use_container_width=True)
        detail_column.link_button("More details", details_url, use_container_width=True)
        if favourite_column.button("Saved" if saved else "Save", key=f"save_{row['id']}", use_container_width=True, disabled=saved):
            st.session_state.favourite_ids.append(str(row["id"]))
            st.rerun()
        with st.expander("Cost breakdown"):
            pieces = {"Flights allowance": estimate * 0.30, "Hotel / stay": estimate * 0.35, "Food": estimate * 0.20, "Activities & local travel": estimate * 0.15}
            columns = st.columns(4)
            for column, (label, value) in zip(columns, pieces.items()):
                column.metric(label, format_money(value, currency))
            st.caption(f"Estimated total: {format_money(estimate, currency)} (about USD {estimate_usd:,.0f}). Flights are a generic allowance because no origin or travel dates were entered.")


def render_weather(row: pd.Series) -> None:
    weather = get_weather(float(row["latitude"]), float(row["longitude"]))
    if not weather:
        st.info("Live weather is temporarily unavailable. Please try again shortly.")
        return
    current, daily = weather.get("current", {}), weather.get("daily", {})
    st.markdown(f"#### {row['city']}, {row['country']}")
    first, second, third = st.columns(3)
    first.metric("Now", f"{float(current.get('temperature_2m', 0)):.0f}°C")
    second.metric("Feels like", f"{float(current.get('apparent_temperature', 0)):.0f}°C")
    second.caption(WEATHER_LABELS.get(int(current.get("weather_code", -1)), "Conditions unavailable"))
    dates = daily.get("time", [])
    highs, lows, codes = daily.get("temperature_2m_max", []), daily.get("temperature_2m_min", []), daily.get("weather_code", [])
    forecast = []
    for day, high, low, code in zip(dates, highs, lows, codes):
        forecast.append({"Date": day, "Forecast": WEATHER_LABELS.get(int(code), "Conditions unavailable"), "High": f"{float(high):.0f}°C", "Low": f"{float(low):.0f}°C"})
    third.metric("3-day forecast", "Available")
    st.dataframe(pd.DataFrame(forecast), hide_index=True, use_container_width=True)
    st.caption("Weather data: Open-Meteo. Forecasts are live estimates and can change.")


def render_favourites(cities: pd.DataFrame, currency: str, rates: dict[str, float], duration: str) -> None:
    favourites = favourite_rows(cities)
    st.markdown("### Saved favourites")
    if favourites.empty:
        st.caption("Save destinations from your recommendations to compare or export them.")
        return
    export = favourites[["city", "country", "region", "budget_level", "ideal_durations"]].copy()
    export.columns = ["Destination", "Country", "Region", "Dataset budget tier", "Suggested stay"]
    estimates = [estimated_cost(row, duration, currency, rates)[0] for _, row in favourites.iterrows()]
    export["Planning estimate"] = [format_money(value, currency) for value in estimates]
    st.dataframe(export, hide_index=True, use_container_width=True)
    csv_column, pdf_column, clear_column, _ = st.columns([1.1, 1.1, 0.9, 2.9])
    csv_column.download_button("Download CSV", export.to_csv(index=False).encode("utf-8"), "wayfinder-favourites.csv", "text/csv", use_container_width=True)
    try:
        pdf_bytes = build_favourites_pdf(favourites, currency, rates, duration)
        pdf_column.download_button("Download PDF", pdf_bytes, "wayfinder-favourites.pdf", "application/pdf", use_container_width=True)
    except ModuleNotFoundError:
        pdf_column.caption("Install ReportLab to export PDF.")
    if clear_column.button("Clear saved", use_container_width=True):
        st.session_state.favourite_ids = []
        st.rerun()


def render_comparison(cities: pd.DataFrame, results: pd.DataFrame, currency: str, rates: dict[str, float], duration: str) -> None:
    pool = pd.concat([results, favourite_rows(cities)]).drop_duplicates(subset="id")
    if len(pool) < 2:
        return
    labels = {str(row["id"]): f"{row['city']}, {row['country']}" for _, row in pool.iterrows()}
    selected = st.multiselect("Choose 2 or 3 places", list(labels), format_func=lambda item: labels[item], max_selections=3, key="comparison_ids")
    if len(selected) < 2:
        return
    compared = pool[pool["id"].astype(str).isin(selected)].set_index("id").loc[selected].reset_index()
    columns = st.columns(len(compared))
    for column, (_, place) in zip(columns, compared.iterrows()):
        estimate, _, days = estimated_cost(place, duration, currency, rates)
        with column:
            st.markdown(f"#### {place['city']}")
            st.caption(str(place["country"]))
            st.metric("Planning estimate", format_money(estimate, currency))
            st.metric("Suggested stay", f"{days:.0f} days")
            st.caption(f"Dataset category: {place['budget_level']}")


def install_shared_settings() -> None:
    """Read compact URL parameters before widgets are created, so shared links restore a plan."""
    defaults = {"budget_amount": 1000.0, "currency": "USD", "duration": "Any number of days", "region": "Any region", "country": "Anywhere"}
    query = st.query_params
    for key, default in defaults.items():
        if key in st.session_state:
            continue
        raw = query.get({"budget_amount": "b", "currency": "cur", "duration": "d", "region": "r", "country": "c"}[key])
        try:
            st.session_state[key] = float(raw) if key == "budget_amount" and raw else raw or default
        except (TypeError, ValueError):
            st.session_state[key] = default
    for index, style in enumerate(TRAVEL_STYLES):
        key = f"style_{index}"
        if key not in st.session_state:
            try:
                st.session_state[key] = max(0, min(5, int(query.get(f"i{index}", 3))))
            except (TypeError, ValueError):
                st.session_state[key] = 3


def update_share_link() -> None:
    parameters = {"b": str(st.session_state.budget_amount), "cur": st.session_state.currency, "d": st.session_state.duration, "r": st.session_state.region, "c": st.session_state.country}
    parameters.update({f"i{index}": str(st.session_state[f'style_{index}']) for index in range(len(TRAVEL_STYLES))})
    st.query_params.clear()
    st.query_params.update(parameters)


try:
    cities = load_city_data()
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not load the travel dataset: {error}")
    st.stop()

for key, value in {"favourite_ids": []}.items():
    if key not in st.session_state:
        st.session_state[key] = value
install_shared_settings()
rates, rate_date, live_rates = get_live_usd_rates()

palette = {"ink": "#17352f", "muted": "#637970", "green": "#176256", "line": "#dce9e1", "surface": "rgba(255, 255, 255, .88)", "metric": "#f5f9f6", "field": "#ffffff", "background": "radial-gradient(circle at 8% 0%,#dcefe4 0,transparent 25%),linear-gradient(180deg,#fbfdfb 0%,#f0f6f2 100%)", "shadow": "rgba(25, 76, 59, .08)"}
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
:root {{ --ink:{palette['ink']}; --muted:{palette['muted']}; --green:{palette['green']}; --line:{palette['line']}; --surface:{palette['surface']}; --metric:{palette['metric']}; --field:{palette['field']}; --shadow:{palette['shadow']}; }}
.stApp {{ background:{palette['background']} !important; color:var(--ink) !important; font-family:'DM Sans',sans-serif; font-size:16px; }}
.block-container {{ max-width:1040px; padding:1.1rem 1.25rem 4rem; }}
h1,h2,h3,p,label,[data-testid="stWidgetLabel"], [data-testid="stCaptionContainer"] {{ color:var(--ink) !important; }}
h1,h2,h3 {{ font-family:'Manrope',sans-serif; letter-spacing:-.04em; }}
.hero {{ min-height:255px; display:flex; align-items:flex-end; padding:2.2rem; border-radius:28px; background-image:linear-gradient(90deg,rgba(10,47,46,.96) 0%,rgba(10,47,46,.7) 48%,rgba(10,47,46,.08) 100%),url('https://images.unsplash.com/photo-1500534623283-312aade485b7?auto=format&fit=crop&w=1800&q=90'); background-size:cover; background-position:center; box-shadow:0 22px 52px var(--shadow); overflow:hidden; }}
.hero-kicker {{ color:#b9d8cb; font-size:.72rem; font-weight:800; letter-spacing:.17em; text-transform:uppercase; }} .hero-title {{ color:#fff; font-family:'Manrope',sans-serif; font-size:clamp(2.25rem,5vw,3.85rem); font-weight:800; line-height:1.03; letter-spacing:-.06em; margin:.45rem 0 .6rem; max-width:610px; }} .hero-copy {{ color:#e2eee8; max-width:575px; }}
.planner-label {{ font-family:'Manrope',sans-serif; font-weight:800; font-size:1.45rem; margin:1.7rem 0 .15rem; }} .planner-copy,.destination-country,.result-kicker {{ color:var(--muted) !important; }}
div[data-testid="stVerticalBlockBorderWrapper"] {{ background:var(--surface) !important; border:1px solid var(--line) !important; border-radius:21px; box-shadow:0 12px 32px var(--shadow); }} [data-testid="stVerticalBlockBorderWrapper"] > div {{ padding:.45rem; }}
div.stButton > button, div[data-testid="stLinkButton"] a {{ border-radius:999px; font-weight:800; }} div.stButton > button[kind="primary"] {{ background:var(--green); border-color:var(--green); }}
[data-testid="stMetric"] {{ background:var(--metric) !important; border:1px solid var(--line); border-radius:14px; padding:.7rem .95rem; }} [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{ color:var(--ink) !important; }}
div[data-baseweb="select"] > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input {{ background:var(--field) !important; color:var(--ink) !important; border-color:var(--line) !important; }} [data-testid="stNumberInput"] button {{ background:var(--field) !important; color:var(--ink) !important; }}
.destination-photo {{ min-height:138px; height:100%; border-radius:15px; background-position:center; background-size:cover; }} .destination-name {{ font-family:'Manrope',sans-serif; font-size:1.65rem; font-weight:800; letter-spacing:-.045em; line-height:1.08; color:var(--ink); margin-top:.25rem; }}
@media (max-width:700px) {{ .hero {{ min-height:235px; padding:1.5rem; }} .destination-photo {{ min-height:105px; }} }}
</style>""", unsafe_allow_html=True)

st.markdown("<section class='hero'><div><div class='hero-kicker'>Personalised travel matches</div><div class='hero-title'>Find your next destination.</div><div class='hero-copy'>Set the essentials, then get five city matches based on your travel style.</div></div></section>", unsafe_allow_html=True)
st.markdown("<div class='planner-label'>Plan your trip</div><div class='planner-copy'>Budget, days, location, and the travel experiences you enjoy.</div>", unsafe_allow_html=True)

with st.container(border=True):
    budget_column, currency_column = st.columns(2)
    with budget_column:
        budget_amount = st.number_input("Total trip budget", min_value=1.0, step=100.0, key="budget_amount")
    with currency_column:
        selected_currency = st.selectbox("Currency", list(FALLBACK_USD_PER_CURRENCY), key="currency")
    selected_budget, budget_in_usd = budget_tier(budget_amount, selected_currency, rates)
    rate_note = f"Live exchange rate dated {rate_date}" if live_rates else "Exchange service unavailable; using an offline reference rate"
    st.caption(f"{rate_note}. Your entered total is about USD {budget_in_usd:,.0f}; it filters the dataset's **{selected_budget}** category.")

    days_column, region_column = st.columns(2)
    with days_column:
        available_durations = {item for value in cities["ideal_durations"] for item in parse_duration_tags(value)}
        duration_options = ["Any number of days"] + [tag for tag in DURATION_LABELS if tag in available_durations]
        if st.session_state.duration not in duration_options:
            st.session_state.duration = "Any number of days"
        selected_duration = st.selectbox("Days", duration_options, format_func=lambda value: DURATION_LABELS.get(value, value), key="duration")
    with region_column:
        region_options = ["Any region"] + list(REGION_LABELS)
        if st.session_state.region not in region_options:
            st.session_state.region = "Any region"
        selected_region = st.selectbox("Region", region_options, format_func=lambda value: REGION_LABELS.get(value, value), key="region")
    country_data = cities if selected_region == "Any region" else cities[cities["region"] == selected_region]
    country_options = ["Anywhere"] + sorted(country_data["country"].unique().tolist())
    if st.session_state.country not in country_options:
        st.session_state.country = "Anywhere"
    selected_country = st.selectbox("Country", country_options, key="country")

    st.markdown("#### Interests")
    columns = st.columns(3)
    preferences: dict[str, int] = {}
    for index, style in enumerate(TRAVEL_STYLES):
        with columns[index % 3]:
            preferences[style] = st.slider(style, 0, 5, key=f"style_{index}")
    submit_column, share_column, _ = st.columns([1.45, 1.45, 3.1])
    submitted = submit_column.button("Find my top 5  →", type="primary", use_container_width=True)
    if share_column.button("Create share link", use_container_width=True):
        update_share_link()
        st.toast("Plan settings were added to the browser address bar. Copy that URL to share this plan.")

if submitted:
    st.session_state.recommendations = run_recommendations(cities, expand_travel_styles(preferences), selected_budget, selected_duration, selected_region, selected_country).to_dict(orient="records")

if st.session_state.get("recommendations"):
    results = pd.DataFrame(st.session_state.recommendations)
    st.markdown("### Your top matches")
    st.caption("Results are ranked by standardised KNN distance. Smaller is closer. Planning estimates are shown in your selected currency.")
    for rank, (_, destination) in enumerate(results.iterrows(), start=1):
        render_result(destination, rank, selected_currency, rates, selected_duration)
    st.markdown("### Recommended destinations map")
    st.map(results[["latitude", "longitude"]].rename(columns={"latitude": "lat", "longitude": "lon"}), height=360, use_container_width=True)
    with st.expander("Live weather forecast"):
        selected_weather_city = st.selectbox("Destination", list(results["id"].astype(str)), format_func=lambda item: f"{results.loc[results['id'].astype(str).eq(item), 'city'].iloc[0]}, {results.loc[results['id'].astype(str).eq(item), 'country'].iloc[0]}")
        render_weather(results.loc[results["id"].astype(str).eq(selected_weather_city)].iloc[0])
    with st.expander("Compare destinations"):
        render_comparison(cities, results, selected_currency, rates, selected_duration)
    with st.expander("Saved favourites and exports"):
        render_favourites(cities, selected_currency, rates, selected_duration)
elif submitted:
    st.warning("No destinations match those choices. Try a broader budget, days, region, or country.")
else:
    with st.expander("Saved favourites and exports"):
        render_favourites(cities, selected_currency, rates, selected_duration)

st.caption("Exchange rates: Frankfurter (live when available). Weather: Open-Meteo. Destination profiles are dataset records. Cost estimates are planning aids, not booking quotes.")
