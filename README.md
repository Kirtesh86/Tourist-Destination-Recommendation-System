# Wayfinder — Travel Destination Recommendation System

Wayfinder is a simple Streamlit travel planner that recommends five cities from a traveler's budget, trip length, location, and travel styles.

## Project objective

Help travelers discover places that fit their interests and practical preferences. It is a discovery planner, not a booking service.

## Dataset

The primary file, `worldwide_travel_cities.csv`, is the Kaggle **Worldwide Travel Cities (Ratings and Climate)** dataset. It contains 560 city profiles in 167 countries and seven regions.

See [DATA_SOURCES.md](DATA_SOURCES.md) for dataset comparison, attribution, and limitations.

## Features

- Six adjustable travel styles: Cultural Immersion, Nature and Wildlife, Adventure and Thrill Sports, Food and Culinary Tourism, Urban Exploration, and Wellness and Slow Travel.
- Budget amount with currency selection: USD, INR, EUR, GBP, AED, AUD, CAD, or JPY.
- Days, region, and country selectors.
- An interactive map, direct Google Maps links, and destination-detail links.
- Live exchange rates from Frankfurter, with a cached fallback when the service is unavailable.
- Currency planning estimates and a flight, stay, food, and activity cost split for every match.
- Current conditions and a three-day forecast from Open-Meteo.
- Saved favourites, CSV export, and PDF export.
- A two or three destination comparison tool.
- Shareable planner settings in the browser URL.
- Light/dark mode plus larger-text and reduced-motion accessibility controls.
- Five nearest city profiles ranked with scaled KNN distance.

## How KNN recommendations work

The dataset provides city ratings from 0 to 5 for nine travel themes. The app:

1. Loads the CSV with Pandas and selects the source's nine theme scores.
2. Converts the six travel-style sliders into a weighted nine-theme traveller profile.
3. Uses `StandardScaler` to scale the source scores before distance calculations.
4. Applies the selected budget, days, region, and country filters.
5. Fits scikit-learn's `NearestNeighbors` model to the filtered cities and queries it with the user's interest profile.
6. Shows up to five nearest destinations. A lower Euclidean distance means the city profile is closer to the selected interests.

The entered budget is converted to USD with a live exchange rate and mapped to the source's `Budget`, `Mid-range`, or `Luxury` category for filtering. Recommendation cards show a planning estimate in the selected currency. The underlying dataset supplies only a category, so that estimate is calculated from the category and suggested trip duration. It is not a live hotel, flight, or booking quote.

## Live-data services

- **Exchange rates:** [Frankfurter](https://frankfurter.dev/), cached for one hour.
- **Weather:** [Open-Meteo](https://open-meteo.com/), cached for 15 minutes.

These services can be temporarily unavailable. The app continues to recommend destinations; exchange conversion then uses a clearly labelled offline fallback and weather shows an availability message.

## Install dependencies

Use Python 3.10 or newer. From the project directory, install the packages:

```bash
python -m pip install -r requirements.txt
```

## Run the project

```bash
streamlit run app.py
```

Streamlit opens Wayfinder in your browser. Enter a budget and currency, select days, region, country, and travel styles, then choose **Find my top 5**.

## Project files

```text
app.py
worldwide_travel_cities.csv
requirements.txt
DATA_SOURCES.md
README.md
```
