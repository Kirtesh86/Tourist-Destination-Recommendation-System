# Data sources, comparison, and limitations

## Primary recommendation data: Worldwide Travel Cities

The app uses [Worldwide Travel Cities (Ratings and Climate)](https://www.kaggle.com/datasets/furkanima/worldwide-travel-cities-ratings-and-climate), published by Furkan on Kaggle. Kaggle's public dataset metadata identifies the license as **MIT**, describes the file as curated travel information, and reports 560 cities. The included CSV has 560 records across 167 countries.

The CSV includes city and country names, region, coordinates, a short description, nine thematic ratings, ideal-duration tags, a budget tier, and monthly average/minimum/maximum temperatures. The themes are culture, adventure, nature, beaches, nightlife, cuisine, wellness, urban, and seclusion.

The city ratings, descriptions, budget tiers, ideal durations, and climate values are publisher-provided curated data. They are not independently verified visitor reviews, live weather readings, forecasts, current prices, or guaranteed itineraries. The app displays the budget tier and suggested durations as source categories, not as estimates of spend or trip plans.

**Attribution:** Furkan, “Worldwide Travel Cities (Ratings and Climate),” Kaggle, MIT License. See the [dataset page](https://www.kaggle.com/datasets/furkanima/worldwide-travel-cities-ratings-and-climate).

## Dataset comparison

| Source | Coverage and fit | Decision |
| --- | --- | --- |
| [Worldwide Travel Cities (Ratings and Climate)](https://www.kaggle.com/datasets/furkanima/worldwide-travel-cities-ratings-and-climate) | 560 city profiles across 167 countries, with nine interest ratings, budget tiers, suggested trip lengths, coordinates, and monthly climate fields. These fields directly support the planner and KNN matching. | **Selected** as the primary recommender dataset. Its scores and climate fields remain publisher-provided and should be treated as curated profile data. |
| [Travel Recommendation Dataset](https://www.kaggle.com/datasets/amanmehra23/travel-recommendation-dataset) | 1,000 rows but only five distinct destination names in the comparison performed for this project. The dataset is India-only and its license statements were inconsistent. | Not selected: limited destination diversity and unclear reuse terms. |
| [UNESCO World Heritage List on DataHub](https://data.unesco.org/explore/dataset/whc001/information/?flg=en-us) | Official heritage records. The separate Indian catalog contains 137 component locations across 45 inscribed properties, with categories, descriptions, coordinates, and source links. | Kept as a separate reference catalog; not used as scored travel-preference data. |

## UNESCO attribution and refresh

The UNESCO catalog was assembled from the [World Heritage List dataset on DataHub](https://data.unesco.org/explore/dataset/whc001/information/?flg=en-us), licensed **CC BY-SA 4.0**. The CSV includes the source URL and license fields. UNESCO component coordinates support map browsing; official property pages are linked in the app.

To refresh the catalog from the public DataHub API:

```bash
python scripts/update_unesco_india_sites.py
```

The script writes `unesco_india_sites.csv`. A refresh requires internet access.

## Model and interpretation

KNN compares a traveler's nine selected theme scores with city profiles. `StandardScaler` standardizes the scores before Euclidean distances are measured. Filters limit the candidate cities before the nearest-neighbor search. The displayed KNN distance is a relative feature-space distance; it is not a probability or a quality rating.

## Limitations

- City profiles are curated by the Kaggle publisher; the app does not verify them against visitor reviews or a live travel API.
- Monthly temperatures are source values, not current conditions or forecasts.
- Budget is a tier (`Budget`, `Mid-range`, or `Luxury`), not a trip-cost estimate.
- Ideal durations are source tags and are not a personalized itinerary.
- The saved shortlist exists only for the current Streamlit browser session.
- UNESCO records describe heritage properties; the UNESCO catalog does not produce the KNN recommendations.
