"""
fetch_real_attractions_osm.py

Purpose:
    Fetch real-world POI data from OpenStreetMap Overpass API and convert it
    into the same schema used by the existing travel recommendation project.

Output:
    data/attractions.csv

Notes:
    - This script replaces the simulated attraction generator step.
    - It does NOT require an API key.
    - OSM does not provide complete ratings, ticket status, or true popularity.
      Therefore, popularity and some preference scores are estimated from OSM tags.
"""

import os
import time
import math
import argparse
from typing import Dict, List, Any

import requests
import numpy as np
import pandas as pd


OVERPASS_URL = "https://overpass-api.de/api/interpreter"

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "attractions.csv")


# =========================
# Feature schema expected by the original project
# =========================

PROJECT_COLUMNS = [
    "attraction_id",
    "name",
    "realistic_scene_name",
    "attraction_type",
    "latitude",
    "longitude",
    "osm_id",
    "osm_category",
    "cost_level",
    "indoor_score",
    "photo_value",
    "nature_value",
    "culture_value",
    "food_value",
    "popularity",
    "estimated_time",
]


# =========================
# Category mapping
# =========================
# These are heuristic mappings from OSM tags to your project features.
# Scores are in the same 0~5 scale as your simulated attraction data.

CATEGORY_PROFILES: Dict[str, Dict[str, float]] = {
    "museum": {
        "indoor_score": 4.6,
        "cost_level": 2.6,
        "photo_value": 3.7,
        "nature_value": 0.4,
        "culture_value": 4.8,
        "food_value": 1.0,
        "popularity": 3.2,
        "estimated_time": 2.3,
    },
    "gallery": {
        "indoor_score": 4.5,
        "cost_level": 2.3,
        "photo_value": 4.0,
        "nature_value": 0.3,
        "culture_value": 4.5,
        "food_value": 1.0,
        "popularity": 2.8,
        "estimated_time": 1.8,
    },
    "park": {
        "indoor_score": 0.4,
        "cost_level": 0.6,
        "photo_value": 3.6,
        "nature_value": 4.5,
        "culture_value": 1.0,
        "food_value": 1.2,
        "popularity": 2.8,
        "estimated_time": 2.5,
    },
    "viewpoint": {
        "indoor_score": 0.3,
        "cost_level": 0.8,
        "photo_value": 4.8,
        "nature_value": 3.8,
        "culture_value": 1.0,
        "food_value": 0.8,
        "popularity": 3.0,
        "estimated_time": 1.5,
    },
    "attraction": {
        "indoor_score": 2.0,
        "cost_level": 2.0,
        "photo_value": 3.8,
        "nature_value": 2.0,
        "culture_value": 2.5,
        "food_value": 1.5,
        "popularity": 3.2,
        "estimated_time": 2.0,
    },
    "aquarium": {
        "indoor_score": 4.2,
        "cost_level": 2.8,
        "photo_value": 3.6,
        "nature_value": 3.2,
        "culture_value": 2.0,
        "food_value": 1.0,
        "popularity": 3.4,
        "estimated_time": 1.8,
    },
    "zoo": {
        "indoor_score": 0.8,
        "cost_level": 2.8,
        "photo_value": 4.0,
        "nature_value": 4.2,
        "culture_value": 1.5,
        "food_value": 1.5,
        "popularity": 4.0,
        "estimated_time": 4.0,
    },
    "historic": {
        "indoor_score": 1.5,
        "cost_level": 0.8,
        "photo_value": 3.8,
        "nature_value": 1.2,
        "culture_value": 4.8,
        "food_value": 1.0,
        "popularity": 3.0,
        "estimated_time": 1.5,
    },
    "monument": {
        "indoor_score": 0.8,
        "cost_level": 0.5,
        "photo_value": 4.0,
        "nature_value": 1.5,
        "culture_value": 4.4,
        "food_value": 0.6,
        "popularity": 3.0,
        "estimated_time": 1.0,
    },
    "garden": {
        "indoor_score": 0.5,
        "cost_level": 0.8,
        "photo_value": 4.0,
        "nature_value": 4.6,
        "culture_value": 1.2,
        "food_value": 0.8,
        "popularity": 2.8,
        "estimated_time": 2.0,
    },
    "library": {
        "indoor_score": 4.8,
        "cost_level": 0.3,
        "photo_value": 2.0,
        "nature_value": 0.2,
        "culture_value": 3.8,
        "food_value": 0.5,
        "popularity": 2.0,
        "estimated_time": 1.5,
    },
    "theatre": {
        "indoor_score": 4.8,
        "cost_level": 3.0,
        "photo_value": 2.8,
        "nature_value": 0.2,
        "culture_value": 4.3,
        "food_value": 1.2,
        "popularity": 3.0,
        "estimated_time": 2.5,
    },
    "cinema": {
        "indoor_score": 5.0,
        "cost_level": 2.5,
        "photo_value": 1.6,
        "nature_value": 0.0,
        "culture_value": 2.0,
        "food_value": 2.5,
        "popularity": 3.2,
        "estimated_time": 2.3,
    },
    "theme_park": {
        "indoor_score": 1.8,
        "cost_level": 4.0,
        "photo_value": 4.0,
        "nature_value": 1.0,
        "culture_value": 1.2,
        "food_value": 3.0,
        "popularity": 4.2,
        "estimated_time": 5.5,
    },
    "temple": {
        "indoor_score": 2.5,
        "cost_level": 0.8,
        "photo_value": 3.4,
        "nature_value": 1.2,
        "culture_value": 4.6,
        "food_value": 1.3,
        "popularity": 2.8,
        "estimated_time": 1.6,
    },
    "place_of_worship": {
        "indoor_score": 2.8,
        "cost_level": 0.8,
        "photo_value": 3.2,
        "nature_value": 1.0,
        "culture_value": 4.3,
        "food_value": 1.0,
        "popularity": 2.5,
        "estimated_time": 1.5,
    },
    "restaurant": {
        "indoor_score": 4.0,
        "cost_level": 2.8,
        "photo_value": 2.6,
        "nature_value": 0.3,
        "culture_value": 1.6,
        "food_value": 4.7,
        "popularity": 3.4,
        "estimated_time": 1.4,
    },
    "cafe": {
        "indoor_score": 4.2,
        "cost_level": 2.8,
        "photo_value": 4.0,
        "nature_value": 0.4,
        "culture_value": 2.0,
        "food_value": 4.0,
        "popularity": 3.2,
        "estimated_time": 1.3,
    },
    "marketplace": {
        "indoor_score": 1.8,
        "cost_level": 2.0,
        "photo_value": 3.0,
        "nature_value": 0.2,
        "culture_value": 2.8,
        "food_value": 4.5,
        "popularity": 4.0,
        "estimated_time": 2.0,
    },
    "mall": {
        "indoor_score": 4.8,
        "cost_level": 4.0,
        "photo_value": 2.8,
        "nature_value": 0.2,
        "culture_value": 1.2,
        "food_value": 3.6,
        "popularity": 4.0,
        "estimated_time": 3.0,
    },
}


# Human-readable category names for UI.
CATEGORY_DISPLAY_NAME = {
    "museum": "博物館",
    "gallery": "藝文展館",
    "park": "公園",
    "viewpoint": "觀景景點",
    "attraction": "觀光景點",
    "aquarium": "水族館",
    "zoo": "動物園",
    "historic": "歷史文化景點",
    "monument": "紀念碑／地標",
    "garden": "花園",
    "library": "圖書館",
    "theatre": "劇場",
    "cinema": "電影院",
    "theme_park": "主題樂園",
    "temple": "寺廟",
    "place_of_worship": "宗教文化景點",
    "restaurant": "餐廳",
    "cafe": "咖啡廳",
    "marketplace": "市集",
    "mall": "購物中心",
}


# Overpass query blocks. Keep the scope moderate to avoid heavy requests.
OVERPASS_FILTERS = [
    'node["tourism"="museum"](area.searchArea);',
    'way["tourism"="museum"](area.searchArea);',
    'relation["tourism"="museum"](area.searchArea);',

    'node["tourism"="gallery"](area.searchArea);',
    'way["tourism"="gallery"](area.searchArea);',
    'relation["tourism"="gallery"](area.searchArea);',

    'node["leisure"="park"](area.searchArea);',
    'way["leisure"="park"](area.searchArea);',
    'relation["leisure"="park"](area.searchArea);',

    'node["tourism"="viewpoint"](area.searchArea);',
    'way["tourism"="viewpoint"](area.searchArea);',

    'node["tourism"="attraction"](area.searchArea);',
    'way["tourism"="attraction"](area.searchArea);',
    'relation["tourism"="attraction"](area.searchArea);',

    'node["tourism"="theme_park"](area.searchArea);',
    'way["tourism"="theme_park"](area.searchArea);',

    'node["amenity"="place_of_worship"](area.searchArea);',
    'way["amenity"="place_of_worship"](area.searchArea);',
    'relation["amenity"="place_of_worship"](area.searchArea);',

    'node["amenity"="restaurant"](area.searchArea);',
    'node["amenity"="cafe"](area.searchArea);',
    'node["amenity"="marketplace"](area.searchArea);',

    'node["shop"="mall"](area.searchArea);',
    'way["shop"="mall"](area.searchArea);',
]


def clip_score(value: float, low: float = 0.0, high: float = 5.0) -> float:
    return float(np.clip(value, low, high))



def is_excluded_poi(tags: Dict[str, Any]) -> bool:
    """Return True for POIs that are not suitable as tourist-attraction recommendations."""
    name = " ".join(str(tags.get(k, "")) for k in ["name", "name:zh", "name:en"]).lower()
    tourism = str(tags.get("tourism", "")).lower()
    amenity = str(tags.get("amenity", "")).lower()
    shop = str(tags.get("shop", "")).lower()
    office = str(tags.get("office", "")).lower()

    excluded_tourism = {
        "hotel", "hostel", "guest_house", "motel", "apartment", "camp_site",
        "chalet", "caravan_site", "information",
    }
    excluded_amenity = {
        "parking", "toilets", "bank", "atm", "clinic", "hospital", "pharmacy",
        "police", "post_office", "fuel", "charging_station",
    }
    excluded_shop = {
        "convenience", "supermarket", "clothes", "hairdresser", "laundry",
        "car", "mobile_phone", "electronics",
    }
    excluded_name_keywords = [
        "hotel", "hostel", "backpacker", "backpackers", "guest house", "guesthouse",
        "motel", "inn", "apartment", "旅館", "飯店", "酒店", "背包客", "民宿",
        "停車場", "廁所", "公司", "辦公室", "分店", "門市",
    ]

    if tourism in excluded_tourism or amenity in excluded_amenity or shop in excluded_shop or office:
        return True
    return any(keyword in name for keyword in excluded_name_keywords)


def infer_category(tags: Dict[str, Any]) -> str:
    tourism = tags.get("tourism")
    amenity = tags.get("amenity")
    leisure = tags.get("leisure")
    shop = tags.get("shop")
    historic = tags.get("historic")
    religion = tags.get("religion")

    if is_excluded_poi(tags):
        return "__exclude__"

    if tourism in {"museum", "gallery", "viewpoint", "theme_park", "aquarium", "zoo"}:
        return tourism
    if leisure == "park":
        return "park"
    if leisure == "garden":
        return "garden"
    if historic:
        if historic == "monument":
            return "monument"
        return "historic"
    if amenity == "place_of_worship":
        if religion in {"buddhist", "taoist", "shinto"}:
            return "temple"
        return "place_of_worship"
    if amenity in {"restaurant", "cafe", "marketplace", "library", "theatre", "cinema"}:
        return amenity
    if shop == "mall":
        return "mall"

    # Do not silently turn unknown POIs into generic attraction.
    return "__exclude__"


def estimate_popularity(tags: Dict[str, Any], category: str) -> float:
    """Estimate popularity using available OSM metadata.

    OSM does not provide ratings or visit counts. This is a heuristic score.
    """
    base = CATEGORY_PROFILES[category]["popularity"]

    if tags.get("wikipedia") or tags.get("wikidata"):
        base += 0.7
    if tags.get("website"):
        base += 0.3
    if tags.get("name:en"):
        base += 0.2
    if tags.get("opening_hours"):
        base += 0.2

    return clip_score(base)


def adjust_scores_from_tags(profile: Dict[str, float], tags: Dict[str, Any]) -> Dict[str, float]:
    """Small rule-based adjustments from OSM tags."""
    score = profile.copy()

    fee = str(tags.get("fee", "")).lower()
    if fee in {"yes", "true", "1"}:
        score["cost_level"] = clip_score(score["cost_level"] + 0.8)
    elif fee in {"no", "false", "0"}:
        score["cost_level"] = clip_score(score["cost_level"] - 0.6)

    indoor = str(tags.get("indoor", "")).lower()
    if indoor in {"yes", "true", "1"}:
        score["indoor_score"] = clip_score(score["indoor_score"] + 0.8)

    if tags.get("historic"):
        score["culture_value"] = clip_score(score["culture_value"] + 0.8)
        score["photo_value"] = clip_score(score["photo_value"] + 0.3)

    if tags.get("natural"):
        score["nature_value"] = clip_score(score["nature_value"] + 1.0)
        score["indoor_score"] = clip_score(score["indoor_score"] - 0.8)

    return score


def get_coordinates(element: Dict[str, Any]) -> tuple:
    """Return latitude and longitude for node/way/relation.

    For ways and relations, Overpass can return center coordinates when using out center.
    """
    if "lat" in element and "lon" in element:
        return element["lat"], element["lon"]

    center = element.get("center")
    if center:
        return center.get("lat"), center.get("lon")

    return None, None


def build_overpass_query(area_name: str, limit: int) -> str:
    filters = "\n".join(OVERPASS_FILTERS)
    return f"""
[out:json][timeout:60];
area["name"="{area_name}"]->.searchArea;
(
{filters}
);
out center {limit};
"""


def fetch_osm_pois(area_name: str = "臺北市", limit: int = 300, sleep_sec: float = 1.0) -> List[Dict[str, Any]]:
    query = build_overpass_query(area_name=area_name, limit=limit)

    print(f"Fetching POIs from OpenStreetMap Overpass API: {area_name}")

    response = requests.post(
        OVERPASS_URL,
        data={"data": query},
        timeout=90,
        headers={"User-Agent": "travel-clustering-project/1.0"},
    )
    response.raise_for_status()

    time.sleep(sleep_sec)

    payload = response.json()
    return payload.get("elements", [])


def elements_to_attractions(elements: List[Dict[str, Any]], max_items: int = 300) -> pd.DataFrame:
    rows = []
    seen_osm_ids = set()

    for element in elements:
        tags = element.get("tags", {})
        lat, lon = get_coordinates(element)

        if lat is None or lon is None:
            continue

        osm_id = f"{element.get('type')}/{element.get('id')}"
        if osm_id in seen_osm_ids:
            continue
        seen_osm_ids.add(osm_id)

        name = tags.get("name") or tags.get("name:zh") or tags.get("name:en")
        if not name:
            continue

        category = infer_category(tags)
        if category == "__exclude__":
            continue
        if category not in CATEGORY_PROFILES:
            continue

        profile = adjust_scores_from_tags(CATEGORY_PROFILES[category], tags)
        popularity = estimate_popularity(tags, category)

        attraction_id = len(rows) + 1
        display_category = CATEGORY_DISPLAY_NAME.get(category, category)

        rows.append({
            "attraction_id": attraction_id,
            "name": f"OSM_{attraction_id:03d}_{category}",
            "realistic_scene_name": name,
            "attraction_type": category,
            "latitude": float(lat),
            "longitude": float(lon),
            "osm_id": osm_id,
            "osm_category": display_category,
            "cost_level": clip_score(profile["cost_level"]),
            "indoor_score": clip_score(profile["indoor_score"]),
            "photo_value": clip_score(profile["photo_value"]),
            "nature_value": clip_score(profile["nature_value"]),
            "culture_value": clip_score(profile["culture_value"]),
            "food_value": clip_score(profile["food_value"]),
            "popularity": popularity,
            "estimated_time": float(profile["estimated_time"]),
        })

        if len(rows) >= max_items:
            break

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError("No usable OSM POIs were returned. Try a different area name or smaller scope.")

    return df[PROJECT_COLUMNS]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def add_distance_from_center(df: pd.DataFrame, center_lat: float, center_lon: float) -> pd.DataFrame:
    df = df.copy()
    df["distance_from_center_km"] = df.apply(
        lambda row: haversine_km(center_lat, center_lon, row["latitude"], row["longitude"]),
        axis=1,
    )
    return df


def fetch_open_meteo_weather(latitude: float, longitude: float) -> Dict[str, float]:
    """Fetch current weather variables from Open-Meteo.

    This is optional metadata. The current recommendation pipeline does not require it.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "Asia/Taipei",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    current = response.json().get("current", {})

    return {
        "current_temperature_2m": current.get("temperature_2m", np.nan),
        "current_precipitation": current.get("precipitation", np.nan),
        "current_wind_speed_10m": current.get("wind_speed_10m", np.nan),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", type=str, default="臺北市", help="OSM area name, e.g., 臺北市, 台北市, 新北市")
    parser.add_argument("--max-items", type=int, default=300, help="Maximum number of attractions to save")
    parser.add_argument("--center-lat", type=float, default=25.0330, help="Center latitude for distance calculation")
    parser.add_argument("--center-lon", type=float, default=121.5654, help="Center longitude for distance calculation")
    parser.add_argument("--add-weather", action="store_true", help="Add current city-level Open-Meteo weather columns")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    elements = fetch_osm_pois(area_name=args.area, limit=args.max_items * 3)
    df = elements_to_attractions(elements, max_items=args.max_items)
    df = add_distance_from_center(df, args.center_lat, args.center_lon)

    if args.add_weather:
        weather = fetch_open_meteo_weather(args.center_lat, args.center_lon)
        for key, value in weather.items():
            df[key] = value

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Attraction count: {len(df)}")
    print(df.head(10))


if __name__ == "__main__":
    main()
