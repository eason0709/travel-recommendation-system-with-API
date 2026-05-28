"""
fetch_real_attractions_osm.py

Purpose:
    Fetch real-world POI data from OpenStreetMap Overpass API and convert it
    into the schema used by the travel recommendation project.

What this expanded version changes:
    1. Queries many more OSM POI categories.
    2. Does not stop after early categories fill the quota.
       It queries all groups first, then samples items by category so the output
       is not dominated by museums / galleries / parks.
    3. Adds more real-world metadata columns such as address, website,
       opening hours, phone, wikipedia, wikidata, fee, wheelchair, etc.
    4. Keeps the original required recommendation columns, so the existing
       downstream pipeline can still run.

Output:
    data/attractions.csv

Notes:
    - OSM does not provide complete ratings, ticket status, or true popularity.
      Therefore popularity and some preference scores are estimated from tags.
    - Some metadata columns may be blank because OSM tags are community-entered.
"""

import os
import re
import time
import math
import json
import argparse
from typing import Dict, List, Any, Tuple

import requests
import numpy as np
import pandas as pd

OUTPUT_DIR = "data"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "attractions.csv")

# Columns required by the existing recommendation pipeline.
REQUIRED_PROJECT_COLUMNS = [
    "attraction_id",
    "name",
    "realistic_scene_name",
    "attraction_type",
    "cost_level",
    "indoor_score",
    "photo_value",
    "nature_value",
    "culture_value",
    "food_value",
    "popularity",
    "estimated_time",
]

# Extra real-world columns. Downstream code can ignore these safely.
EXTRA_COLUMNS = [
    "latitude",
    "longitude",
    "osm_id",
    "osm_element_type",
    "osm_category",
    "raw_osm_main_key",
    "raw_osm_main_value",
    "address_full",
    "address_city",
    "address_district",
    "address_street",
    "address_housenumber",
    "opening_hours",
    "website",
    "phone",
    "email",
    "operator",
    "fee",
    "wheelchair",
    "wikipedia",
    "wikidata",
    "source_tag_count",
    "has_wikipedia",
    "has_website",
    "has_opening_hours",
    "distance_from_center_km",
    "source_tags_json",
]

PROJECT_COLUMNS = REQUIRED_PROJECT_COLUMNS[:4] + EXTRA_COLUMNS[:2] + EXTRA_COLUMNS[2:] + REQUIRED_PROJECT_COLUMNS[4:]

OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]

# Scores are heuristic 0~5 values designed to match the existing simulated schema.
CATEGORY_PROFILES: Dict[str, Dict[str, float]] = {
    "museum": dict(indoor_score=4.6, cost_level=2.6, photo_value=3.7, nature_value=0.4, culture_value=4.8, food_value=1.0, popularity=3.2, estimated_time=2.3),
    "gallery": dict(indoor_score=4.5, cost_level=2.3, photo_value=4.0, nature_value=0.3, culture_value=4.5, food_value=1.0, popularity=2.8, estimated_time=1.8),
    "arts_centre": dict(indoor_score=4.3, cost_level=2.5, photo_value=3.8, nature_value=0.3, culture_value=4.4, food_value=1.2, popularity=3.0, estimated_time=2.0),
    "theatre": dict(indoor_score=4.7, cost_level=3.1, photo_value=2.8, nature_value=0.1, culture_value=4.3, food_value=1.2, popularity=3.1, estimated_time=2.4),
    "cinema": dict(indoor_score=4.8, cost_level=3.0, photo_value=2.0, nature_value=0.1, culture_value=2.3, food_value=2.5, popularity=3.6, estimated_time=2.5),
    "library": dict(indoor_score=4.7, cost_level=0.4, photo_value=2.0, nature_value=0.2, culture_value=3.8, food_value=0.5, popularity=2.3, estimated_time=1.8),
    "park": dict(indoor_score=0.4, cost_level=0.6, photo_value=3.6, nature_value=4.5, culture_value=1.0, food_value=1.2, popularity=2.8, estimated_time=2.5),
    "garden": dict(indoor_score=0.5, cost_level=0.8, photo_value=4.0, nature_value=4.6, culture_value=1.4, food_value=1.0, popularity=2.8, estimated_time=1.8),
    "nature_reserve": dict(indoor_score=0.2, cost_level=0.5, photo_value=4.2, nature_value=5.0, culture_value=0.8, food_value=0.5, popularity=2.5, estimated_time=3.0),
    "viewpoint": dict(indoor_score=0.3, cost_level=0.8, photo_value=4.8, nature_value=3.8, culture_value=1.0, food_value=0.8, popularity=3.0, estimated_time=1.5),
    "picnic_site": dict(indoor_score=0.1, cost_level=0.4, photo_value=3.2, nature_value=4.1, culture_value=0.5, food_value=2.0, popularity=2.2, estimated_time=2.0),
    "zoo": dict(indoor_score=1.2, cost_level=2.8, photo_value=4.2, nature_value=4.0, culture_value=1.2, food_value=2.0, popularity=4.0, estimated_time=4.0),
    "aquarium": dict(indoor_score=4.5, cost_level=3.5, photo_value=4.0, nature_value=3.0, culture_value=1.3, food_value=1.5, popularity=3.8, estimated_time=3.0),
    "attraction": dict(indoor_score=2.0, cost_level=2.0, photo_value=3.8, nature_value=2.0, culture_value=2.5, food_value=1.5, popularity=3.2, estimated_time=2.0),
    "theme_park": dict(indoor_score=1.8, cost_level=4.0, photo_value=4.0, nature_value=1.0, culture_value=1.2, food_value=3.0, popularity=4.2, estimated_time=5.5),
    "temple": dict(indoor_score=2.5, cost_level=0.8, photo_value=3.4, nature_value=1.2, culture_value=4.6, food_value=1.3, popularity=2.8, estimated_time=1.6),
    "place_of_worship": dict(indoor_score=2.8, cost_level=0.8, photo_value=3.2, nature_value=1.0, culture_value=4.3, food_value=1.0, popularity=2.5, estimated_time=1.5),
    "historic_site": dict(indoor_score=1.6, cost_level=1.0, photo_value=4.0, nature_value=1.5, culture_value=4.7, food_value=1.0, popularity=3.0, estimated_time=1.8),
    "monument": dict(indoor_score=0.8, cost_level=0.3, photo_value=3.8, nature_value=1.0, culture_value=4.2, food_value=0.4, popularity=2.8, estimated_time=0.8),
    "memorial": dict(indoor_score=1.0, cost_level=0.3, photo_value=3.2, nature_value=1.0, culture_value=4.1, food_value=0.3, popularity=2.5, estimated_time=0.8),
    "artwork": dict(indoor_score=1.5, cost_level=0.2, photo_value=4.4, nature_value=0.8, culture_value=3.8, food_value=0.3, popularity=2.4, estimated_time=0.6),
    "restaurant": dict(indoor_score=4.0, cost_level=2.8, photo_value=2.6, nature_value=0.3, culture_value=1.6, food_value=4.7, popularity=3.4, estimated_time=1.4),
    "cafe": dict(indoor_score=4.2, cost_level=2.8, photo_value=4.0, nature_value=0.4, culture_value=2.0, food_value=4.0, popularity=3.2, estimated_time=1.3),
    "fast_food": dict(indoor_score=4.0, cost_level=1.8, photo_value=1.8, nature_value=0.1, culture_value=0.8, food_value=3.6, popularity=3.3, estimated_time=0.8),
    "food_court": dict(indoor_score=4.5, cost_level=2.2, photo_value=2.2, nature_value=0.1, culture_value=1.0, food_value=4.2, popularity=3.7, estimated_time=1.1),
    "ice_cream": dict(indoor_score=3.8, cost_level=2.0, photo_value=3.5, nature_value=0.2, culture_value=1.0, food_value=3.8, popularity=3.0, estimated_time=0.6),
    "bar": dict(indoor_score=4.2, cost_level=3.2, photo_value=3.0, nature_value=0.1, culture_value=1.8, food_value=3.2, popularity=3.4, estimated_time=2.0),
    "pub": dict(indoor_score=4.2, cost_level=3.0, photo_value=2.8, nature_value=0.1, culture_value=1.8, food_value=3.2, popularity=3.2, estimated_time=2.0),
    "nightclub": dict(indoor_score=4.4, cost_level=3.8, photo_value=3.2, nature_value=0.1, culture_value=1.3, food_value=2.5, popularity=3.5, estimated_time=3.0),
    "marketplace": dict(indoor_score=1.8, cost_level=2.0, photo_value=3.0, nature_value=0.2, culture_value=2.8, food_value=4.5, popularity=4.0, estimated_time=2.0),
    "mall": dict(indoor_score=4.8, cost_level=4.0, photo_value=2.8, nature_value=0.2, culture_value=1.2, food_value=3.6, popularity=4.0, estimated_time=3.0),
    "department_store": dict(indoor_score=4.9, cost_level=4.2, photo_value=2.6, nature_value=0.1, culture_value=1.0, food_value=3.3, popularity=4.0, estimated_time=2.5),
    "bookstore": dict(indoor_score=4.7, cost_level=2.2, photo_value=2.8, nature_value=0.1, culture_value=3.0, food_value=1.2, popularity=2.5, estimated_time=1.0),
    "sports_centre": dict(indoor_score=3.5, cost_level=2.5, photo_value=1.8, nature_value=1.0, culture_value=0.5, food_value=0.8, popularity=2.8, estimated_time=2.0),
    "stadium": dict(indoor_score=2.0, cost_level=2.8, photo_value=3.0, nature_value=0.8, culture_value=1.0, food_value=2.5, popularity=3.6, estimated_time=3.0),
    "swimming_pool": dict(indoor_score=2.5, cost_level=2.0, photo_value=1.5, nature_value=1.0, culture_value=0.3, food_value=0.5, popularity=2.6, estimated_time=2.0),
    "playground": dict(indoor_score=0.2, cost_level=0.1, photo_value=2.0, nature_value=2.5, culture_value=0.2, food_value=0.2, popularity=2.0, estimated_time=1.0),
    "hotel": dict(indoor_score=4.8, cost_level=4.0, photo_value=3.2, nature_value=0.3, culture_value=1.0, food_value=2.8, popularity=3.3, estimated_time=8.0),
    "guest_house": dict(indoor_score=4.5, cost_level=2.7, photo_value=3.0, nature_value=0.5, culture_value=1.5, food_value=1.5, popularity=2.5, estimated_time=8.0),
}

CATEGORY_DISPLAY_NAME = {
    "museum": "博物館", "gallery": "藝文展館", "arts_centre": "藝文中心", "theatre": "劇院", "cinema": "電影院", "library": "圖書館",
    "park": "公園", "garden": "花園", "nature_reserve": "自然保護區", "viewpoint": "觀景點", "picnic_site": "野餐地點", "zoo": "動物園", "aquarium": "水族館",
    "attraction": "觀光景點", "theme_park": "主題樂園", "temple": "寺廟", "place_of_worship": "宗教文化景點", "historic_site": "歷史景點", "monument": "紀念碑", "memorial": "紀念地", "artwork": "公共藝術",
    "restaurant": "餐廳", "cafe": "咖啡廳", "fast_food": "速食店", "food_court": "美食街", "ice_cream": "冰品甜點", "bar": "酒吧", "pub": "酒館", "nightclub": "夜生活場所", "marketplace": "市集",
    "mall": "購物中心", "department_store": "百貨公司", "bookstore": "書店",
    "sports_centre": "運動中心", "stadium": "體育場", "swimming_pool": "游泳池", "playground": "遊戲場",
    "hotel": "飯店", "guest_house": "旅宿",
}

# Each group is intentionally small to reduce Overpass timeout risk.
# Format: group name, filter list.
OVERPASS_FILTER_GROUPS: List[Tuple[str, List[str]]] = [
    ("museum", ['node["tourism"="museum"](area.searchArea);', 'way["tourism"="museum"](area.searchArea);', 'relation["tourism"="museum"](area.searchArea);']),
    ("gallery", ['node["tourism"="gallery"](area.searchArea);', 'way["tourism"="gallery"](area.searchArea);', 'relation["tourism"="gallery"](area.searchArea);']),
    ("arts_centre", ['node["amenity"="arts_centre"](area.searchArea);', 'way["amenity"="arts_centre"](area.searchArea);']),
    ("theatre", ['node["amenity"="theatre"](area.searchArea);', 'way["amenity"="theatre"](area.searchArea);']),
    ("cinema", ['node["amenity"="cinema"](area.searchArea);', 'way["amenity"="cinema"](area.searchArea);']),
    ("library", ['node["amenity"="library"](area.searchArea);', 'way["amenity"="library"](area.searchArea);']),
    ("park", ['node["leisure"="park"](area.searchArea);', 'way["leisure"="park"](area.searchArea);', 'relation["leisure"="park"](area.searchArea);']),
    ("garden", ['node["leisure"="garden"](area.searchArea);', 'way["leisure"="garden"](area.searchArea);']),
    ("nature_reserve", ['node["leisure"="nature_reserve"](area.searchArea);', 'way["leisure"="nature_reserve"](area.searchArea);', 'relation["leisure"="nature_reserve"](area.searchArea);']),
    ("viewpoint", ['node["tourism"="viewpoint"](area.searchArea);', 'way["tourism"="viewpoint"](area.searchArea);']),
    ("picnic_site", ['node["tourism"="picnic_site"](area.searchArea);']),
    ("zoo", ['node["tourism"="zoo"](area.searchArea);', 'way["tourism"="zoo"](area.searchArea);']),
    ("aquarium", ['node["tourism"="aquarium"](area.searchArea);', 'way["tourism"="aquarium"](area.searchArea);']),
    ("attraction", ['node["tourism"="attraction"](area.searchArea);', 'way["tourism"="attraction"](area.searchArea);', 'relation["tourism"="attraction"](area.searchArea);']),
    ("theme_park", ['node["tourism"="theme_park"](area.searchArea);', 'way["tourism"="theme_park"](area.searchArea);']),
    ("place_of_worship", ['node["amenity"="place_of_worship"](area.searchArea);', 'way["amenity"="place_of_worship"](area.searchArea);', 'relation["amenity"="place_of_worship"](area.searchArea);']),
    ("historic", ['node["historic"](area.searchArea);', 'way["historic"](area.searchArea);', 'relation["historic"](area.searchArea);']),
    ("artwork", ['node["tourism"="artwork"](area.searchArea);', 'way["tourism"="artwork"](area.searchArea);']),
    ("restaurant", ['node["amenity"="restaurant"](area.searchArea);']),
    ("cafe", ['node["amenity"="cafe"](area.searchArea);']),
    ("fast_food", ['node["amenity"="fast_food"](area.searchArea);']),
    ("food_court", ['node["amenity"="food_court"](area.searchArea);']),
    ("ice_cream", ['node["amenity"="ice_cream"](area.searchArea);']),
    ("bar", ['node["amenity"="bar"](area.searchArea);']),
    ("pub", ['node["amenity"="pub"](area.searchArea);']),
    ("nightclub", ['node["amenity"="nightclub"](area.searchArea);']),
    ("marketplace", ['node["amenity"="marketplace"](area.searchArea);', 'way["amenity"="marketplace"](area.searchArea);']),
    ("mall", ['node["shop"="mall"](area.searchArea);', 'way["shop"="mall"](area.searchArea);']),
    ("department_store", ['node["shop"="department_store"](area.searchArea);', 'way["shop"="department_store"](area.searchArea);']),
    ("bookstore", ['node["shop"="books"](area.searchArea);']),
    ("sports_centre", ['node["leisure"="sports_centre"](area.searchArea);', 'way["leisure"="sports_centre"](area.searchArea);']),
    ("stadium", ['node["leisure"="stadium"](area.searchArea);', 'way["leisure"="stadium"](area.searchArea);']),
    ("swimming_pool", ['node["leisure"="swimming_pool"](area.searchArea);', 'way["leisure"="swimming_pool"](area.searchArea);']),
    ("playground", ['node["leisure"="playground"](area.searchArea);', 'way["leisure"="playground"](area.searchArea);']),
    ("hotel", ['node["tourism"="hotel"](area.searchArea);', 'way["tourism"="hotel"](area.searchArea);']),
    ("guest_house", ['node["tourism"="guest_house"](area.searchArea);', 'node["tourism"="hostel"](area.searchArea);']),
]


def clip_score(value: float, low: float = 0.0, high: float = 5.0) -> float:
    return float(np.clip(value, low, high))


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def infer_category(tags: Dict[str, Any]) -> str:
    tourism = tags.get("tourism")
    amenity = tags.get("amenity")
    leisure = tags.get("leisure")
    shop = tags.get("shop")
    historic = tags.get("historic")
    religion = tags.get("religion")

    if tourism in CATEGORY_PROFILES:
        return tourism
    if amenity in CATEGORY_PROFILES:
        if amenity == "place_of_worship" and religion in {"buddhist", "taoist", "shinto"}:
            return "temple"
        return amenity
    if leisure in CATEGORY_PROFILES:
        return leisure
    if shop == "mall":
        return "mall"
    if shop == "department_store":
        return "department_store"
    if shop == "books":
        return "bookstore"
    if historic:
        if historic in {"monument"}:
            return "monument"
        if historic in {"memorial"}:
            return "memorial"
        return "historic_site"
    return "attraction"


def get_main_osm_tag(tags: Dict[str, Any], category: str) -> Tuple[str, str]:
    for key in ["tourism", "amenity", "leisure", "shop", "historic", "natural"]:
        if key in tags:
            return key, safe_text(tags.get(key))
    return "inferred", category


def estimate_popularity(tags: Dict[str, Any], category: str) -> float:
    base = CATEGORY_PROFILES[category]["popularity"]
    if tags.get("wikipedia") or tags.get("wikidata"):
        base += 0.7
    if tags.get("website") or tags.get("contact:website"):
        base += 0.3
    if tags.get("name:en"):
        base += 0.2
    if tags.get("opening_hours"):
        base += 0.2
    if tags.get("tourism") in {"attraction", "theme_park", "zoo", "aquarium"}:
        base += 0.2
    return clip_score(base)


def adjust_scores_from_tags(profile: Dict[str, float], tags: Dict[str, Any]) -> Dict[str, float]:
    score = profile.copy()
    fee = safe_text(tags.get("fee")).lower()
    if fee in {"yes", "true", "1"}:
        score["cost_level"] = clip_score(score["cost_level"] + 0.8)
    elif fee in {"no", "false", "0"}:
        score["cost_level"] = clip_score(score["cost_level"] - 0.6)

    indoor = safe_text(tags.get("indoor")).lower()
    if indoor in {"yes", "true", "1"}:
        score["indoor_score"] = clip_score(score["indoor_score"] + 0.8)
    elif indoor in {"no", "false", "0"}:
        score["indoor_score"] = clip_score(score["indoor_score"] - 0.5)

    if tags.get("historic"):
        score["culture_value"] = clip_score(score["culture_value"] + 0.8)
        score["photo_value"] = clip_score(score["photo_value"] + 0.3)
    if tags.get("natural"):
        score["nature_value"] = clip_score(score["nature_value"] + 1.0)
        score["indoor_score"] = clip_score(score["indoor_score"] - 0.8)
    if tags.get("wikipedia") or tags.get("wikidata"):
        score["culture_value"] = clip_score(score["culture_value"] + 0.2)
        score["popularity"] = clip_score(score["popularity"] + 0.4)
    return score


def get_coordinates(element: Dict[str, Any]) -> Tuple[Any, Any]:
    if "lat" in element and "lon" in element:
        return element["lat"], element["lon"]
    center = element.get("center")
    if center:
        return center.get("lat"), center.get("lon")
    return None, None


def build_overpass_query(area_name: str, filters: List[str], limit: int) -> str:
    filters_text = "\n".join(filters)
    return f"""
[out:json][timeout:90];
area["name"="{area_name}"]->.searchArea;
(
{filters_text}
);
out tags center {limit};
"""


def post_overpass_with_retry(query: str, endpoint: str, max_retries: int = 2, sleep_sec: float = 2.0) -> Dict[str, Any]:
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                endpoint,
                data={"data": query},
                timeout=120,
                headers={"User-Agent": "travel-clustering-project-expanded/1.2 (student project)"},
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as exc:
            last_error = exc
            print(f"  Request failed at {endpoint} attempt {attempt}/{max_retries}: {exc}")
            time.sleep(sleep_sec * attempt)
    raise RuntimeError(f"Overpass request failed after retries: {last_error}")


def fetch_osm_pois(area_name: str, per_group_limit: int = 80, sleep_sec: float = 1.5, endpoint: str = "auto") -> List[Dict[str, Any]]:
    print(f"Fetching POIs from OpenStreetMap Overpass API: {area_name}")
    endpoints = OVERPASS_ENDPOINTS if endpoint == "auto" else [endpoint]
    all_elements: List[Dict[str, Any]] = []
    seen = set()

    for group_index, (group_name, filters) in enumerate(OVERPASS_FILTER_GROUPS, start=1):
        query = build_overpass_query(area_name=area_name, filters=filters, limit=per_group_limit)
        group_success = False
        for current_endpoint in endpoints:
            try:
                print(f"Query group {group_index}/{len(OVERPASS_FILTER_GROUPS)} [{group_name}] via {current_endpoint}")
                payload = post_overpass_with_retry(query=query, endpoint=current_endpoint, max_retries=2, sleep_sec=sleep_sec)
                elements = payload.get("elements", [])
                added = 0
                for element in elements:
                    key = (element.get("type"), element.get("id"))
                    if key not in seen:
                        seen.add(key)
                        all_elements.append(element)
                        added += 1
                print(f"  Retrieved {len(elements)} elements; added {added}; total unique: {len(all_elements)}")
                group_success = True
                break
            except RuntimeError as exc:
                print(f"  Endpoint failed: {exc}")
                continue
        if not group_success:
            print(f"  Skip group [{group_name}]; all endpoints failed.")
        time.sleep(sleep_sec)

    if not all_elements:
        raise RuntimeError("No OSM elements were returned. Try another area name such as 台北市, Taipei City, 新北市, or reduce request size.")
    return all_elements


def build_address(tags: Dict[str, Any]) -> Dict[str, str]:
    city = safe_text(tags.get("addr:city"))
    district = safe_text(tags.get("addr:district") or tags.get("addr:suburb"))
    street = safe_text(tags.get("addr:street"))
    housenumber = safe_text(tags.get("addr:housenumber"))
    full = safe_text(tags.get("addr:full"))
    if not full:
        parts = [city, district, street, housenumber]
        full = "".join([p for p in parts if p])
    return dict(address_full=full, address_city=city, address_district=district, address_street=street, address_housenumber=housenumber)


def clean_category_name(category: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", category).strip("_")


def elements_to_raw_rows(elements: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    seen_osm_ids = set()

    for element in elements:
        tags = element.get("tags", {})
        lat, lon = get_coordinates(element)
        if lat is None or lon is None:
            continue

        name = tags.get("name") or tags.get("name:zh") or tags.get("name:en")
        if not name:
            continue

        osm_id = f"{element.get('type')}/{element.get('id')}"
        if osm_id in seen_osm_ids:
            continue
        seen_osm_ids.add(osm_id)

        category = infer_category(tags)
        profile = adjust_scores_from_tags(CATEGORY_PROFILES[category], tags)
        popularity = estimate_popularity(tags, category)
        main_key, main_value = get_main_osm_tag(tags, category)
        address = build_address(tags)

        website = safe_text(tags.get("website") or tags.get("contact:website") or tags.get("url"))
        phone = safe_text(tags.get("phone") or tags.get("contact:phone"))
        email = safe_text(tags.get("email") or tags.get("contact:email"))

        rows.append({
            "name": "",  # filled after final sampling
            "realistic_scene_name": safe_text(name),
            "attraction_type": category,
            "latitude": float(lat),
            "longitude": float(lon),
            "osm_id": osm_id,
            "osm_element_type": safe_text(element.get("type")),
            "osm_category": CATEGORY_DISPLAY_NAME.get(category, category),
            "raw_osm_main_key": main_key,
            "raw_osm_main_value": main_value,
            **address,
            "opening_hours": safe_text(tags.get("opening_hours")),
            "website": website,
            "phone": phone,
            "email": email,
            "operator": safe_text(tags.get("operator") or tags.get("brand")),
            "fee": safe_text(tags.get("fee")),
            "wheelchair": safe_text(tags.get("wheelchair")),
            "wikipedia": safe_text(tags.get("wikipedia")),
            "wikidata": safe_text(tags.get("wikidata")),
            "source_tag_count": int(len(tags)),
            "has_wikipedia": int(bool(tags.get("wikipedia") or tags.get("wikidata"))),
            "has_website": int(bool(website)),
            "has_opening_hours": int(bool(tags.get("opening_hours"))),
            "cost_level": clip_score(profile["cost_level"]),
            "indoor_score": clip_score(profile["indoor_score"]),
            "photo_value": clip_score(profile["photo_value"]),
            "nature_value": clip_score(profile["nature_value"]),
            "culture_value": clip_score(profile["culture_value"]),
            "food_value": clip_score(profile["food_value"]),
            "popularity": popularity,
            "estimated_time": float(profile["estimated_time"]),
            "source_tags_json": json.dumps(tags, ensure_ascii=False, sort_keys=True),
        })

    if not rows:
        raise RuntimeError("No usable named OSM POIs were returned.")
    return pd.DataFrame(rows)


def balanced_sample(df: pd.DataFrame, max_items: int, seed: int = 42) -> pd.DataFrame:
    """Sample POIs so early, common categories do not dominate output."""
    if len(df) <= max_items:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    categories = sorted(df["attraction_type"].unique())
    per_category_base = max(1, max_items // max(1, len(categories)))

    selected_parts = []
    remaining_parts = []
    for category in categories:
        group = df[df["attraction_type"] == category].copy()
        # Prefer more informative POIs first, then randomize among similar rows.
        group["_metadata_score"] = (
            group["has_wikipedia"] * 3
            + group["has_website"] * 2
            + group["has_opening_hours"] * 2
            + group["source_tag_count"].clip(upper=12) / 12
        )
        group = group.sort_values(["_metadata_score", "popularity"], ascending=False)
        take_n = min(per_category_base, len(group))
        selected_parts.append(group.head(take_n))
        remaining_parts.append(group.iloc[take_n:])

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
    remaining_slots = max_items - len(selected)

    if remaining_slots > 0 and remaining_parts:
        remaining = pd.concat(remaining_parts, ignore_index=True)
        if not remaining.empty:
            # Weighted random-ish fill: prioritize metadata quality but keep diversity.
            remaining = remaining.sort_values(["_metadata_score", "popularity"], ascending=False)
            selected = pd.concat([selected, remaining.head(remaining_slots)], ignore_index=True)

    if len(selected) > max_items:
        selected = selected.head(max_items)

    selected = selected.drop(columns=["_metadata_score"], errors="ignore")
    selected = selected.sample(frac=1, random_state=seed).reset_index(drop=True)
    return selected


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def add_distance_from_center(df: pd.DataFrame, center_lat: float, center_lon: float) -> pd.DataFrame:
    df = df.copy()
    df["distance_from_center_km"] = df.apply(lambda row: haversine_km(center_lat, center_lon, row["latitude"], row["longitude"]), axis=1)
    return df


def finalize_attraction_ids(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    df.insert(0, "attraction_id", np.arange(1, len(df) + 1))
    df["name"] = df.apply(lambda row: f"OSM_{int(row['attraction_id']):03d}_{clean_category_name(row['attraction_type'])}", axis=1)
    for col in PROJECT_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[PROJECT_COLUMNS]


def fetch_open_meteo_weather(latitude: float, longitude: float) -> Dict[str, float]:
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
    parser.add_argument("--area", type=str, default="臺北市", help="OSM area name, e.g., 臺北市, 台北市, Taipei City, 新北市")
    parser.add_argument("--max-items", type=int, default=300, help="Maximum number of attractions to save")
    parser.add_argument("--per-group-limit", type=int, default=80, help="Maximum OSM elements requested per category group")
    parser.add_argument("--center-lat", type=float, default=25.0330, help="Center latitude for distance calculation")
    parser.add_argument("--center-lon", type=float, default=121.5654, help="Center longitude for distance calculation")
    parser.add_argument("--add-weather", action="store_true", help="Add current city-level Open-Meteo weather columns")
    parser.add_argument("--overpass-url", type=str, default="auto", help="Overpass endpoint URL, or auto to try multiple endpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for balanced sampling")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    elements = fetch_osm_pois(
        area_name=args.area,
        per_group_limit=args.per_group_limit,
        endpoint=args.overpass_url,
    )
    raw_df = elements_to_raw_rows(elements)
    raw_df = add_distance_from_center(raw_df, args.center_lat, args.center_lon)
    sampled_df = balanced_sample(raw_df, max_items=args.max_items, seed=args.seed)
    final_df = finalize_attraction_ids(sampled_df)

    if args.add_weather:
        weather = fetch_open_meteo_weather(args.center_lat, args.center_lon)
        for key, value in weather.items():
            final_df[key] = value

    final_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Attraction count: {len(final_df)}")
    print(f"Column count: {len(final_df.columns)}")
    print("Category distribution:")
    print(final_df["attraction_type"].value_counts().head(50))
    print(final_df.head(10))


if __name__ == "__main__":
    main()
