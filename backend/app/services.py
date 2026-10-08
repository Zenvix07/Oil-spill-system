import math
from datetime import datetime, timedelta
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlambda/2)**2
    return 2 * r * math.asin(math.sqrt(a))


def vessel_candidates(origin_lat=10.5, origin_lon=72.5):
    """
    Deterministic demonstration AIS candidates.
    Replace this with historical AIS/PostGIS queries in production.
    """
    vessels = [
        {"mmsi": "419001234", "name": "MV OCEAN STAR", "lat": 10.48, "lon": 72.47,
         "speed": 7.2, "course": 181, "type": "Tanker"},
        {"mmsi": "419005678", "name": "MT BLUE HORIZON", "lat": 10.62, "lon": 72.56,
         "speed": 12.4, "course": 92, "type": "Cargo"},
        {"mmsi": "419008901", "name": "MV SOUTHERN WIND", "lat": 10.35, "lon": 72.68,
         "speed": 5.1, "course": 275, "type": "Tanker"},
        {"mmsi": "419003456", "name": "MV COASTAL TRADER", "lat": 10.72, "lon": 72.39,
         "speed": 15.8, "course": 44, "type": "Cargo"},
    ]

    result = []
    for v in vessels:
        d = haversine_km(origin_lat, origin_lon, v["lat"], v["lon"])
        proximity = max(0, 100 - d * 7.5)

        # Demo temporal/trajectory/anomaly factors.
        temporal = {"419001234": 92, "419005678": 73, "419008901": 81, "419003456": 59}[v["mmsi"]]
        trajectory = {"419001234": 91, "419005678": 84, "419008901": 66, "419003456": 62}[v["mmsi"]]
        anomaly = {"419001234": 87, "419005678": 61, "419008901": 72, "419003456": 45}[v["mmsi"]]

        score = (
            0.30 * proximity +
            0.25 * temporal +
            0.20 * trajectory +
            0.15 * anomaly +
            0.10 * 70
        )

        result.append({
            **v,
            "distance_km": round(d, 2),
            "proximity": round(proximity, 1),
            "temporal": temporal,
            "trajectory": trajectory,
            "anomaly": anomaly,
            "score": round(score, 1),
        })

    return sorted(result, key=lambda x: x["score"], reverse=True)


def drift_track(lat, lon, steps=12):
    """
    Lightweight visual demo. Replace with Copernicus/NOAA current + wind forcing
    for the scientific hindcast/forecast layer.
    """
    points = []
    for i in range(steps):
        t = i / (steps - 1)
        points.append([
            round(lat - 0.16 * t + 0.025 * math.sin(i * 0.7), 5),
            round(lon - 0.21 * t + 0.018 * math.cos(i * 0.5), 5),
        ])
    return points
