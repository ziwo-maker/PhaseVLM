import json
import os
from copy import deepcopy
from typing import Dict, List, Tuple


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def build_single_intersection_roadnet(
    src_roadnet: str,
    dst_roadnet: str,
    target_intersection: str,
) -> Tuple[Dict, List[str]]:
    roadnet = _load_json(src_roadnet)
    intersections = {item["id"]: item for item in roadnet["intersections"]}
    roads = {item["id"]: item for item in roadnet["roads"]}

    if target_intersection not in intersections:
        raise ValueError(f"Intersection not found: {target_intersection}")

    target = intersections[target_intersection]
    keep_roads = list(target.get("roads", []))

    keep_intersections = set([target_intersection])
    for road_id in keep_roads:
        road = roads[road_id]
        keep_intersections.add(road["startIntersection"])
        keep_intersections.add(road["endIntersection"])

    virtual_template = next(
        (item for item in roadnet["intersections"] if item.get("virtual")), None
    )

    new_intersections = []
    for inter_id in keep_intersections:
        inter = deepcopy(intersections[inter_id])
        inter["roads"] = [rid for rid in inter.get("roads", []) if rid in keep_roads]
        if inter_id != target_intersection:
            inter["virtual"] = True
            inter["roadLinks"] = []
            if virtual_template is not None:
                inter["trafficLight"] = deepcopy(virtual_template.get("trafficLight", {}))
            else:
                inter["trafficLight"] = {"roadLinkIndices": [], "lightphases": []}
        new_intersections.append(inter)

    new_roads = [deepcopy(roads[rid]) for rid in keep_roads]

    new_roadnet = {"intersections": new_intersections, "roads": new_roads}
    os.makedirs(os.path.dirname(dst_roadnet), exist_ok=True)
    _save_json(dst_roadnet, new_roadnet)
    return new_roadnet, keep_roads


def build_single_intersection_traffic(
    src_traffic: str,
    dst_traffic: str,
    roadnet: Dict,
    target_intersection: str,
    flow_interval: float = 2.0,
    start_time: int = 0,
    end_time: int = 3600,
) -> None:
    vehicle_template = None
    if src_traffic and os.path.exists(src_traffic):
        traffic_data = _load_json(src_traffic)
        if traffic_data:
            vehicle_template = traffic_data[0].get("vehicle")
    if vehicle_template is None:
        vehicle_template = {
            "length": 5.0,
            "width": 2.0,
            "maxPosAcc": 2.0,
            "maxNegAcc": 4.5,
            "usualPosAcc": 2.0,
            "usualNegAcc": 4.5,
            "minGap": 2.5,
            "maxSpeed": 11.111,
            "headwayTime": 2,
        }

    inter = next(i for i in roadnet["intersections"] if i["id"] == target_intersection)
    road_links = inter.get("roadLinks", [])
    routes = []
    for link in road_links:
        route = [link["startRoad"], link["endRoad"]]
        if route not in routes:
            routes.append(route)

    flows = []
    for route in routes:
        flows.append(
            {
                "vehicle": deepcopy(vehicle_template),
                "route": route,
                "interval": float(flow_interval),
                "startTime": int(start_time),
                "endTime": int(end_time),
            }
        )

    os.makedirs(os.path.dirname(dst_traffic), exist_ok=True)
    _save_json(dst_traffic, flows)
