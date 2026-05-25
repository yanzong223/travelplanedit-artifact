"""
Spatio-Temporal Coherence (STC) Calculator for TripCraft-style evaluation.

Implements the STC formula:
STC = (1/|E|) * Σ I{Time(i) + Travel(i,j) <= Time(j)}

where:
- E: set of directed edges between adjacent activities
- Time(i): end time of activity i
- Travel(i,j): estimated travel time from i to j
- Time(j): start time of activity j
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import math

from utils.chinatravel_plan import require_chinatravel_plan

logger = logging.getLogger(__name__)


def compute_stc(plan: Dict[str, Any], world_env=None) -> Dict[str, Any]:
    """
    Calculate Spatio-Temporal Coherence (STC) for a given travel plan.

    Args:
        plan: Canonical ChinaTravel plan with itinerary
        world_env: Optional ChinaTravel world_env for accurate travel time calculation

    Returns:
        Dictionary containing STC results:
        - stc: float (overall STC score)
        - edge_count: int (total number of edges evaluated)
        - valid_edge_count: int (number of valid edges satisfying time constraint)
        - invalid_edges: List[Dict] (detailed information about invalid edges)
    """
    try:
        # Extract itinerary from plan
        plan = require_chinatravel_plan(plan, context="stc_calculator.plan")
        itinerary = plan.get("itinerary", [])
        if not itinerary:
            return {
                "stc": 1.0,
                "edge_count": 0,
                "valid_edge_count": 0,
                "invalid_edges": [],
                "message": "No itinerary found"
            }

        # Generate edges between adjacent activities
        edges = _generate_activity_edges(itinerary)

        if not edges:
            return {
                "stc": 1.0,
                "edge_count": 0,
                "valid_edge_count": 0,
                "invalid_edges": [],
                "message": "No activity edges found"
            }

        # Evaluate each edge
        valid_edges = 0
        invalid_edges = []

        for edge in edges:
            is_valid, travel_time, details = _evaluate_edge(edge, world_env)

            if is_valid:
                valid_edges += 1
            else:
                invalid_edges.append({
                    "edge": edge,
                    "travel_time": travel_time,
                    "details": details
                })

        # Calculate STC
        stc = valid_edges / len(edges) if edges else 1.0

        return {
            "stc": stc,
            "edge_count": len(edges),
            "valid_edge_count": valid_edges,
            "invalid_edges": invalid_edges
        }

    except Exception as e:
        logger.error(f"STC calculation failed: {e}")
        return {
            "stc": 0.0,
            "edge_count": 0,
            "valid_edge_count": 0,
            "invalid_edges": [],
            "error": str(e)
        }


def _generate_activity_edges(itinerary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate edges between adjacent activities in the itinerary.

    Args:
        itinerary: List of days with activities

    Returns:
        List of edge dictionaries connecting adjacent activities
    """
    edges = []

    for day_idx, day in enumerate(itinerary):
        activities = day.get("activities", [])

        # Sort activities by start time
        sorted_activities = _sort_activities_by_time(activities)

        # Generate edges between consecutive activities within the same day
        for i in range(len(sorted_activities) - 1):
            current = sorted_activities[i]
            next_activity = sorted_activities[i + 1]

            edges.append({
                "type": "intra_day",
                "day": day_idx + 1,
                "current": current,
                "next": next_activity
            })

        # Generate edge from last activity of current day to first activity of next day
        if day_idx < len(itinerary) - 1:
            next_day = itinerary[day_idx + 1]
            next_day_activities = next_day.get("activities", [])
            next_day_sorted = _sort_activities_by_time(next_day_activities)

            if sorted_activities and next_day_sorted:
                edges.append({
                    "type": "inter_day",
                    "from_day": day_idx + 1,
                    "to_day": day_idx + 2,
                    "current": sorted_activities[-1],
                    "next": next_day_sorted[0]
                })

    return edges


def _sort_activities_by_time(activities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort activities by start time."""
    valid_activities = []

    for activity in activities:
        start_time = activity.get("startTime")
        if start_time:
            try:
                # Parse time to datetime for sorting
                time_obj = datetime.strptime(start_time, "%H:%M")
                valid_activities.append((time_obj, activity))
            except ValueError:
                # Skip activities with invalid time format
                logger.warning(f"Invalid time format: {start_time}")
                continue

    # Sort by time and return activities
    valid_activities.sort(key=lambda x: x[0])
    return [activity for _, activity in valid_activities]


def _evaluate_edge(edge: Dict[str, Any], world_env=None) -> Tuple[bool, float, Dict[str, Any]]:
    """
    Evaluate if an edge satisfies the temporal constraint.

    Args:
        edge: Edge dictionary with current and next activities
        world_env: Optional world_env for travel time calculation

    Returns:
        Tuple of (is_valid: bool, travel_time_minutes: float, details: Dict)
    """
    current = edge["current"]
    next_activity = edge["next"]

    # Extract times
    current_end_time = current.get("endTime", current.get("startTime"))
    next_start_time = next_activity.get("startTime")

    if not current_end_time or not next_start_time:
        return False, 0.0, {"error": "Missing time information"}

    try:
        # Parse times
        current_end = datetime.strptime(current_end_time, "%H:%M")
        next_start = datetime.strptime(next_start_time, "%H:%M")

        # Calculate available time window
        if current_end > next_start:
            # Current activity ends after next starts - invalid
            available_time = -1.0
        else:
            available_time = (next_start - current_end).total_seconds() / 60.0  # Convert to minutes

        # Estimate travel time
        travel_time = _estimate_travel_time(current, next_activity, world_env)

        # Check if constraint is satisfied
        is_valid = travel_time <= available_time

        details = {
            "current_end_time": current_end_time,
            "next_start_time": next_start_time,
            "available_time_minutes": available_time,
            "estimated_travel_time_minutes": travel_time,
            "time_margin_minutes": available_time - travel_time,
            "edge_type": edge["type"]
        }

        return is_valid, travel_time, details

    except ValueError as e:
        return False, 0.0, {"error": f"Time parsing error: {e}"}
    except Exception as e:
        return False, 0.0, {"error": f"Evaluation error: {e}"}


def _estimate_travel_time(
    from_activity: Dict[str, Any],
    to_activity: Dict[str, Any],
    world_env=None
) -> float:
    """
    Estimate travel time between two activities.

    Args:
        from_activity: Starting activity
        to_activity: Destination activity
        world_env: Optional world_env for accurate calculation

    Returns:
        Estimated travel time in minutes
    """
    # If world_env is available, use it for accurate travel time calculation
    if world_env and hasattr(world_env, 'get_travel_time'):
        try:
            # Extract POI names or locations
            from_poi = from_activity.get("poiName", "")
            to_poi = to_activity.get("poiName", "")

            if from_poi and to_poi:
                # Use world_env to get accurate travel time
                travel_time = world_env.get_travel_time(from_poi, to_poi)
                if travel_time is not None:
                    return float(travel_time)
        except Exception as e:
            logger.warning(f"world_env travel time calculation failed: {e}")

    # Fallback: use simplified estimation based on activity types and locations
    return _estimate_travel_time_simple(from_activity, to_activity)


def _estimate_travel_time_simple(
    from_activity: Dict[str, Any],
    to_activity: Dict[str, Any]
) -> float:
    """
    Simple travel time estimation when world_env is not available.
    Uses heuristics based on activity types and POI names.

    Args:
        from_activity: Starting activity
        to_activity: Destination activity

    Returns:
        Estimated travel time in minutes
    """
    from_poi = from_activity.get("poiName", "")
    to_poi = to_activity.get("poiName", "")
    from_type = from_activity.get("type", "")
    to_type = to_activity.get("type", "")

    # Same POI - minimal travel time
    if from_poi == to_poi:
        return 5.0

    # Check for obvious duplicates or nearby locations
    if _are_same_location(from_poi, to_poi):
        return 10.0

    # Use activity type-based heuristics
    if from_type == "accommodation" or to_type == "accommodation":
        # Travel to/from accommodation usually shorter
        return 20.0

    if from_type == "restaurant" and to_type == "attraction":
        # Restaurant to attraction: moderate travel time
        return 30.0

    if from_type == "attraction" and to_type == "restaurant":
        # Attraction to restaurant: moderate travel time
        return 30.0

    if from_type == "attraction" and to_type == "attraction":
        # Between attractions: longer travel time
        return 45.0

    # Default estimation
    return 35.0


def _are_same_location(poi1: str, poi2: str) -> bool:
    """
    Check if two POI names refer to the same general location.
    Simple heuristic based on common substrings.
    """
    if not poi1 or not poi2:
        return False

    # Normalize names
    poi1_lower = poi1.lower().replace(" ", "")
    poi2_lower = poi2.lower().replace(" ", "")

    # Check for common indicators of same location
    common_indicators = [
        "station", "机场", "airport", "hotel", "酒店", "mall", "商场",
        "park", "公园", "square", "广场", "street", "街"
    ]

    for indicator in common_indicators:
        if indicator in poi1_lower and indicator in poi2_lower:
            # Check if there are other distinguishing words
            words1 = set(poi1_lower.replace(indicator, "").split("_"))
            words2 = set(poi2_lower.replace(indicator, "").split("_"))

            # If few distinguishing words, likely same location
            if len(words1.intersection(words2)) >= 1 or len(words1) <= 2 or len(words2) <= 2:
                return True

    # Check for exact substring match
    if poi1_lower in poi2_lower or poi2_lower in poi1_lower:
        return True

    return False
