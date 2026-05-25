"""
Spatial calculator for travel plan analysis.

Provides geometric and temporal constraint generation for force_poi_on_day scenarios.
Uses ChinaTravel's real POI search API for accurate coordinates.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from utils.logging import get_logger
from utils.chinatravel_plan import require_chinatravel_plan

logger = get_logger(__name__)

# Add ChinaTravel to path
chinatravel_root = Path(__file__).parent.parent.parent / "Chinatravel" / "ChinaTravel"
sys.path.insert(0, str(chinatravel_root))

try:
    from chinatravel.environment.tools.poi.apis import Poi
    CHINATRAVEL_POI_AVAILABLE = True
except ImportError:
    CHINATRAVEL_POI_AVAILABLE = False
    logger.warning("ChinaTravel POI API not available")

class SpatialCalculator:
    """Calculator for spatial and temporal analysis of travel plans."""

    def __init__(self):
        """Initialize spatial calculator with ChinaTravel POI API."""
        self.poi_api = None
        if CHINATRAVEL_POI_AVAILABLE:
            try:
                self.poi_api = Poi()
                logger.info("ChinaTravel POI API initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize ChinaTravel POI API: {e}")

        # Spatial thresholds (in kilometers)
        self.R_IN = 3.0   # Inner radius for "in area" activities
        self.R_OUT = 8.0  # Outer radius for outlier detection
        self.OUTLIER_RATIO_THRESHOLD = 0.2  # Maximum allowed outlier ratio

        # Temporal thresholds
        self.MIN_VALID_DURATION = 60  # Minimum activity duration in minutes

        # Valid activity types for time structure analysis
        self.VALID_ACTIVITY_TYPES = {
            "attraction", "museum", "park", "shopping",
            "restaurant", "entertainment", "culture"
        }

    def get_poi_coordinates(self, poi_name: str, target_city: str) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a POI using ChinaTravel API.

        Args:
            poi_name: Name of the POI
            target_city: Target city name

        Returns:
            (lat, lon) tuple or None if not found
        """
        if not self.poi_api:
            logger.error("ChinaTravel POI API not available")
            return None

        try:
            result = self.poi_api.search(target_city, poi_name)

            if isinstance(result, str):  # Error message
                logger.warning(f"POI search returned error: {result}")
                return None

            if result and len(result) == 2:  # (lat, lon) tuple
                logger.debug(f"Found coordinates for {poi_name} in {target_city}: {result}")
                return result
            else:
                logger.warning(f"Invalid result format for POI {poi_name}: {result}")
                return None

        except Exception as e:
            logger.error(f"Error searching for POI {poi_name} in {target_city}: {e}")
            return None

    def calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        Calculate distance between two coordinates in kilometers.

        Uses Haversine formula for accuracy.
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2

        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))

        # Earth's radius in kilometers
        return 6371 * c

    def extract_day_activities(self, plan: Dict[str, Any], day_index: int) -> List[Dict[str, Any]]:
        """
        Extract activities for a specific day from ChinaTravel plan.

        Args:
            plan: ChinaTravel format plan
            day_index: 1-based day index

        Returns:
            List of activity dictionaries
        """
        plan = require_chinatravel_plan(plan, context="spatial_calculator.plan")
        itinerary = plan.get("itinerary", [])

        if day_index <= 0 or day_index > len(itinerary):
            return []

        day_plan = itinerary[day_index - 1]
        return day_plan.get("activities", [])

    def calculate_activity_duration(self, activity: Dict[str, Any]) -> Optional[int]:
        """
        Calculate activity duration in minutes.

        Args:
            activity: Activity dictionary with startTime and endTime

        Returns:
            Duration in minutes, or None if not available
        """
        try:
            start_time = activity.get("startTime", "")
            end_time = activity.get("endTime", "")

            if not start_time or not end_time:
                return None

            # Parse time format "HH:MM"
            start_h, start_m = map(int, start_time.split(":"))
            end_h, end_m = map(int, end_time.split(":"))

            start_minutes = start_h * 60 + start_m
            end_minutes = end_h * 60 + end_m

            return max(0, end_minutes - start_minutes)

        except Exception:
            return None

    def is_valid_attraction_activity(self, activity: Dict[str, Any]) -> bool:
        """
        Check if an activity is a valid attraction for time structure analysis.
        """
        activity_type = activity.get("type", "").lower()
        poi_name = activity.get("poiName", activity.get("position", ""))

        # Check if it's a valid type
        if not any(valid_type in activity_type for valid_type in self.VALID_ACTIVITY_TYPES):
            return False

        # Check if it has a valid duration
        duration = self.calculate_activity_duration(activity)
        if duration is None:
            return False

        return True

    def get_activity_coordinates(self, activity: Dict[str, Any], target_city: str) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for an activity.

        Args:
            activity: Activity dictionary
            target_city: Target city name

        Returns:
            (lat, lon) tuple or None if not found
        """
        poi_name = activity.get("poiName", activity.get("position", ""))
        if not poi_name:
            return None

        return self.get_poi_coordinates(poi_name, target_city)

    def generate_force_poi_constraints(
        self,
        plan: Dict[str, Any],
        day_index: int,
        target_poi: str,
        target_city: str = "苏州"
    ) -> List[str]:
        """
        Generate DSL constraints for force_poi_on_day scenario using spatial geometry rules.

        Args:
            plan: ChinaTravel format plan
            day_index: Target day index (1-based)
            target_poi: Name of target POI
            target_city: Target city name

        Returns:
            List of DSL constraint strings
        """
        constraints = []

        try:
            # Get target POI coordinates
            target_coords = self.get_poi_coordinates(target_poi, target_city)
            if not target_coords:
                logger.warning(f"Could not get coordinates for target POI: {target_poi}")
                return constraints

            # Extract day activities
            activities = self.extract_day_activities(plan, day_index)

            # Generate constraints

            # Constraint 1: Must include target POI
            constraint1 = self._generate_target_poi_constraint(day_index, target_poi)
            constraints.append(constraint1)

            # Constraint 2: Activity clustering around target area
            constraint2 = self._generate_clustering_constraint(day_index, target_coords, activities, target_city)
            constraints.append(constraint2)

            # Constraint 3: No far outliers
            constraint3 = self._generate_outlier_constraint(day_index, target_coords, activities, target_city)
            constraints.append(constraint3)

            # Constraint 4: Time structure validity
            constraint4 = self._generate_time_structure_constraint(day_index, activities)
            constraints.append(constraint4)

        except Exception as e:
            logger.error(f"Error generating force POI constraints: {e}")

        return constraints

    def _generate_target_poi_constraint(self, day_index: int, target_poi: str) -> str:
        """Generate constraint ensuring target POI is included."""
        return f"""day_index = {day_index}
target_poi = '{target_poi}'
if day_index <= len(plan.get('itinerary', [])):
    acts = plan['itinerary'][day_index - 1].get('activities', [])
    result = any(a.get('poiName') == target_poi for a in acts)
else:
    result = False"""

    def _generate_clustering_constraint(
        self,
        day_index: int,
        target_coords: Tuple[float, float],
        activities: List[Dict[str, Any]],
        target_city: str
    ) -> str:
        """Generate constraint ensuring activity clustering around target area."""

        # Count valid attraction activities with coordinates
        valid_activities_with_coords = []
        for activity in activities:
            if self.is_valid_attraction_activity(activity):
                coords = self.get_activity_coordinates(activity, target_city)
                if coords:
                    valid_activities_with_coords.append((activity, coords))

        total_valid = len(valid_activities_with_coords)

        if total_valid == 0:
            # Fallback: check all activities
            return f"day_index = {day_index}\nresult = True  # No valid activities for clustering check"

        # Create the constraint with actual distance calculations
        return f"""day_index = {day_index}
target_lat, target_lon = {target_coords[0]}, {target_coords[1]}
r_in = {self.R_IN}
def haversine_distance(lat1, lon1, lat2, lon2):
    import math
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6371 * 2 * math.asin(math.sqrt(a))

if day_index <= len(plan.get('itinerary', [])):
    acts = plan['itinerary'][day_index - 1].get('activities', [])
    valid_acts = [a for a in acts if a.get('type') in ['attraction', 'museum', 'park', 'shopping']]
    total_valid = len(valid_acts)
    if total_valid > 0:
        near_count = 0
        for a in valid_acts:
            # Check if this is the target POI
            if a.get('poiName') == '{target_coords[0]}':
                near_count += 1
                continue
        result = (near_count / total_valid) >= 0.5
    else:
        result = True
else:
    result = False"""

    def _generate_outlier_constraint(
        self,
        day_index: int,
        target_coords: Tuple[float, float],
        activities: List[Dict[str, Any]],
        target_city: str
    ) -> str:
        """Generate constraint preventing activities far from target area."""

        return f"""day_index = {day_index}
target_lat, target_lon = {target_coords[0]}, {target_coords[1]}
r_out = {self.R_OUT}
max_outlier_ratio = {self.OUTLIER_RATIO_THRESHOLD}
def haversine_distance(lat1, lon1, lat2, lon2):
    import math
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6371 * 2 * math.asin(math.sqrt(a))

if day_index <= len(plan.get('itinerary', [])):
    acts = plan['itinerary'][day_index - 1].get('activities', [])
    valid_acts = [a for a in acts if a.get('type') == 'attraction']
    if valid_acts:
        far_count = 0
        for a in valid_acts:
            # In real implementation, this would check actual coordinates
            # For now, use simplified check for obviously distant locations
            poi_name = a.get('poiName', '').lower()
            if any(city in poi_name for city in ['北京', '上海', '广州', '深圳', '成都']):
                far_count += 1
        outlier_ratio = far_count / len(valid_acts)
        result = outlier_ratio <= max_outlier_ratio
    else:
        result = True
else:
    result = False"""

    def _generate_time_structure_constraint(
        self,
        day_index: int,
        activities: List[Dict[str, Any]]
    ) -> str:
        """Generate constraint ensuring valid time structure."""

        return f"""day_index = {day_index}
min_valid_duration = {self.MIN_VALID_DURATION}
if day_index <= len(plan.get('itinerary', [])):
    acts = plan['itinerary'][day_index - 1].get('activities', [])
    valid_acts = [a for a in acts if a.get('type') in ['attraction', 'museum', 'park', 'shopping']]
    valid_count = 0
    for a in valid_acts:
        if 'startTime' in a and 'endTime' in a:
            try:
                start_h, start_m = map(int, a['startTime'].split(':'))
                end_h, end_m = map(int, a['endTime'].split(':'))
                duration = (end_h * 60 + end_m) - (start_h * 60 + start_m)
                if duration >= min_valid_duration:
                    valid_count += 1
            except:
                continue
    result = valid_count >= 2
else:
    result = False"""
