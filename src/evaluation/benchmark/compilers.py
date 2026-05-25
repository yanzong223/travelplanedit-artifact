"""Compile structured logical constraints into executable hard_logic_py snippets."""

from __future__ import annotations

from typing import Iterable

from .models import LogicalConstraintObject


def _python_set_literal(values: Iterable[str]) -> str:
    escaped = [repr(str(item)) for item in values if str(item)]
    return "{" + ", ".join(escaped) + "}"


def _indent(code: str, spaces: int = 2) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line else line for line in code.splitlines())


def _day_loop(day_value: object, body: str, *, day_var: str = "_day") -> str:
    if day_value not in (None, "all"):
        return f"{day_var}={int(day_value)}\n" + body
    return (
        f"{day_var}=1\n"
        f"while {day_var} <= day_count(plan):\n"
        f"{_indent(body)}\n"
        f"  {day_var}+=1"
    )


def compile_logical_constraint(constraint: LogicalConstraintObject) -> str | None:
    params = constraint.params
    target = constraint.target
    operator = constraint.operator or "=="
    value = constraint.value

    if constraint.type == "python_expression":
        code = params.get("code")
        return str(code) if code else None

    if constraint.type == "day_count":
        return f"result=(day_count(plan){operator}{int(value)})"

    if constraint.type == "people_count":
        return f"result=(people_count(plan){operator}{int(value)})"

    if constraint.type == "budget_total":
        return (
            "total_cost=0\n"
            "for activity in allactivities(plan):\n"
            "    total_cost+=activity_cost(activity)\n"
            "    total_cost += innercity_transport_cost(activity_transports(activity))\n"
            f"result=(total_cost{operator}{float(value)})"
        )

    if constraint.type == "ticket_budget_total":
        return (
            "def _ticket_cost(activity):\n"
            "  if 'price' in activity and activity.get('price') is not None:\n"
            "    return activity.get('price', 0)\n"
            "  return activity.get('cost', 0)\n"
            "ticket_cost=0\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity)=='attraction':\n"
            "    ticket_cost+=_ticket_cost(activity)\n"
            f"result=(ticket_cost{operator}{float(value)})"
        )

    if constraint.type == "required_intercity_transport_type":
        relation = "==" if operator == "equals_set" else "<="
        return (
            "intercity_transport_set=set()\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity) in ['train', 'airplane']:\n"
            "    intercity_transport_set.add(intercity_transport_type(activity))\n"
            f"result=({_python_set_literal(value or [])}{relation}intercity_transport_set)"
        )

    if constraint.type == "required_innercity_transport_type":
        relation = "==" if operator == "equals_set" else "<="
        return (
            "innercity_transport_set=set()\n"
            "for activity in allactivities(plan):\n"
            "  if activity_transports(activity)!=[]:\n"
            "    innercity_transport_set.add(innercity_transport_type(activity_transports(activity)))\n"
            f"result=({_python_set_literal(value or [])}{relation}innercity_transport_set)"
        )

    if constraint.type == "required_attraction_name":
        relation = "==" if operator == "equals_set" else "<="
        return (
            "attraction_name_set=set()\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity)=='attraction':\n"
            "    attraction_name_set.add(activity_position(activity))\n"
            f"result=({_python_set_literal(value or [])}{relation}attraction_name_set)"
        )

    if constraint.type == "forbidden_attraction_name":
        return (
            "attraction_name_set=set()\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity)=='attraction':\n"
            "    attraction_name_set.add(activity_position(activity))\n"
            f"result=not bool({_python_set_literal(value or [])} & attraction_name_set)"
        )

    if constraint.type == "required_restaurant_name":
        relation = "==" if operator == "equals_set" else "<="
        return (
            "restaurant_name_set=set()\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n"
            "    restaurant_name_set.add(activity_position(activity))\n"
            f"result=({_python_set_literal(value or [])}{relation}restaurant_name_set)"
        )

    if constraint.type == "required_hotel_name":
        relation = "==" if operator == "equals_set" else "<="
        return (
            "accommodation_name_set=set()\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity)=='accommodation':\n"
            "    accommodation_name_set.add(activity_position(activity))\n"
            f"result=({_python_set_literal(value or [])}{relation}accommodation_name_set)"
        )

    if constraint.type == "required_attraction_type":
        relation = "==" if operator == "equals_set" else "<="
        return (
            "attraction_type_set=set()\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity)=='attraction':\n"
            "    attraction_type_set.add(attraction_type(activity, target_city(plan)))\n"
            f"result=({_python_set_literal(value or [])}{relation}attraction_type_set)"
        )

    if constraint.type == "required_restaurant_type":
        relation = "==" if operator == "equals_set" else "<="
        return (
            "restaurant_type_set=set()\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']:\n"
            "    restaurant_type_set.add(restaurant_type(activity, target_city(plan)))\n"
            f"result=({_python_set_literal(value or [])}{relation}restaurant_type_set)"
        )

    if constraint.type == "required_hotel_feature":
        relation = "==" if operator == "equals_set" else "<="
        return (
            "accommodation_type_set=set()\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity)=='accommodation':\n"
            "    accommodation_type_set.add(accommodation_type(activity, target_city(plan)))\n"
            f"result=({_python_set_literal(value or [])}{relation}accommodation_type_set)"
        )

    if constraint.type == "semantic_type_requirement":
        min_count = int(params.get("min_count", 1) or 1)
        day_value = params.get("day", target.get("day"))
        strict_majority = bool(params.get("strict_majority"))
        accepted_types = [str(item) for item in value] if isinstance(value, (list, tuple, set)) else [str(value)]
        accepted_expr = _python_set_literal(accepted_types)
        if day_value is None:
            code = (
                "_matched=0\n"
                "_total=0\n"
                f"_accepted_types={accepted_expr}\n"
                "for activity in allactivities(plan):\n"
                "  if activity_type(activity)=='attraction':\n"
                "    _total+=1\n"
                "    if attraction_type(activity, target_city(plan)) in _accepted_types: _matched+=1\n"
            )
        else:
            code = (
                "_matched=0\n"
                "_total=0\n"
                f"_accepted_types={accepted_expr}\n"
                f"for activity in dayactivities(plan, {int(day_value)}):\n"
                "  if activity_type(activity)=='attraction':\n"
                "    _total+=1\n"
                "    if attraction_type(activity, target_city(plan)) in _accepted_types: _matched+=1\n"
            )
        if strict_majority:
            code += f"result=(_matched>={min_count} and _total>0 and _matched*2>_total)"
        else:
            code += f"result=(_matched>={min_count})"
        return code

    if constraint.type == "required_room_type":
        return (
            "result=True\n"
            "for activity in allactivities(plan):\n"
            f"  if activity_type(activity)=='accommodation' and room_type(activity)!={int(value)}: result=False"
        )

    if constraint.type == "required_room_count":
        return (
            "result=True\n"
            "for activity in allactivities(plan):\n"
            f"  if activity_type(activity)=='accommodation' and room_count(activity)!={int(value)}: result=False"
        )

    if constraint.type == "ticket_count_match":
        include_metro = params.get("include_metro", True)
        code = (
            "result=True\n"
            "for activity in allactivities(plan):\n"
            f"  if activity_type(activity) in ['attraction', 'airplane', 'train'] and activity_tickets(activity)!={int(value)}: result=False\n"
        )
        if include_metro:
            code += (
                f"  if innercity_transport_type(activity_transports(activity))=='metro' and metro_tickets(activity_transports(activity))!={int(value)}: result=False"
            )
        return code

    if constraint.type == "taxi_car_count_match":
        return (
            "result=True\n"
            "for activity in allactivities(plan):\n"
            f"  if innercity_transport_type(activity_transports(activity))=='taxi' and taxi_cars(activity_transports(activity))!={int(value)}: result=False"
        )

    if constraint.type == "poi_logic":
        poi_names = value or []
        relation = params.get("relation", "conjunction")
        if relation == "conjunction":
            return (
                "poi_name_set=set()\n"
                "for activity in allactivities(plan):\n"
                "  if activity_type(activity) in ['attraction', 'breakfast', 'lunch', 'dinner', 'accommodation']:\n"
                "    poi_name_set.add(activity_position(activity))\n"
                f"result=({_python_set_literal(poi_names)}<=poi_name_set)"
            )
        if relation == "disjunction":
            return (
                "poi_name_set=set()\n"
                "for activity in allactivities(plan):\n"
                "  if activity_type(activity) in ['attraction', 'breakfast', 'lunch', 'dinner', 'accommodation']:\n"
                "    poi_name_set.add(activity_position(activity))\n"
                f"result=bool({_python_set_literal(poi_names)} & poi_name_set)"
            )
        if relation == "negation":
            return (
                "poi_name_set=set()\n"
                "for activity in allactivities(plan):\n"
                "  if activity_type(activity) in ['attraction', 'breakfast', 'lunch', 'dinner', 'accommodation']:\n"
                "    poi_name_set.add(activity_position(activity))\n"
                f"result=not bool({_python_set_literal(poi_names)} & poi_name_set)"
            )
        return None

    if constraint.type == "activity_duration_limit":
        target_name = target.get("poi_name")
        target_type = target.get("activity_type")
        matcher = (
            f"activity_position(activity)=={target_name!r}"
            if target_name
            else f"activity_type(activity)=={target_type!r}"
        )
        return (
            "result=True\n"
            "def _to_min(value):\n"
            "  hour, minute = value.split(':')[:2]\n"
            "  return int(hour) * 60 + int(minute)\n"
            "for activity in allactivities(plan):\n"
            f"  if {matcher}:\n"
            "    duration = _to_min(activity_end_time(activity)) - _to_min(activity_start_time(activity))\n"
            f"    if not (duration{operator}{int(value)}): result=False"
        )

    if constraint.type == "activity_budget_limit":
        metric = params.get("metric", "activity_cost")
        target_name = target.get("poi_name")
        target_type = target.get("activity_type")
        matcher = []
        if target_name:
            matcher.append(f"activity_position(activity)=={target_name!r}")
        if target_type:
            if target_type == "meal":
                matcher.append("activity_type(activity) in ['breakfast', 'lunch', 'dinner']")
            else:
                matcher.append(f"activity_type(activity)=={target_type!r}")
        cond = " and ".join(matcher) if matcher else "True"
        if metric == "avg_cost_per_person_per_night":
            return (
                "hotel_cost=0\n"
                "for activity in allactivities(plan):\n"
                f"  if {cond}: hotel_cost+=activity_cost(activity)\n"
                f"result=(hotel_cost/people_count(plan)/max(day_count(plan)-1,1){operator}{float(value)})"
            )
        if metric == "avg_cost_per_meal_per_person":
            return (
                "food_cost,food_count=0,0\n"
                "for activity in allactivities(plan):\n"
                f"  if {cond}:\n"
                "    food_cost+=activity_cost(activity)\n"
                "    food_count+=1\n"
                f"result=((food_cost/max(food_count,1)/people_count(plan)){operator}{float(value)})"
            )
        return (
            "budget_cost=0\n"
            "for activity in allactivities(plan):\n"
            f"  if {cond}: budget_cost+=activity_cost(activity)\n"
            f"result=(budget_cost{operator}{float(value)})"
        )

    if constraint.type == "day_end_time_limit":
        target_day = target.get("day")
        body = (
            "_activities = dayactivities(plan, _day)\n"
            "if _activities == []:\n"
            "  result=False\n"
            "else:\n"
            "  _last_end=None\n"
            "  for activity in _activities:\n"
            "    if activity_end_time(activity):\n"
            "      _candidate=activity_end_time(activity)\n"
            "      if _last_end is None or _candidate > _last_end:\n"
            "        _last_end=_candidate\n"
            "  if _last_end is None:\n"
            "    result=False\n"
            f"  elif not (_last_end{operator}{value!r}):\n"
            "    result=False"
        )
        return "result=True\n" + _day_loop(target_day, body)

    if constraint.type == "city_split_requirement":
        expected_cities = [str(item) for item in value or [] if str(item)]
        if len(expected_cities) < 2:
            return None
        supported_cities = [str(item) for item in params.get("supported_cities", []) if str(item)]
        return (
            f"_expected={_python_set_literal(expected_cities)}\n"
            f"_supported={_python_set_literal(supported_cities)}\n"
            "_seen=set()\n"
            "if target_city(plan):\n"
            "  _seen.add(target_city(plan))\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity) in ['train','airplane']:\n"
            "    for _endpoint in [intercity_transport_origin(activity), intercity_transport_destination(activity)]:\n"
            "      if not _endpoint:\n"
            "        continue\n"
            "      for _city in _supported:\n"
            "        if _city in str(_endpoint):\n"
            "          _seen.add(_city)\n"
            "result=(_expected<=_seen)"
        )

    if constraint.type == "poi_time_window":
        poi_name = target.get("poi_name")
        day_value = params.get("day")
        window_map = {
            "morning": ("06:00", "12:00"),
            "afternoon": ("12:00", "18:00"),
            "evening": ("18:00", "22:00"),
            "night": ("22:00", "24:00"),
        }
        if value not in window_map or not poi_name:
            return None
        low, high = window_map[str(value)]
        body = (
            "for activity in dayactivities(plan, _day):\n"
            f"  if activity_position(activity)=={poi_name!r}:\n"
            "    _matched=True\n"
            "    _start=activity_start_time(activity)\n"
            "    _end=activity_end_time(activity)\n"
            f"    if not ({low!r} <= _start and _end <= {high!r}): result=False"
        )
        return (
            "result=True\n"
            "_matched=False\n"
            f"{_day_loop(day_value, body)}\n"
            "if not _matched: result=False"
        )

    if constraint.type == "poi_clock_time_window":
        poi_name = target.get("poi_name")
        day_value = params.get("day", target.get("day"))
        mode = str(params.get("mode", "") or "").strip().lower()
        if mode not in {"after", "before", "between"}:
            if operator in {">=", ">"}:
                mode = "after"
            elif operator in {"<=", "<"}:
                mode = "before"
            elif operator in {"between", "in"} and isinstance(value, (list, tuple)) and len(value) == 2:
                mode = "between"
        if not poi_name or mode not in {"after", "before", "between"}:
            return None
        if mode == "between":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                return None
            low = str(value[0])
            high = str(value[1])
            predicate = f"(_to_min(_start) >= _to_min({low!r}) and _to_min(_end) <= _to_min({high!r}))"
        elif mode == "after":
            predicate = f"(_to_min(_start) >= _to_min({str(value)!r}))"
        else:
            predicate = f"(_to_min(_end) <= _to_min({str(value)!r}))"
        body = (
            "for activity in dayactivities(plan, _day):\n"
            f"  if activity_position(activity)=={poi_name!r}:\n"
            "    _matched=True\n"
            "    _start=activity_start_time(activity)\n"
            "    _end=activity_end_time(activity)\n"
            f"    if not {predicate}: result=False"
        )
        return (
            "result=True\n"
            "_matched=False\n"
            "def _to_min(value):\n"
            "  hour, minute = str(value).split(':')[:2]\n"
            "  return int(hour) * 60 + int(minute)\n"
            f"{_day_loop(day_value, body)}\n"
            "if not _matched: result=False"
        )

    if constraint.type == "daily_poi_cap":
        count_types = [str(item) for item in params.get("count_types", ["attraction"])]
        type_expr = repr(count_types)
        target_day = target.get("day")
        body = (
            "_count=0\n"
            "for activity in dayactivities(plan, _day):\n"
            "  if activity_type(activity) in _types:\n"
            "    _count+=1\n"
            f"if not (_count{operator}{int(value)}): result=False"
        )
        return (
            "result=True\n"
            "_types=set(" + type_expr + ")\n"
            "if 'restaurant' in _types:\n"
            "  _types.remove('restaurant')\n"
            "  _types.update(['breakfast','lunch','dinner'])\n"
            "if 'meal' in _types:\n"
            "  _types.remove('meal')\n"
            "  _types.update(['breakfast','lunch','dinner'])\n"
            f"{_day_loop(target_day, body)}"
        )

    if constraint.type == "adjacent_travel_time_cap":
        activity_types = [str(item) for item in target.get("activity_types", params.get("activity_types", ["attraction"]))]
        type_expr = repr(activity_types)
        target_day = params.get("day", "all")
        body = (
            "_filtered=[]\n"
            "for activity in dayactivities(plan, _day):\n"
            "  if activity_type(activity) in _types:\n"
            "    _filtered.append(activity)\n"
            "_prev=None\n"
            "for _curr in _filtered:\n"
            "  if _prev is not None:\n"
            "    _minutes=innercity_transport_time(activity_transports(_curr))\n"
            f"    if not (_minutes{operator}{int(value)}): result=False\n"
            "  _prev=_curr"
        )
        return (
            "result=True\n"
            "_types=set(" + type_expr + ")\n"
            "if 'restaurant' in _types:\n"
            "  _types.remove('restaurant')\n"
            "  _types.update(['breakfast','lunch','dinner'])\n"
            "if 'meal' in _types:\n"
            "  _types.remove('meal')\n"
            "  _types.update(['breakfast','lunch','dinner'])\n"
            f"{_day_loop(target_day, body)}"
        )

    if constraint.type == "adjacent_travel_distance_cap":
        activity_types = [str(item) for item in target.get("activity_types", params.get("activity_types", ["attraction"]))]
        type_expr = repr(activity_types)
        target_day = params.get("day", "all")
        body = (
            "_filtered=[]\n"
            "for activity in dayactivities(plan, _day):\n"
            "  if activity_type(activity) in _types:\n"
            "    _filtered.append(activity)\n"
            "_prev=None\n"
            "for _curr in _filtered:\n"
            "  if _prev is not None:\n"
            "    _distance=innercity_transport_distance(activity_transports(_curr))\n"
            f"    if not (_distance{operator}{float(value)}): result=False\n"
            "  _prev=_curr"
        )
        return (
            "result=True\n"
            "_types=set(" + type_expr + ")\n"
            "if 'restaurant' in _types:\n"
            "  _types.remove('restaurant')\n"
            "  _types.update(['breakfast','lunch','dinner'])\n"
            "if 'meal' in _types:\n"
            "  _types.remove('meal')\n"
            "  _types.update(['breakfast','lunch','dinner'])\n"
            f"{_day_loop(target_day, body)}"
        )

    if constraint.type == "anchor_neighbor_commute_distance_cap":
        anchor_name = target.get("anchor_name")
        if not anchor_name:
            return None
        target_day = params.get("day", target.get("day"))
        body = (
            "_activities = dayactivities(plan, _day)\n"
            "_anchor_present=False\n"
            "for activity in _activities:\n"
            f"  if activity_position(activity)=={anchor_name!r}:\n"
            "    _anchor_present=True\n"
            "if _anchor_present:\n"
            "  for activity in _activities:\n"
            "    if activity_type(activity) in ['train', 'airplane']:\n"
            "      continue\n"
            "    if activity_position(activity)=='' and activity_type(activity) not in ['attraction', 'accommodation', 'breakfast', 'lunch', 'dinner']:\n"
            "      continue\n"
            "    _matched=True\n"
            f"    _distance = 0 if activity_position(activity)=={anchor_name!r} else poi_distance(target_city(plan), {anchor_name!r}, activity_position(activity), activity_start_time(activity), 'walk')\n"
            f"    if not (_distance{operator}{float(value)}): result=False\n"
        )
        return (
            "result=True\n"
            "_matched=False\n"
            f"{_day_loop(target_day, body)}\n"
            "if not _matched: result=False"
        )

    if constraint.type == "poi_day_binding":
        poi_name = target.get("poi_name")
        if not poi_name:
            return None
        return (
            "result=False\n"
            f"for activity in dayactivities(plan, {int(value)}):\n"
            f"  if activity_position(activity)=={poi_name!r}: result=True"
        )

    if constraint.type == "poi_order_constraint":
        first_poi = target.get("first_poi")
        second_poi = target.get("second_poi")
        if not first_poi or not second_poi:
            return None
        body = (
            "_idx=0\n"
            "for activity in dayactivities(plan, _day):\n"
            f"  if _first_day is None and activity_position(activity)=={first_poi!r}:\n"
            "    _first_day=_day\n"
            "    _first_idx=_idx\n"
            f"  if _second_day is None and activity_position(activity)=={second_poi!r}:\n"
            "    _second_day=_day\n"
            "    _second_idx=_idx\n"
            "  _idx+=1"
        )
        return (
            "_first_day=None\n"
            "_first_idx=0\n"
            "_second_day=None\n"
            "_second_idx=0\n"
            f"{_day_loop(None, body)}\n"
            "result=(\n"
            "  _first_day is not None and _second_day is not None and (\n"
            "    _first_day < _second_day or (_first_day == _second_day and _first_idx < _second_idx)\n"
            "  )\n"
            ")"
        )

    if constraint.type == "pair_same_day_no_overlap":
        first_poi = target.get("first_poi")
        second_poi = target.get("second_poi")
        if not first_poi or not second_poi:
            return None
        return (
            "result=False\n"
            "_first=None\n"
            "_second=None\n"
            "def _to_min(value):\n"
            "  hour, minute = value.split(':')[:2]\n"
            "  return int(hour) * 60 + int(minute)\n"
            f"for activity in dayactivities(plan, {int(value)}):\n"
            f"  if _first is None and activity_position(activity)=={first_poi!r}: _first=activity\n"
            f"  if _second is None and activity_position(activity)=={second_poi!r}: _second=activity\n"
            "if _first is not None and _second is not None:\n"
            "  _first_start=_to_min(activity_start_time(_first))\n"
            "  _first_end=_to_min(activity_end_time(_first))\n"
            "  _second_start=_to_min(activity_start_time(_second))\n"
            "  _second_end=_to_min(activity_end_time(_second))\n"
            "  result=(_first_end <= _second_start or _second_end <= _first_start)"
        )

    if constraint.type == "pair_time_window_no_overlap":
        first_poi = target.get("first_poi")
        second_poi = target.get("second_poi")
        window_map = {
            "morning": (6 * 60, 12 * 60),
            "afternoon": (12 * 60, 18 * 60),
            "evening": (18 * 60, 22 * 60),
            "night": (22 * 60, 24 * 60),
        }
        if not first_poi or not second_poi or value not in window_map:
            return None
        low, high = window_map[str(value)]
        body = (
            "_first=None\n"
            "_second=None\n"
            "for activity in dayactivities(plan, _day):\n"
            f"  if _first is None and activity_position(activity)=={first_poi!r}: _first=activity\n"
            f"  if _second is None and activity_position(activity)=={second_poi!r}: _second=activity\n"
            "if _first is not None and _second is not None:\n"
            "  _first_start=_to_min(activity_start_time(_first))\n"
            "  _first_end=_to_min(activity_end_time(_first))\n"
            "  _second_start=_to_min(activity_start_time(_second))\n"
            "  _second_end=_to_min(activity_end_time(_second))\n"
            f"  _in_window=({low} <= _first_start and _first_end <= {high} and {low} <= _second_start and _second_end <= {high})\n"
            "  if _in_window and (_first_end <= _second_start or _second_end <= _first_start):\n"
            "    result=True"
        )
        return (
            "result=False\n"
            "def _to_min(value):\n"
            "  hour, minute = value.split(':')[:2]\n"
            "  return int(hour) * 60 + int(minute)\n"
            f"{_day_loop(None, body)}"
        )

    if constraint.type == "pairwise_transport_mode_distance_cap":
        first_poi = target.get("first_poi")
        second_poi = target.get("second_poi")
        mode = target.get("mode")
        if not first_poi or not second_poi or not mode:
            return None
        mode_check = f"innercity_transport_type(activity_transports(_curr))=={mode!r}"
        body = (
            "_activities = dayactivities(plan, _day)\n"
            "_prev=None\n"
            "for _curr in _activities:\n"
            "  if _prev is not None:\n"
            f"    _names=set([activity_position(_prev), activity_position(_curr)])\n"
            f"    if _names==set([{first_poi!r}, {second_poi!r}]):\n"
            "      _matched=True\n"
            f"      if {mode_check} and innercity_transport_distance(activity_transports(_curr)){operator}{float(value)}:\n"
            "        result=True\n"
            "  _prev=_curr"
        )
        return (
            "result=False\n"
            "_matched=False\n"
            f"{_day_loop(None, body)}"
        )

    if constraint.type == "poi_inbound_travel_time_cap":
        poi_name = target.get("poi_name")
        if not poi_name:
            return None
        return (
            "result=True\n"
            "_matched=False\n"
            "def _transport_min(transports):\n"
            "  _total=0\n"
            "  _found=False\n"
            "  for item in transports:\n"
            "    _start=item.get('start_time','')\n"
            "    _end=item.get('end_time','')\n"
            "    if ':' not in str(_start) or ':' not in str(_end):\n"
            "      continue\n"
            "    _sh,_sm=str(_start).split(':')[:2]\n"
            "    _eh,_em=str(_end).split(':')[:2]\n"
            "    _total += int(_eh)*60 + int(_em) - int(_sh)*60 - int(_sm)\n"
            "    _found=True\n"
            "  return _total if _found else None\n"
            "for activity in allactivities(plan):\n"
            f"  if activity_position(activity)=={poi_name!r}:\n"
            "    _matched=True\n"
            "    _minutes=_transport_min(activity_transports(activity))\n"
            "    if _minutes is None:\n"
            "      result=False\n"
            f"    elif not (_minutes{operator}{int(value)}): result=False\n"
            "if not _matched: result=False"
        )

    if constraint.type == "nearby_meal_requirement":
        anchor_poi = target.get("anchor_poi") or target.get("poi_name")
        if not anchor_poi:
            return None
        selected_meal = str(params.get("selected_meal_name") or "").strip()
        max_distance = float(params.get("max_distance_km") or 3.0)
        meal_guard = (
            f" and activity_position(activity)=={selected_meal!r}"
            if selected_meal
            else ""
        )
        body = (
            f"compiler_semantics_limited={{'distance_km': {max_distance!r}}}\n"
            "_anchor_found=False\n"
            "_meal_found=False\n"
            "for activity in dayactivities(plan, _day):\n"
            f"  if activity_type(activity)=='attraction' and activity_position(activity)=={anchor_poi!r}: _anchor_found=True\n"
            "for activity in dayactivities(plan, _day):\n"
            f"  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']{meal_guard}: _meal_found=True\n"
            "if _anchor_found and _meal_found:\n"
            "  result=True"
        )
        return "result=False\n" + _day_loop(None, body)

    if constraint.type == "anchor_bundle_budget_limit":
        anchor_poi = target.get("anchor_poi") or target.get("poi_name")
        if not anchor_poi:
            return None
        selected_meal = str(params.get("selected_meal_name") or "").strip()
        max_distance = float(params.get("max_distance_km") or 3.0)
        meal_guard = (
            f" and activity_position(activity)=={selected_meal!r}"
            if selected_meal
            else ""
        )
        body = (
            f"compiler_semantics_limited={{'distance_km': {max_distance!r}}}\n"
            "_anchor_cost=None\n"
            "_meal_costs=[]\n"
            "for activity in dayactivities(plan, _day):\n"
            f"  if activity_type(activity)=='attraction' and activity_position(activity)=={anchor_poi!r}:\n"
            "    _cost=activity_cost(activity)\n"
            "    if _anchor_cost is None or _cost<_anchor_cost:\n"
            "      _anchor_cost=_cost\n"
            f"  if activity_type(activity) in ['breakfast', 'lunch', 'dinner']{meal_guard}:\n"
            "    _meal_costs.append(activity_cost(activity))\n"
            "if _anchor_cost is not None and _meal_costs:\n"
            "  for _meal_cost in _meal_costs:\n"
            "    _bundle_cost=_anchor_cost+_meal_cost\n"
            f"    if _bundle_cost{operator}{float(value)}:\n"
            "      result=True"
        )
        return "result=False\n" + _day_loop(None, body)

    if constraint.type == "ticket_price_cap_or_substitute":
        poi_name = target.get("poi_name")
        activity_type = target.get("activity_type")
        if not poi_name or not activity_type:
            return None
        if activity_type == "meal":
            type_guard = "activity_type(activity) in ['breakfast', 'lunch', 'dinner']"
        else:
            type_guard = f"activity_type(activity)=={activity_type!r}"
        day_value = params.get("day")
        target_eligible = bool(params.get("target_eligible", False))
        if target_eligible:
            body = (
                "for activity in dayactivities(plan, _day):\n"
                f"  if {type_guard} and activity_position(activity)=={poi_name!r}: result=True"
            )
            return (
                "result=False\n"
                f"{_day_loop(day_value, body)}"
            )
        candidate_names = [str(item) for item in params.get("candidate_names", []) if str(item)]
        if not candidate_names:
            return None
        body = (
            "for activity in dayactivities(plan, _day):\n"
            f"  if {type_guard} and activity_position(activity) in _candidate_names: result=True"
        )
        return (
            "result=False\n"
            f"_candidate_names={_python_set_literal(candidate_names)}\n"
            f"{_day_loop(day_value, body)}"
        )

    if constraint.type == "transport_time_window":
        leg = params.get("leg", "outbound")
        field = params.get("field", "start_time")
        if leg in {"return", "back"}:
            activity_filter = (
                "intercity_transport_origin(activity) == target_city(plan)"
                if field == "start_time"
                else "intercity_transport_destination(activity) == start_city(plan)"
            )
            choose_guard = "_chosen is None or _candidate_idx >= _chosen_idx"
        else:
            activity_filter = (
                "intercity_transport_origin(activity) == start_city(plan)"
                if field == "start_time"
                else "intercity_transport_destination(activity) == target_city(plan)"
            )
            choose_guard = "_chosen is None"
        getter = "activity_start_time(_chosen)" if field == "start_time" else "activity_end_time(_chosen)"
        variable_name = f"{leg}_{field}"
        return (
            f"{variable_name} = ''\n"
            "_chosen = None\n"
            "_chosen_idx = -1\n"
            "_candidate_idx = -1\n"
            "for activity in allactivities(plan):\n"
            "  if activity_type(activity) in ['train','airplane']:\n"
            "    _candidate_idx += 1\n"
            f"    if {activity_filter} and {choose_guard}:\n"
            "      _chosen = activity\n"
            "      _chosen_idx = _candidate_idx\n"
            f"if _chosen is not None: {variable_name} = {getter}\n"
            f"result=({variable_name}{operator}{value!r})"
        )

    return None


def compile_logical_constraints(constraints: list[LogicalConstraintObject]) -> list[str]:
    compiled: list[str] = []
    for constraint in constraints:
        code = compile_logical_constraint(constraint)
        if code:
            compiled.append(code)
    return compiled
