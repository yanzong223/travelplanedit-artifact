from evaluation.benchmark.levels import evaluate_level3
from evaluation.cascade_analysis import FullGateResult, analyze_cascade


def _plan(activities):
    return {
        "start_city": "南京",
        "target_city": "武汉",
        "itinerary": [{"day": 1, "activities": activities}],
    }


def _flight(flight_id, start_time="06:00", end_time="07:30"):
    return {
        "type": "airplane",
        "FlightID": flight_id,
        "start": "南京禄口国际机场",
        "end": "武汉天河国际机场",
        "start_time": start_time,
        "end_time": end_time,
        "transports": [],
    }


def _train(train_id="G1"):
    return {
        "type": "train",
        "TrainID": train_id,
        "start": "南京南站",
        "end": "武汉站",
        "start_time": "06:00",
        "end_time": "09:00",
        "transports": [],
    }


def _passing_gate(_plan):
    return FullGateResult(
        passed=True,
        components={"plan_validity": True, "edit_correctness": True},
        reason="ok",
    )


def test_flight_id_change_is_an_intercity_service_replacement():
    origin = _plan([_flight("FL717", "00:06", "00:40")])
    edited = _plan([_flight("FL711", "06:47", "07:21")])

    result = evaluate_level3(origin, edited, {"pass": True}, {"pass": True})

    assert result["atomic_counts"]["replace"] == 1
    assert result["atomic_counts"]["change_time"] == 0
    assert result["atomic_counts"]["change_attribute"] == 0


def test_train_to_airplane_is_one_cross_mode_replacement():
    origin = _plan([_train()])
    edited = _plan([_flight("FL141", "11:21", "12:40")])

    result = evaluate_level3(origin, edited, {"pass": True}, {"pass": True})

    assert result["atomic_counts"]["replace"] == 1
    assert result["atomic_counts"]["delete"] == 0
    assert result["atomic_counts"]["insert"] == 0


def test_requested_intercity_mode_replacement_is_direct_target():
    origin = _plan([_train()])
    edited = _plan([_flight("FL141", "11:21", "12:40")])
    constraints = [{
        "id": "edit_logic_0",
        "source": "edit",
        "type": "required_intercity_transport_type",
        "target": {"activity_type": "intercity_transport"},
        "operator": "contains_all",
        "value": ["airplane"],
    }]

    result = analyze_cascade(
        origin,
        edited,
        constraints,
        full_gate_validator=_passing_gate,
    )

    assert result["four_way_total_impact_count"] == 1
    assert result["four_way_direct_target_count"] == 1
    assert result["verified_removable_change_count"] == 0
    assert result["rollback_required_support_change_count"] == 0
