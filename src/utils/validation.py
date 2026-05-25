"""
Data validation utilities for TPE system.

Provides robust data validation and error handling for various data types.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import ValidationError

from core.models.base import ValidationResult
from core.models.enums import ConstraintType, VariableType


class ValidationError(Exception):
    """Custom validation exception."""

    pass


class BaseValidator:
    """Base class for all validators."""

    def __init__(self, name: str):
        self.name = name

    def validate(self, data: Any) -> ValidationResult:
        """Validate the given data."""
        raise NotImplementedError("Subclasses must implement validate method")


class UUIDValidator(BaseValidator):
    """Validator for UUID strings."""

    UUID_PATTERN = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
    )

    def __init__(self):
        super().__init__("UUID")

    def validate(self, data: Any) -> ValidationResult:
        """Validate UUID format."""
        result = ValidationResult(is_valid=True)

        if not isinstance(data, str):
            result.add_error("UUID must be a string")
            return result

        if not self.UUID_PATTERN.match(data):
            result.add_error(f"Invalid UUID format: {data}")

        return result


class DateTimeValidator(BaseValidator):
    """Validator for ISO 8601 datetime strings."""

    def __init__(self):
        super().__init__("DateTime")

    def validate(self, data: Any) -> ValidationResult:
        """Validate datetime format."""
        result = ValidationResult(is_valid=True)

        if not isinstance(data, str):
            result.add_error("DateTime must be a string")
            return result

        try:
            # Try parsing with timezone
            if data.endswith("Z"):
                datetime.fromisoformat(data.replace("Z", "+00:00"))
            else:
                datetime.fromisoformat(data)
        except ValueError:
            result.add_error(f"Invalid datetime format: {data}")

        return result


# Email validator removed - not needed for travel plan editing system


class FlightDataValidator(BaseValidator):
    """Validator for flight data."""

    def __init__(self):
        super().__init__("FlightData")

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate flight data structure."""
        result = ValidationResult(is_valid=True)

        if not isinstance(data, dict):
            result.add_error("Flight data must be a dictionary")
            return result

        # Required fields
        required_fields = ["from", "to", "departure", "arrival"]
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")

        # Validate airport codes
        for field in ["from", "to"]:
            if field in data:
                airport_code = data[field]
                if not isinstance(airport_code, str) or len(airport_code) not in [3, 4]:
                    result.add_error(
                        f"Invalid airport code for {field}: {airport_code}"
                    )

        # Validate datetime fields
        datetime_validator = DateTimeValidator()
        for field in ["departure", "arrival"]:
            if field in data:
                datetime_result = datetime_validator.validate(data[field])
                if not datetime_result.is_valid:
                    result.add_error(
                        f"Invalid {field} datetime: {datetime_result.errors}"
                    )

        # Validate departure before arrival
        if "departure" in data and "arrival" in data:
            try:
                departure = datetime.fromisoformat(
                    data["departure"].replace("Z", "+00:00")
                )
                arrival = datetime.fromisoformat(data["arrival"].replace("Z", "+00:00"))
                if departure >= arrival:
                    result.add_error("Departure time must be before arrival time")
            except ValueError:
                # Already caught by datetime validation
                pass

        return result


class HotelDataValidator(BaseValidator):
    """Validator for hotel data."""

    def __init__(self):
        super().__init__("HotelData")

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate hotel data structure."""
        result = ValidationResult(is_valid=True)

        if not isinstance(data, dict):
            result.add_error("Hotel data must be a dictionary")
            return result

        # Required fields
        required_fields = ["name", "location", "check_in", "check_out"]
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")

        # Validate name
        if "name" in data:
            name = data["name"]
            if not isinstance(name, str) or len(name.strip()) == 0:
                result.add_error("Hotel name must be a non-empty string")

        # Validate location
        if "location" in data:
            location = data["location"]
            if not isinstance(location, str) or len(location.strip()) == 0:
                result.add_error("Hotel location must be a non-empty string")

        # Validate datetime fields
        datetime_validator = DateTimeValidator()
        for field in ["check_in", "check_out"]:
            if field in data:
                datetime_result = datetime_validator.validate(data[field])
                if not datetime_result.is_valid:
                    result.add_error(
                        f"Invalid {field} datetime: {datetime_result.errors}"
                    )

        # Validate check_in before check_out
        if "check_in" in data and "check_out" in data:
            try:
                check_in = datetime.fromisoformat(
                    data["check_in"].replace("Z", "+00:00")
                )
                check_out = datetime.fromisoformat(
                    data["check_out"].replace("Z", "+00:00")
                )
                if check_in >= check_out:
                    result.add_error("Check-in time must be before check-out time")
            except ValueError:
                # Already caught by datetime validation
                pass

        return result


class ActivityDataValidator(BaseValidator):
    """Validator for activity data."""

    def __init__(self):
        super().__init__("ActivityData")

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate activity data structure."""
        result = ValidationResult(is_valid=True)

        if not isinstance(data, dict):
            result.add_error("Activity data must be a dictionary")
            return result

        # Required fields
        required_fields = ["name", "location", "start_time", "end_time"]
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")

        # Validate name
        if "name" in data:
            name = data["name"]
            if not isinstance(name, str) or len(name.strip()) == 0:
                result.add_error("Activity name must be a non-empty string")

        # Validate location
        if "location" in data:
            location = data["location"]
            if not isinstance(location, str) or len(location.strip()) == 0:
                result.add_error("Activity location must be a non-empty string")

        # Validate datetime fields
        datetime_validator = DateTimeValidator()
        for field in ["start_time", "end_time"]:
            if field in data:
                datetime_result = datetime_validator.validate(data[field])
                if not datetime_result.is_valid:
                    result.add_error(
                        f"Invalid {field} datetime: {datetime_result.errors}"
                    )

        # Validate start_time before end_time
        if "start_time" in data and "end_time" in data:
            try:
                start_time = datetime.fromisoformat(
                    data["start_time"].replace("Z", "+00:00")
                )
                end_time = datetime.fromisoformat(
                    data["end_time"].replace("Z", "+00:00")
                )
                if start_time >= end_time:
                    result.add_error("Start time must be before end time")
            except ValueError:
                # Already caught by datetime validation
                pass

        # Validate duration (reasonable limits)
        if "start_time" in data and "end_time" in data:
            try:
                start_time = datetime.fromisoformat(
                    data["start_time"].replace("Z", "+00:00")
                )
                end_time = datetime.fromisoformat(
                    data["end_time"].replace("Z", "+00:00")
                )
                duration_hours = (end_time - start_time).total_seconds() / 3600

                if duration_hours > 24:  # Activities shouldn't exceed 24 hours
                    result.add_warning("Activity duration exceeds 24 hours")
                elif (
                    duration_hours < 0.25
                ):  # Activities shouldn't be less than 15 minutes
                    result.add_warning("Activity duration is very short (< 15 minutes)")

            except ValueError:
                pass

        return result


class VariableNodeValidator(BaseValidator):
    """Validator for variable nodes."""

    def __init__(self):
        super().__init__("VariableNode")

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate variable node structure."""
        result = ValidationResult(is_valid=True)

        if not isinstance(data, dict):
            result.add_error("Variable node must be a dictionary")
            return result

        # Validate required fields
        required_fields = ["id", "type", "data"]
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")

        # Validate ID
        if "id" in data:
            uuid_validator = UUIDValidator()
            uuid_result = uuid_validator.validate(data["id"])
            if not uuid_result.is_valid:
                result.add_error(f"Invalid ID: {uuid_result.errors}")

        # Validate type
        if "type" in data:
            var_type = data["type"]
            if var_type not in [t.value for t in VariableType]:
                result.add_error(f"Invalid variable type: {var_type}")

        # Validate data
        if "data" in data:
            if not isinstance(data["data"], dict):
                result.add_error("Variable data must be a dictionary")
            else:
                # Validate data based on type
                if "type" in data:
                    self._validate_variable_data_by_type(
                        data["type"], data["data"], result
                    )

        # Validate constraints
        if "constraints" in data:
            constraints = data["constraints"]
            if not isinstance(constraints, list):
                result.add_error("Constraints must be a list")
            else:
                uuid_validator = UUIDValidator()
                for constraint_id in constraints:
                    constraint_result = uuid_validator.validate(constraint_id)
                    if not constraint_result.is_valid:
                        result.add_error(
                            f"Invalid constraint ID: {constraint_result.errors}"
                        )

        return result

    def _validate_variable_data_by_type(
        self, var_type: str, data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate variable data based on its type."""
        if var_type == VariableType.FLIGHT:
            validator = FlightDataValidator()
            flight_result = validator.validate(data)
            result.errors.extend(flight_result.errors)
            result.warnings.extend(flight_result.warnings)
            if not flight_result.is_valid:
                result.is_valid = False

        elif var_type == VariableType.HOTEL:
            validator = HotelDataValidator()
            hotel_result = validator.validate(data)
            result.errors.extend(hotel_result.errors)
            result.warnings.extend(hotel_result.warnings)
            if not hotel_result.is_valid:
                result.is_valid = False

        elif var_type == VariableType.ACTIVITY:
            validator = ActivityDataValidator()
            activity_result = validator.validate(data)
            result.errors.extend(activity_result.errors)
            result.warnings.extend(activity_result.warnings)
            if not activity_result.is_valid:
                result.is_valid = False


class ConstraintNodeValidator(BaseValidator):
    """Validator for constraint nodes."""

    def __init__(self):
        super().__init__("ConstraintNode")

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate constraint node structure."""
        result = ValidationResult(is_valid=True)

        if not isinstance(data, dict):
            result.add_error("Constraint node must be a dictionary")
            return result

        # Validate required fields
        required_fields = ["id", "type", "variables", "weight"]
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")

        # Validate ID
        if "id" in data:
            uuid_validator = UUIDValidator()
            uuid_result = uuid_validator.validate(data["id"])
            if not uuid_result.is_valid:
                result.add_error(f"Invalid ID: {uuid_result.errors}")

        # Validate type
        if "type" in data:
            constraint_type = data["type"]
            if constraint_type not in [t.value for t in ConstraintType]:
                result.add_error(f"Invalid constraint type: {constraint_type}")

        # Validate variables
        if "variables" in data:
            variables = data["variables"]
            if not isinstance(variables, list):
                result.add_error("Variables must be a list")
            elif len(variables) == 0:
                result.add_error("Variables list cannot be empty")
            else:
                uuid_validator = UUIDValidator()
                for var_id in variables:
                    var_result = uuid_validator.validate(var_id)
                    if not var_result.is_valid:
                        result.add_error(f"Invalid variable ID: {var_result.errors}")

        # Validate weight
        if "weight" in data:
            weight = data["weight"]
            if not isinstance(weight, (int, float)) or weight < 0:
                result.add_error("Weight must be a non-negative number")

        # Validate parameters
        if "parameters" in data:
            parameters = data["parameters"]
            if not isinstance(parameters, dict):
                result.add_error("Parameters must be a dictionary")

        return result


class FactorGraphValidator(BaseValidator):
    """Validator for factor graphs."""

    def __init__(self):
        super().__init__("FactorGraph")

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Validate factor graph structure."""
        result = ValidationResult(is_valid=True)

        if not isinstance(data, dict):
            result.add_error("Factor graph must be a dictionary")
            return result

        # Validate required fields
        required_fields = ["variables", "constraints", "edges"]
        for field in required_fields:
            if field not in data:
                result.add_error(f"Missing required field: {field}")

        # Validate variables
        if "variables" in data:
            variables = data["variables"]
            if not isinstance(variables, dict):
                result.add_error("Variables must be a dictionary")
            else:
                variable_validator = VariableNodeValidator()
                for var_id, var_data in variables.items():
                    if var_data.get("id") != var_id:
                        result.add_error(
                            f"Variable ID mismatch: {var_id} != {var_data.get('id')}"
                        )

                    var_result = variable_validator.validate(var_data)
                    result.errors.extend(var_result.errors)
                    result.warnings.extend(var_result.warnings)
                    if not var_result.is_valid:
                        result.is_valid = False

        # Validate constraints
        if "constraints" in data:
            constraints = data["constraints"]
            if not isinstance(constraints, dict):
                result.add_error("Constraints must be a dictionary")
            else:
                constraint_validator = ConstraintNodeValidator()
                for con_id, con_data in constraints.items():
                    if con_data.get("id") != con_id:
                        result.add_error(
                            f"Constraint ID mismatch: {con_id} != {con_data.get('id')}"
                        )

                    con_result = constraint_validator.validate(con_data)
                    result.errors.extend(con_result.errors)
                    result.warnings.extend(con_result.warnings)
                    if not con_result.is_valid:
                        result.is_valid = False

        # Validate edges
        if "edges" in data:
            edges = data["edges"]
            if not isinstance(edges, list):
                result.add_error("Edges must be a list")
            else:
                uuid_validator = UUIDValidator()
                for edge in edges:
                    if not isinstance(edge, dict):
                        result.add_error("Edge must be a dictionary")
                        continue

                    # Validate edge structure
                    edge_required = ["source", "target", "type"]
                    for field in edge_required:
                        if field not in edge:
                            result.add_error(f"Edge missing required field: {field}")

                    # Validate source and target are valid UUIDs
                    for field in ["source", "target"]:
                        if field in edge:
                            uuid_result = uuid_validator.validate(edge[field])
                            if not uuid_result.is_valid:
                                result.add_error(
                                    f"Invalid {field} in edge: {uuid_result.errors}"
                                )

        # Validate graph connectivity
        if result.is_valid and "variables" in data and "constraints" in data:
            self._validate_graph_connectivity(data, result)

        return result

    def _validate_graph_connectivity(
        self, data: Dict[str, Any], result: ValidationResult
    ) -> None:
        """Validate that the graph is properly connected."""
        variables = data.get("variables", {})
        constraints = data.get("constraints", {})

        if not variables:
            return  # Empty graph is trivially connected

        variable_ids = set(variables.keys())
        constraint_variables = set()

        for constraint_data in constraints.values():
            if "variables" in constraint_data:
                constraint_variables.update(constraint_data["variables"])

        # Check that all variables referenced in constraints exist
        missing_vars = constraint_variables - variable_ids
        if missing_vars:
            result.add_error(
                f"Constraints reference non-existent variables: {missing_vars}"
            )

        # Check connectivity (simplified - just check if any variable is connected)
        if constraint_variables and not variable_ids.intersection(constraint_variables):
            result.add_warning("No variables are connected to any constraints")


class CompositeValidator(BaseValidator):
    """Validator that combines multiple validators."""

    def __init__(self, name: str, validators: List[BaseValidator]):
        super().__init__(name)
        self.validators = validators

    def validate(self, data: Any) -> ValidationResult:
        """Run all validators and combine results."""
        result = ValidationResult(is_valid=True)

        for validator in self.validators:
            validator_result = validator.validate(data)
            result.errors.extend(validator_result.errors)
            result.warnings.extend(validator_result.warnings)
            if not validator_result.is_valid:
                result.is_valid = False

        return result


class ValidationEngine:
    """Main validation engine."""

    def __init__(self):
        self._validators = {
            "uuid": UUIDValidator(),
            "datetime": DateTimeValidator(),
            # "email": EmailValidator(),  # Removed - not needed for travel plan editing
            "flight": FlightDataValidator(),
            "hotel": HotelDataValidator(),
            "activity": ActivityDataValidator(),
            "variable_node": VariableNodeValidator(),
            "constraint_node": ConstraintNodeValidator(),
            "factor_graph": FactorGraphValidator(),
        }

    def get_validator(self, validator_name: str) -> Optional[BaseValidator]:
        """Get a validator by name."""
        return self._validators.get(validator_name)

    def register_validator(self, validator_name: str, validator: BaseValidator) -> None:
        """Register a new validator."""
        self._validators[validator_name] = validator

    def validate_pydantic_model(self, model: Any) -> ValidationResult:
        """Validate a Pydantic model."""
        try:
            # Trigger Pydantic validation
            _ = model.model_dump()
            return ValidationResult(is_valid=True)
        except ValidationError as e:
            result = ValidationResult(is_valid=False)
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error["loc"])
                result.add_error(f"Field '{field_path}': {error['msg']}")
            return result
        except Exception as e:
            return ValidationResult(
                is_valid=False, errors=[f"Validation error: {str(e)}"]
            )

    def validate_schema(self, data: Any, schema_name: str) -> ValidationResult:
        """Validate data against a specific schema."""
        validator = self.get_validator(schema_name)
        if not validator:
            return ValidationResult(
                is_valid=False, errors=[f"Unknown schema: {schema_name}"]
            )

        return validator.validate(data)
