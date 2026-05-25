"""
Data serialization utilities for TPE system.

Provides robust data serialization and deserialization with error handling.
"""

import json
import pickle
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from core.models.base import BaseTPEModel
from utils.validation import ValidationEngine, ValidationError


class SerializationError(Exception):
    """Custom serialization exception."""

    pass


class DeserializationError(Exception):
    """Custom deserialization exception."""

    pass


class BaseSerializer:
    """Base class for all serializers."""

    def __init__(self, name: str):
        self.name = name
        self.validation_engine = ValidationEngine()

    def serialize(self, data: Any) -> Any:
        """Serialize data to target format."""
        raise NotImplementedError("Subclasses must implement serialize method")

    def deserialize(self, data: Any, target_type: Optional[Type] = None) -> Any:
        """Deserialize data from source format."""
        raise NotImplementedError("Subclasses must implement deserialize method")


class JSONSerializer(BaseSerializer):
    """JSON serializer with datetime handling and validation."""

    def __init__(self):
        super().__init__("JSON")

    def serialize(self, data: Any, validate: bool = True) -> str:
        """Serialize data to JSON string."""
        try:
            # Validate data if requested
            if validate and isinstance(data, BaseModel):
                validation_result = self.validation_engine.validate_pydantic_model(data)
                if not validation_result.is_valid:
                    raise SerializationError(
                        f"Validation failed: {validation_result.errors}"
                    )

            # Convert Pydantic model to dict if needed
            if isinstance(data, BaseModel):
                data = data.model_dump()

            # Custom JSON encoder for datetime objects
            json_str = json.dumps(
                data, cls=CustomJSONEncoder, ensure_ascii=False, indent=2
            )
            return json_str

        except Exception as e:
            raise SerializationError(f"Failed to serialize to JSON: {e}")

    def deserialize(
        self, data: Union[str, bytes], target_type: Optional[Type] = None
    ) -> Any:
        """Deserialize data from JSON string."""
        try:
            # Parse JSON
            parsed_data = json.loads(data)

            # Convert to target type if specified
            if target_type:
                if issubclass(target_type, BaseModel):
                    return target_type(**parsed_data)
                else:
                    # For non-Pydantic types, just return the parsed data
                    return parsed_data

            return parsed_data

        except json.JSONDecodeError as e:
            raise DeserializationError(f"Invalid JSON: {e}")
        except Exception as e:
            raise DeserializationError(f"Failed to deserialize from JSON: {e}")


class PickleSerializer(BaseSerializer):
    """Pickle serializer for Python objects."""

    def __init__(self):
        super().__init__("Pickle")

    def serialize(self, data: Any, validate: bool = True) -> bytes:
        """Serialize data to pickle bytes."""
        try:
            # Validate data if requested
            if validate and isinstance(data, BaseModel):
                validation_result = self.validation_engine.validate_pydantic_model(data)
                if not validation_result.is_valid:
                    raise SerializationError(
                        f"Validation failed: {validation_result.errors}"
                    )

            # Serialize with pickle
            pickle_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            return pickle_bytes

        except Exception as e:
            raise SerializationError(f"Failed to serialize to pickle: {e}")

    def deserialize(self, data: bytes, target_type: Optional[Type] = None) -> Any:
        """Deserialize data from pickle bytes."""
        try:
            # Deserialize with pickle
            obj = pickle.loads(data)

            # Validate target type if specified
            if target_type and not isinstance(obj, target_type):
                raise DeserializationError(
                    f"Deserialized object is not of type {target_type}"
                )

            return obj

        except Exception as e:
            raise DeserializationError(f"Failed to deserialize from pickle: {e}")


class CSVSerializer(BaseSerializer):
    """CSV serializer for tabular data."""

    def __init__(self):
        super().__init__("CSV")

    def serialize(
        self, data: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None
    ) -> str:
        """Serialize list of dictionaries to CSV string."""
        try:
            import csv
            import io

            if not data:
                return ""

            # Determine field names
            if fieldnames is None:
                fieldnames = list(data[0].keys())

            # Create CSV string
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

            return output.getvalue()

        except Exception as e:
            raise SerializationError(f"Failed to serialize to CSV: {e}")

    def deserialize(
        self, data: str, target_type: Optional[Type] = None
    ) -> List[Dict[str, Any]]:
        """Deserialize CSV string to list of dictionaries."""
        try:
            import csv
            import io

            if not data.strip():
                return []

            # Parse CSV
            input_io = io.StringIO(data)
            reader = csv.DictReader(input_io)
            result = list(reader)

            # Convert to target type if specified
            if target_type and issubclass(target_type, BaseModel):
                return [target_type(**row) for row in result]

            return result

        except Exception as e:
            raise DeserializationError(f"Failed to deserialize from CSV: {e}")


class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for datetime and other special types."""

    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, date):
            return obj.isoformat()
        elif isinstance(obj, set):
            return list(obj)
        elif hasattr(obj, "model_dump"):  # Pydantic model
            return obj.model_dump()
        elif hasattr(obj, "__dict__"):  # Custom object
            return obj.__dict__
        return super().default(obj)


class SerializationEngine:
    """Main serialization engine."""

    def __init__(self):
        self._serializers = {
            "json": JSONSerializer(),
            "pickle": PickleSerializer(),
            "csv": CSVSerializer(),
        }
        self._default_format = "json"

    def get_serializer(self, format_name: str) -> Optional[BaseSerializer]:
        """Get a serializer by format name."""
        if format_name is None:
            return None
        return self._serializers.get(str(format_name).lower())

    def register_serializer(self, format_name: str, serializer: BaseSerializer) -> None:
        """Register a new serializer."""
        self._serializers[format_name.lower()] = serializer

    def set_default_format(self, format_name: str) -> None:
        """Set the default serialization format."""
        if str(format_name).lower() not in self._serializers:
            raise ValueError(f"Unknown serialization format: {format_name}")
        self._default_format = str(format_name).lower()

    def serialize(
        self,
        data: Any,
        format_name: Optional[str] = None,
        validate: bool = True,
        **kwargs,
    ) -> Union[str, bytes]:
        """Serialize data to specified format."""
        format_name = format_name or self._default_format
        serializer = self.get_serializer(format_name)

        if not serializer:
            raise ValueError(f"Unknown serialization format: {format_name}")

        return serializer.serialize(data, validate=validate, **kwargs)

    def deserialize(
        self,
        data: Union[str, bytes],
        format_name: Optional[str] = None,
        target_type: Optional[Type] = None,
        **kwargs,
    ) -> Any:
        """Deserialize data from specified format."""
        format_name = format_name or self._default_format
        serializer = self.get_serializer(format_name)

        if not serializer:
            raise ValueError(f"Unknown deserialization format: {format_name}")

        return serializer.deserialize(data, target_type=target_type, **kwargs)

    def auto_serialize(
        self, data: Any, file_path: Union[str, Path], validate: bool = True
    ) -> None:
        """Automatically serialize data based on file extension."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        format_map = {
            ".json": "json",
            ".pkl": "pickle",
            ".pickle": "pickle",
            ".csv": "csv",
        }

        format_name = format_map.get(extension)
        if not format_name:
            raise ValueError(
                f"Cannot determine serialization format from extension: {extension}"
            )

        serialized_data = self.serialize(data, format_name, validate=validate)

        # Write to file
        with open(file_path, "w" if isinstance(serialized_data, str) else "wb") as f:
            f.write(serialized_data)

    def auto_deserialize(
        self, file_path: Union[str, Path], target_type: Optional[Type] = None
    ) -> Any:
        """Automatically deserialize data based on file extension."""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        format_map = {
            ".json": "json",
            ".pkl": "pickle",
            ".pickle": "pickle",
            ".csv": "csv",
        }

        format_name = format_map.get(extension)
        if not format_name:
            raise ValueError(
                f"Cannot determine deserialization format from extension: {extension}"
            )

        # Read from file
        mode = "r" if format_name in ["json", "csv"] else "rb"
        with open(file_path, mode) as f:
            data = f.read()

        return self.deserialize(data, format_name, target_type=target_type)


class DataTransformer:
    """Utility class for data transformation operations."""

    def __init__(self):
        self.serialization_engine = SerializationEngine()
        self.validation_engine = ValidationEngine()

    def to_dict(self, obj: Any, validate: bool = True) -> Dict[str, Any]:
        """Convert object to dictionary."""
        try:
            if isinstance(obj, BaseModel):
                if validate:
                    validation_result = self.validation_engine.validate_pydantic_model(
                        obj
                    )
                    if not validation_result.is_valid:
                        raise ValidationError(
                            f"Validation failed: {validation_result.errors}"
                        )
                return obj.model_dump()
            elif isinstance(obj, dict):
                return obj
            else:
                # Try to serialize to JSON and back to dict
                json_str = self.serialization_engine.serialize(
                    obj, "json", validate=validate
                )
                return json.loads(json_str)
        except Exception as e:
            raise SerializationError(f"Failed to convert object to dict: {e}")

    def from_dict(self, data: Dict[str, Any], target_type: Type) -> Any:
        """Convert dictionary to target type."""
        try:
            if issubclass(target_type, BaseModel):
                return target_type(**data)
            else:
                return data
        except Exception as e:
            raise DeserializationError(f"Failed to convert dict to {target_type}: {e}")

    def copy_object(self, obj: Any, deep: bool = True) -> Any:
        """Create a copy of an object."""
        try:
            if isinstance(obj, BaseModel):
                return obj.model_copy(deep=deep)
            elif deep:
                # Use pickle for deep copy
                serialized = self.serialization_engine.serialize(obj, "pickle")
                return self.serialization_engine.deserialize(serialized, "pickle")
            else:
                # Shallow copy
                return obj.__class__(obj)
        except Exception as e:
            raise SerializationError(f"Failed to copy object: {e}")

    def merge_dicts(
        self, dict1: Dict[str, Any], dict2: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge two dictionaries recursively."""
        result = dict1.copy()

        for key, value in dict2.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self.merge_dicts(result[key], value)
            else:
                result[key] = value

        return result

    def flatten_dict(
        self, data: Dict[str, Any], separator: str = ".", prefix: str = ""
    ) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        result = {}

        for key, value in data.items():
            new_key = f"{prefix}{separator}{key}" if prefix else key

            if isinstance(value, dict):
                result.update(self.flatten_dict(value, separator, new_key))
            else:
                result[new_key] = value

        return result

    def unflatten_dict(
        self, data: Dict[str, Any], separator: str = "."
    ) -> Dict[str, Any]:
        """Unflatten a flattened dictionary."""
        result = {}

        for key, value in data.items():
            keys = key.split(separator)
            current = result

            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]

            current[keys[-1]] = value

        return result
