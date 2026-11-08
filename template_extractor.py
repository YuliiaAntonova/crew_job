"""Build a blank JSON-shaped template from an existing YAML/JSON CV."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def _merge_shape(left: Any, right: Any) -> Any:
    """Merge observed structures without copying candidate data."""
    if isinstance(left, dict) and isinstance(right, dict):
        merged = dict(left)
        for key, value in right.items():
            merged[key] = _merge_shape(merged[key], value) if key in merged else value
        return merged
    if isinstance(left, list) and isinstance(right, list):
        if left and right:
            return [_merge_shape(left[0], right[0])]
        return left or right
    # Mixed scalar/object shapes are normalized to the structured object.
    if isinstance(left, dict):
        return left
    if isinstance(right, dict):
        return right
    return None


def extract_fields(data: Any) -> Any:
    if isinstance(data, dict):
        return {key: extract_fields(value) for key, value in data.items()}
    if isinstance(data, list):
        item_shapes = [extract_fields(item) for item in data]
        structured = [item for item in item_shapes if isinstance(item, dict)]
        if structured:
            merged = structured[0]
            for item in structured[1:]:
                merged = _merge_shape(merged, item)
            return [merged]
        return []
    return None


def get_target_fields(path: str | Path) -> dict:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            data = yaml.safe_load(handle)
        elif path.suffix.lower() == ".json":
            data = json.load(handle)
        else:
            raise ValueError("Target template must be .yml, .yaml or .json")
    if not isinstance(data, dict):
        raise ValueError("Target template must be a mapping/object at the root")
    return extract_fields(data)


def get_field_names(data: Any, prefix: str = "") -> list[str]:
    """Leaf paths for debugging / inspecting extracted fields."""
    if isinstance(data, dict):
        result = []
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else key
            result.extend(get_field_names(value, path))
        return result
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return get_field_names(data[0], prefix + "[]")
    return [prefix]


def validate_shape(value: Any, shape: Any, path: str = "root") -> None:
    """Check output keys/types, permitting repeats for list sections."""
    if isinstance(shape, dict):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        missing = set(shape) - set(value)
        extra = set(value) - set(shape)
        if missing or extra:
            raise ValueError(f"{path}: missing={sorted(missing)}, extra={sorted(extra)}")
        for key, child in shape.items():
            validate_shape(value[key], child, f"{path}.{key}")
    elif isinstance(shape, list):
        if not isinstance(value, list):
            raise ValueError(f"{path} must be a list")
        if shape:
            for i, item in enumerate(value):
                validate_shape(item, shape[0], f"{path}[{i}]")
        elif any(isinstance(item, (dict, list)) for item in value):
            raise ValueError(f"{path} must be a list of scalar values")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise ValueError(f"{path} must be scalar or null")
