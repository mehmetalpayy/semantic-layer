"""Helpers for preprocessing model and column metadata."""

import importlib
import logging
import pkgutil
import re
import sys
from collections.abc import Callable
from typing import Any

import orjson

logger = logging.getLogger("wren-ai-service")


class Helper:
    """Callable helper with a predicate for whether it applies."""

    def __init__(
        self,
        condition: Callable[[dict[str, Any]], bool],
        helper: Callable[[dict[str, Any]], Any],
    ):
        """Store predicate and handler for a helper."""
        self.condiction = condition
        self.helper = helper

    def condition(self, column: dict[str, Any], **kwargs) -> bool:
        """Return whether the helper should run for the column."""
        return self.condiction(column, **kwargs)

    def __call__(self, column: dict[str, Any], **kwargs) -> Any:
        """Run the helper for the provided column."""
        return self.helper(column, **kwargs)


def clean_display_name(display_name: str) -> str:
    """Normalize a display name into a safe identifier."""
    if not display_name:
        return display_name

    # Numbers are only invalid at prefix, not in middle or suffix.
    prefix_invalid = set(
        [
            "-",
            "&",
            "%",
            "=",
            "+",
            "'",
            '"',
            "<",
            ">",
            "#",
            "|",
            "!",
            "(",
            ")",
            "*",
            ",",
            "/",
            ";",
            "[",
            "\\",
            "]",
            "^",
            "{",
            "}",
            "~",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "\x00",
            ".",
        ]
    )
    middle_invalid = set(
        [
            "-",
            "&",
            "%",
            "=",
            "+",
            "'",
            '"',
            "<",
            ">",
            "#",
            "|",
            "!",
            "(",
            ")",
            "/",
            "?",
            "[",
            "\\",
            "]",
            "^",
            "`",
            "{",
            "}",
            "~",
            ".",
            "*",
            "@",
            "$",
        ]
    )
    suffix_invalid = set(
        [
            "-",
            "&",
            "%",
            "=",
            "+",
            ":",
            "'",
            '"',
            "<",
            ">",
            "#",
            "|",
            "!",
            "(",
            ")",
            ",",
            ".",
            "/",
            "@",
            "[",
            "\\",
            "]",
            "^",
            "{",
            "}",
            "~",
        ]
    )

    result = list(display_name)
    prefix_prepended = False

    if len(result) > 0 and result[0] in prefix_invalid:
        if result[0].isdigit():
            result.insert(0, "_")
            prefix_prepended = True
        else:
            result[0] = "_"

    start_idx = 2 if prefix_prepended else 1
    end_idx = len(result) - 1
    for i in range(start_idx, end_idx):
        if result[i] in middle_invalid:
            result[i] = "_"

    if len(result) > 1 and result[-1] in suffix_invalid:
        result[-1] = "_"

    if len(display_name) == 1:
        char = display_name[0]
        if char in prefix_invalid or char in suffix_invalid:
            result = ["_"]

    cleaned = "".join(result)
    cleaned = re.sub(r"_+", "_", cleaned)

    return cleaned


def _properties_comment(column: dict[str, Any], **_) -> str:
    """Build a column comment from properties metadata."""
    props = column["properties"]
    column_properties: dict[str, Any] = {}
    alias = clean_display_name(props.get("displayName", ""))
    description = props.get("description", "")
    if alias:
        column_properties["alias"] = alias
    if description:
        column_properties["description"] = description
    if "uniqueValues" in props and props.get("uniqueValues"):
        column_properties["uniqueValues"] = props.get("uniqueValues", [])

    nested = {k: v for k, v in props.items() if k.startswith("nested")}
    if nested:
        column_properties["nested_columns"] = nested

    if (json_type := props.get("json_type", "")) and json_type in [
        "JSON",
        "JSON_ARRAY",
    ]:
        json_fields = {
            k: v for k, v in column["properties"].items() if re.match(r".*json.*", k)
        }
        if json_fields:
            column_properties["json_type"] = json_type
            column_properties["json_fields"] = json_fields

    if not column_properties:
        return ""

    return f"-- {orjson.dumps(column_properties).decode('utf-8')}\n  "


COLUMN_PREPROCESSORS = {
    "properties": Helper(
        condition=lambda column, **_: "properties" in column,
        helper=lambda column, **_: column.get("properties"),
    ),
    "relationship": Helper(
        condition=lambda column, **_: "relationship" in column,
        helper=lambda column, **_: column.get("relationship"),
    ),
    "expression": Helper(
        condition=lambda column, **_: "expression" in column,
        helper=lambda column, **_: column.get("expression"),
    ),
    "isCalculated": Helper(
        condition=lambda column, **_: column.get("isCalculated", False),
        helper=lambda column, **_: column.get("isCalculated"),
    ),
}

COLUMN_COMMENT_HELPERS = {
    "properties": Helper(
        condition=lambda column, **_: "properties" in column,
        helper=_properties_comment,
    ),
    "isCalculated": Helper(
        condition=lambda column, **_: column.get("isCalculated", False),
        helper=lambda column, **_: (
            f"-- This column is a Calculated Field\n  -- column expression: {column['expression']}\n  "
        ),
    ),
}

MODEL_PREPROCESSORS = {}


def load_helpers(package_path: str = "semantic_layer"):
    """Load preprocessors and comment helpers from the given package."""
    package = importlib.import_module(package_path)

    for _, name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        if name in sys.modules:
            continue

        module = importlib.import_module(name)
        logger.info(f"Imported Helper from {name}")
        if hasattr(module, "MODEL_PREPROCESSORS"):
            MODEL_PREPROCESSORS.update(module.MODEL_PREPROCESSORS)
            logger.info(f"Updated Helper for model preprocessors: {name}")
        if hasattr(module, "COLUMN_PREPROCESSORS"):
            COLUMN_PREPROCESSORS.update(module.COLUMN_PREPROCESSORS)
            logger.info(f"Updated Helper for column preprocessors: {name}")
        if hasattr(module, "COLUMN_COMMENT_HELPERS"):
            COLUMN_COMMENT_HELPERS.update(module.COLUMN_COMMENT_HELPERS)
            logger.info(f"Updated Helper for column comment helpers: {name}")
