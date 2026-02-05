from typing import List, Dict, Any
from dataclasses import fields, is_dataclass, asdict


class Configurable:
    def get_configurations(self) -> List[str]:
        if is_dataclass(self):
            return [f.name for f in fields(self)]
        return []

    def export_configurations(self) -> Dict[str, Any]:
        if is_dataclass(self):
            return asdict(self)
        return dict()

    def _format_value(self, value) -> str:
        """
        Helper to format complex values for display.
        """
        s_val = str(value)
        if len(s_val) > 60:
            return s_val[:57] + "..."
        return s_val

    def _simple_table(self, data: list) -> str:
        if not data:
            return ""

        # Calculate column widths
        col1_w = max(len(str(row[0])) for row in data)
        col1_w = max(col1_w, len("Property"))

        col2_w = max(len(str(row[1])) for row in data)
        col2_w = max(col2_w, len("Value"))

        # Build table
        separator = f"+-{'-' * col1_w}-+-{'-' * col2_w}-+"
        header = f"| {'Property':<{col1_w}} | {'Value':<{col2_w}} |"

        lines = [separator, header, separator]
        for name, val in data:
            lines.append(f"| {name:<{col1_w}} | {val:<{col2_w}} |")
        lines.append(separator)

        return "\n".join(lines)

    def display(self) -> str:
        """
        Returns a formatted table string of the object's fields and values.
        """
        if not is_dataclass(self):
            return str(self)

        data = []
        for f in fields(self):
            name = f.name
            value = getattr(self, name)

            # Custom formatting for better readability
            display_value = self._format_value(value)
            data.append([name, display_value])

        return self._simple_table(data)
