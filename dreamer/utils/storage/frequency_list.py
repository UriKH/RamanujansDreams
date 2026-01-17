from typing import Callable, Any, Optional


class FrequencyList:
    """
    A self-organizing list optimized for linear scanning.
    Items accessed most frequently bubble to the top.
    """

    def __init__(self, max_size: int = 100):
        self.items = []  # List of [value, frequency]
        self.max_size = max_size

    def append(self, value: Any):
        """Adds a new item to the cache (starts with freq=0)."""
        # Optional: Check if already exists to prevent duplicates
        for item in self.items:
            if item[0] == value:
                return

        if len(self.items) >= self.max_size:
            self.items.pop()  # Remove least frequent (last item)

        # Insert at end with freq=0
        self.items.append([value, 0])

    def find(self, matcher: Callable[[Any], bool]) -> Optional[Any]:
        """
        Scans list. If matcher(value) is True, increments freq
        and bubbles the item up. Returns the value.
        """
        for i, item in enumerate(self.items):
            # 1. Check condition (The expensive part)
            if matcher(item[0]):

                # 2. Increment Frequency
                item[1] += 1

                # 3. Bubble Up (The optimization)
                # Swap with left neighbor if this item has higher frequency
                curr_idx = i
                while curr_idx > 0 and self.items[curr_idx][1] > self.items[curr_idx - 1][1]:
                    # Python swap is atomic and fast
                    self.items[curr_idx], self.items[curr_idx - 1] = \
                        self.items[curr_idx - 1], self.items[curr_idx]
                    curr_idx -= 1
                return item[0]
        return None
