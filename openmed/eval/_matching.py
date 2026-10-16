"""One-to-one eval matching without materializing the eligible-edge graph."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable


def maximum_matching_count(
    left_count: int, right_count: int, eligible: Callable[[int, int], bool]
) -> int:
    """Count a maximum matching with linear auxiliary memory.

    ``eligible`` must be a stable predicate for the duration of the call.
    Greedily seed a matching, then find augmenting paths from all unmatched
    left vertices together. A failed search proves the matching is maximum;
    it does not repeat the same blocked component for every unmatched root.
    Edges are generated on demand, never stored. Iterative traversal avoids
    Python recursion limits. Counts do not depend on the initial greedy pairs.

    For L and R vertices, auxiliary space is O(L + R). The worst-case time
    is O(L * R * min(L, R)); a greedy maximum or one final blocked search
    takes O(L * R). There is no weighted-overlap tie-breaking objective.
    """
    if not left_count or not right_count:
        return 0
    left_matches = [-1] * left_count
    right_matches = [-1] * right_count
    free_right = dict.fromkeys(range(right_count))
    count = 0
    maximum = min(left_count, right_count)
    for left in range(left_count):
        for right in free_right:
            if eligible(left, right):
                left_matches[left] = right
                right_matches[right] = left
                del free_right[right]
                count += 1
                break
        if count == maximum:
            return count

    while count < maximum:
        queue = deque(left for left, right in enumerate(left_matches) if right < 0)
        seen_left = set(queue)
        parents: dict[int, int] = {}
        free_end = -1
        while queue and free_end < 0:
            left = queue.popleft()
            for right in range(right_count):
                if right in parents or not eligible(left, right):
                    continue
                parents[right] = left
                previous_left = right_matches[right]
                if previous_left < 0:
                    free_end = right
                    break
                if previous_left not in seen_left:
                    seen_left.add(previous_left)
                    queue.append(previous_left)
        if free_end < 0:
            return count
        del free_right[free_end]
        cursor = free_end
        while cursor >= 0:
            left = parents[cursor]
            previous_right = left_matches[left]
            left_matches[left] = cursor
            right_matches[cursor] = left
            cursor = previous_right
        count += 1
    return count
