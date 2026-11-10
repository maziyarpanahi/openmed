"""Before-state public API fixture."""

__all__ = [
    "LegacyClient",
    "narrowed",
    "removed",
    "renamed",
    "retained",
]


class LegacyClient:
    """Fixture class with public attributes and methods."""

    stable_attribute = "stable"
    removed_attribute = "removed"

    def request(self, value: str, optional: int = 1) -> str:
        """Return a fixture response."""

        return f"{value}:{optional}"


def removed(value: str) -> str:
    """Represent a removed public function."""

    return value.upper()


def renamed(value: str) -> str:
    """Represent a renamed public function."""

    return value.casefold()


def narrowed(value: str, optional: int = 1, **options: object) -> str:
    """Represent a callable whose accepted calls later narrow."""

    return f"{value}:{optional}:{len(options)}"


def retained(value: str) -> str:
    """Represent a callable that later becomes deprecated."""

    return value


def hidden_even_though_publicly_named() -> None:
    """Prove that a module ``__all__`` is authoritative."""
