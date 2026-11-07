"""A tiny de-identification demo using synthetic text only."""

from openmed import deidentify


NOTE = "Patient Alex Example called from phone number 555-0100 about a cough."


def main() -> None:
    """Mask personal information in the example note and print the result."""
    result = deidentify(NOTE, method="mask")
    print("Original note:")
    print(NOTE)
    print("\nRedacted note:")
    print(result.deidentified_text)


if __name__ == "__main__":
    main()