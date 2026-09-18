# Custom status vocabulary examples

Clinical status helpers accept a vocabulary mapping when an application needs
local labels. Keep the canonical status value stable and map display labels at
the boundary, so reports and serialized records remain interoperable.

For a custom vocabulary, provide a JSON object with the application's labels,
load it with the status-vocabulary helper, and test both a known label and the
unknown-label failure path. Do not silently map an unknown value to a positive
clinical state.
