# Multilingual grounding resources

The JSON files in this directory are small CC0-1.0 starter crosswalks and
clinical alias tables maintained by OpenMed. They contain no patient records,
credentials, model weights, or license-restricted terminology release data.

Each file declares a schema version, resource version, license, redistribution
flag, and exact source-to-international-code mappings. Larger resources can be
loaded from caller-controlled local storage through `load_crosswalk()`.
