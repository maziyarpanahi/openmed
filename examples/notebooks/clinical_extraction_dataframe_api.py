"""Notebook-style clinical extraction dataframe example.

Run with:
    python examples/notebooks/clinical_extraction_dataframe_api.py
"""

# %%
import pandas as pd

from openmed.interop import get_adapter
from openmed.interop.notebook.widget import ClinicalExtractionWidget

get_adapter("pandas")

notes = pd.DataFrame(
    {
        "note": [
            "Patient [PERSON] takes metformin for type 2 diabetes.",
            "Patient [PERSON] denies cough but reports asthma.",
        ]
    }
)

# %%
entities = notes.openmed.extract("note")
print(entities)

# %%
analytics = (
    entities.groupby(["entity_label", "code"], dropna=False)
    .size()
    .reset_index(name="span_count")
)
print(analytics)

# %%
first_note = notes.loc[0, "note"]
first_note_rows = pd.DataFrame({"note": [first_note]}).openmed.extract("note")
widget = ClinicalExtractionWidget(first_note, tuple(first_note_rows.to_dict("records")))
html = widget.to_html()

if __name__ == "__main__":
    print(html[:160])
