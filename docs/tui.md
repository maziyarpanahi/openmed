# TUI - Interactive Terminal Interface

OpenMed includes a rich terminal user interface (TUI) for interactive clinical NER analysis. Built with [Textual](https://textual.textualize.io/), it provides a full-featured workbench that runs in any terminal.

## Installation

The TUI requires the `tui` extra:

```bash
pip install openmed[tui]
# or with uv
uv pip install openmed[tui]
```

## Quick Start

Launch the TUI from the command line:

```bash
openmed tui
```

Or with a specific model:

```bash
openmed tui --model disease_detection_superclinical --confidence-threshold 0.6
```

## Interface Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  OpenMed TUI                                         Interactive Clinical NER │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─ Input (Ctrl+Enter to analyze) ─────────────────────────────────────────┐ │
│  │ Patient diagnosed with chronic myeloid leukemia, started on imatinib.   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─ Annotated ─────────────────────────────────────────────────────────────┐ │
│  │ Patient diagnosed with [chronic myeloid leukemia], started on           │ │
│  │ [imatinib].                                                             │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  ┌─ Entities (2) ──────────────────────────────────────────────────────────┐ │
│  │  Label     Entity                      Confidence                       │ │
│  │  DISEASE   chronic myeloid leukemia    ████████████████████ 0.98        │ │
│  │  DRUG      imatinib                    ███████████████████░ 0.95        │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ Model: disease_detection_superclinical │ Threshold: 0.50 │ Inference: 23ms   │
├──────────────────────────────────────────────────────────────────────────────┤
│ Ctrl+Enter Analyze  Ctrl+L Clear  F1 Help  Ctrl+Q Quit                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Panels

### Input Panel
- Multi-line text area for entering clinical notes
- Paste text directly or type manually
- Press `Ctrl+Enter` to analyze

### Annotated View
- Shows your text with entities highlighted inline
- Color-coded by entity type:
  - **Red**: Diseases, conditions, diagnoses
  - **Blue**: Drugs, medications, treatments
  - **Green**: Anatomy, body parts, organs
  - **Purple**: Procedures, tests, labs
  - **Amber**: Genes, proteins
  - **Cyan**: Species, organisms

### Entity Table
- Lists all detected entities sorted by confidence
- Visual confidence bars
- Zebra striping for readability

### Status Bar
- Current model name
- Confidence threshold
- Last inference time in milliseconds

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+Enter` | Analyze current text |
| `Ctrl+L` | Clear input and results |
| `Ctrl+Q` | Quit application |
| `F1` | Show help |

## CLI Options

```bash
openmed tui [OPTIONS]

Options:
  --model TEXT                  Model registry key or HuggingFace ID
  --confidence-threshold FLOAT  Minimum confidence (default: 0.5)
  --help                        Show help message
```

## Python API

You can also launch the TUI programmatically:

```python
from openmed.tui import OpenMedTUI

# Basic usage
app = OpenMedTUI()
app.run()

# With custom settings
app = OpenMedTUI(
    model_name="disease_detection_superclinical",
    confidence_threshold=0.6,
)
app.run()

# With custom analysis function
def my_analyzer(text, **kwargs):
    # Custom analysis logic
    return {"entities": [...]}

app = OpenMedTUI(analyze_func=my_analyzer)
app.run()
```

Or use the convenience function:

```python
from openmed.tui.app import run_tui

run_tui(
    model_name="pharma_detection_superclinical",
    confidence_threshold=0.7,
)
```

## Entity Color Reference

| Entity Type | Color | Hex |
|-------------|-------|-----|
| DISEASE, CONDITION, PROBLEM, DIAGNOSIS | Red | `#ef4444` |
| DRUG, MEDICATION, TREATMENT, CHEMICAL | Blue | `#3b82f6` |
| ANATOMY, BODY_PART, ORGAN | Green | `#22c55e` |
| PROCEDURE, TEST, LAB | Purple | `#a855f7` |
| GENE, PROTEIN, GENE_OR_GENE_PRODUCT | Amber | `#f59e0b` |
| SPECIES, ORGANISM | Cyan | `#06b6d4` |
| Other | Gray | `#9ca3af` |

## Tips

1. **Large Documents**: The TUI handles multi-line clinical notes well. Paste entire discharge summaries or progress notes.

2. **Quick Iteration**: Adjust your text and re-analyze instantly with `Ctrl+Enter`.

3. **Remote Access**: The TUI works over SSH, making it perfect for analyzing data on remote servers.

4. **Demo Mode**: Great for presenting OpenMed capabilities in meetings or at conferences.

## Troubleshooting

### TUI doesn't start
Ensure you have the TUI dependencies installed:
```bash
pip install openmed[tui]
```

### Colors not showing
Make sure your terminal supports true color. Most modern terminals do (iTerm2, Windows Terminal, GNOME Terminal, etc.).

### Slow first analysis
The first analysis may take longer as the model is loaded. Subsequent analyses will be faster.

## See Also

- [CLI & Automation](./cli.md) - Command-line interface for batch operations
- [Analyze Text Helper](./analyze-text.md) - Python API for text analysis
- [Configuration Profiles](./profiles.md) - Manage different configurations
