# OpenMed Jupyter Notebooks

This directory contains comprehensive Jupyter notebooks demonstrating OpenMed's capabilities.

## 📚 Available Notebooks

### 🆕 [PII_Detection_Complete_Guide.ipynb](./PII_Detection_Complete_Guide.ipynb)

**Complete guide to PII detection and de-identification (v0.5.0+)**

A comprehensive tutorial covering **everything** about PII functionality:

- **Basic PII Extraction** - Detect PII entities in clinical text
- **Smart Entity Merging** - Fix fragmentation issues (NEW in v0.5.0)
- **De-identification Methods** - Mask, remove, replace, hash, shift dates
- **Re-identification** - Reverse de-identification with mappings
- **Batch Processing** - Process multiple texts efficiently
- **Confidence Thresholding** - Control precision vs recall
- **Custom Patterns** - Add domain-specific PII patterns
- **Clinical Use Cases** - Real-world examples (discharge summaries, research datasets, HIPAA compliance)
- **Visualization** - Display results with highlighting
- **CLI Usage** - Command-line interface examples

**Topics covered:**

- 48 executable cells with detailed explanations
- Smart merging comparison (before/after)
- All 5 de-identification methods demonstrated
- HIPAA compliance checklist
- Production-ready examples
- Best practices and security considerations

**Recommended for:**

- Healthcare data scientists
- Clinical researchers
- HIPAA compliance officers
- Anyone working with protected health information (PHI)

---

### [getting_started.ipynb](./getting_started.ipynb)

**Introduction to OpenMed basics**

Learn the fundamentals of OpenMed:

- Installation and setup
- Basic text analysis with `analyze_text()`
- Model discovery and selection
- Output formatting (JSON, HTML, CSV)
- Entity grouping and filtering

**Recommended for:** First-time users, quick start

---

### [Sentence_Detection_Batching.ipynb](./Sentence_Detection_Batching.ipynb)

**Advanced sentence detection and batch processing**

Master efficient text processing:

- Sentence boundary detection with pySBD
- Batch processing multiple documents
- Performance optimization
- Memory-efficient processing

**Recommended for:** Production deployments, large-scale processing

---

### [Medical_Tokenizer_Demo.ipynb](./Medical_Tokenizer_Demo.ipynb)

**Medical-aware tokenization**

Explore medical text tokenization:

- Medical term boundary preservation
- Tokenization comparison
- Custom tokenizer configuration

**Recommended for:** NLP engineers, model developers

---

### [Medical_Tokenizer_Benchmark.ipynb](./Medical_Tokenizer_Benchmark.ipynb)

**Performance benchmarking**

Compare tokenization performance:

- Speed benchmarks
- Accuracy metrics
- Resource usage

**Recommended for:** Performance optimization

---

### [OpenMed_CLI_Demo.ipynb](./OpenMed_CLI_Demo.ipynb)

**Command-line interface demonstration**

Learn CLI usage:

- Command-line text analysis
- Batch processing via CLI
- Configuration management
- Scripting examples

**Recommended for:** System administrators, automation

---

### [ZeroShot_NER_Tour.ipynb](./ZeroShot_NER_Tour.ipynb)

**Zero-shot NER capabilities**

Explore zero-shot learning:

- No training required
- Custom entity types
- Domain adaptation

**Recommended for:** Research, custom entity detection

---

## 🚀 Getting Started

### Prerequisites

```bash
# Install OpenMed
pip install openmed

# Install Jupyter
pip install jupyter

# For PII notebooks, you may need:
pip install matplotlib ipython
```

### Running the Notebooks

1. **Clone the repository:**

   ```bash
   git clone https://github.com/maziyarpanahi/openmed.git
   cd openmed/examples/notebooks
   ```

2. **Start Jupyter:**

   ```bash
   jupyter notebook
   ```

3. **Open a notebook** and run cells sequentially

### Environment Setup

Some notebooks may require a HuggingFace token for accessing gated models:

```python
import os
os.environ['HF_TOKEN'] = 'your_token_here'
```

Get your token from: <https://huggingface.co/settings/tokens>

---

## 📋 Notebook Quick Reference

| Notebook | Difficulty | Topics | Duration |
|----------|-----------|--------|----------|
| **PII_Detection_Complete_Guide** | Intermediate | PII, De-identification, HIPAA | 30-45 min |
| **getting_started** | Beginner | Basics, Installation | 10-15 min |
| **Sentence_Detection_Batching** | Intermediate | Batch processing | 15-20 min |
| **Medical_Tokenizer_Demo** | Intermediate | Tokenization | 10-15 min |
| **Medical_Tokenizer_Benchmark** | Advanced | Performance | 15-20 min |
| **OpenMed_CLI_Demo** | Beginner | CLI | 10-15 min |
| **ZeroShot_NER_Tour** | Advanced | Zero-shot NER | 20-30 min |

---

## 🎯 Learning Paths

### Path 1: Getting Started with OpenMed

1. `getting_started.ipynb` - Learn basics
2. `OpenMed_CLI_Demo.ipynb` - CLI usage
3. `Sentence_Detection_Batching.ipynb` - Batch processing

### Path 2: PII Detection & De-identification

1. `getting_started.ipynb` - Basics first
2. `PII_Detection_Complete_Guide.ipynb` - Complete PII guide
3. Practice with your own clinical notes

### Path 3: Advanced NER

1. `getting_started.ipynb` - Foundation
2. `ZeroShot_NER_Tour.ipynb` - Zero-shot capabilities
3. `Medical_Tokenizer_Demo.ipynb` - Tokenization deep dive

### Path 4: Production Deployment

1. `Sentence_Detection_Batching.ipynb` - Batch processing
2. `Medical_Tokenizer_Benchmark.ipynb` - Performance tuning
3. `PII_Detection_Complete_Guide.ipynb` - Section 10 (CLI)

---

## 💡 Tips for Using Notebooks

### Best Practices

1. **Run cells sequentially** - Dependencies may exist between cells
2. **Check imports** - Ensure all required packages are installed
3. **Monitor memory** - Some models are large (500MB-1GB+)
4. **Use GPU if available** - Speeds up inference significantly
5. **Save outputs** - Notebooks auto-save, but export important results

### Common Issues

**Issue: ImportError**

```python
# Solution: Install missing package
!pip install package-name
```

**Issue: Model download slow**

```python
# Solution: Use HuggingFace cache
export HF_HOME="/path/to/cache"
```

**Issue: Out of memory**

```python
# Solution: Use smaller batch size or CPU
batch_size=1  # or device="cpu"
```

### Customization

All notebooks are designed to be modified:

- Change model names to try different models
- Adjust confidence thresholds
- Use your own clinical text
- Modify visualization styles

---

## 🔗 Additional Resources

- **Documentation:** <https://github.com/maziyarpanahi/openmed>
- **Model Hub:** <https://huggingface.co/openmed>
- **Issue Tracker:** <https://github.com/maziyarpanahi/openmed/issues>
- **Discussions:** <https://github.com/maziyarpanahi/openmed/discussions>

---

## 📝 Contributing

Have a notebook idea or improvement?

1. Fork the repository
2. Create a new notebook in this directory
3. Follow the existing style and structure
4. Submit a pull request

### Notebook Template Structure

```markdown
# Title

## Overview
- What this notebook covers
- Prerequisites
- Expected outcomes

## Setup
- Imports
- Configuration

## Main Content
- Section 1: Topic A
- Section 2: Topic B
- ...

## Summary
- Key takeaways
- Next steps
```

---

## 📄 License

All notebooks are released under the same license as OpenMed (Apache 2.0).

---

**Last Updated:** 2026-01-13

**OpenMed Version:** v0.5.0+
