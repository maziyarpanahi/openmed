# 快速开始

本指南帮助你在几分钟内从空白工作站开始运行示例并复制文档中的结果。示例使用 [uv](https://github.com/astral-sh/uv) 管理依赖，但任何 Python 3.11+ 环境都可以使用。

## 1. 初始化环境

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # 安装 uv（已安装则跳过）
uv venv --python 3.11                           # 创建专用虚拟环境
source .venv/bin/activate                       # 也可以直接使用 `uv python`

# 安装 OpenMed、Hugging Face 扩展和文档工具
uv pip install ".[hf]"
```

需要 zero-shot GLiNER 技术栈或开发工具？按需组合扩展：

```bash
uv pip install ".[hf,gliner]"      # 添加 GLiNER 与 transformers
uv pip install ".[dev]"            # pytest、覆盖率和代码检查
```

若要处理扫描图像和文档 OCR，请安装多模态扩展以及系统 Tesseract 二进制文件：

```bash
uv pip install ".[multimodal]"
brew install tesseract             # macOS
sudo apt-get install tesseract-ocr # Debian/Ubuntu
```

PaddleOCR 可作为较重的可选 OCR 后端：

```bash
uv pip install ".[ocr-paddle]"
```

核心安装已包含 CDA/C-CDA XML 去标识化。它会对结构化头部中的 PHI 执行脱敏，扫描 CDA 章节叙述文本，保持 XML 可解析，并且只处理看起来像 CDA 文档的 `.xml` 文件：

```python
from openmed.interop.cda import redact_cda

redacted_xml = redact_cda("synthetic_ccda.xml", date_shift_days=30)
```

在 Apple Silicon Mac 上，可以直接使用新的 MLX 路径：

```bash
uv pip install ".[mlx]"            # Python MLX 运行时以及分词器/制品依赖
uv run python -c "from openmed.core.backends import get_backend; print(type(get_backend()).__name__)"
```

如果希望在一台机器上使用完整功能，可组合安装：

```bash
uv pip install ".[hf,mlx,docs]"
```

## 2. 运行 `analyze_text`

```python
from openmed import analyze_text

text = "Metastatic breast cancer treated with paclitaxel and trastuzumab."

resp = analyze_text(text, model_name="disease_detection_superclinical")
print(resp.entities[0])

# 如需可直接嵌入的 HTML，请选择 "html" 输出格式
html = analyze_text(text, model_name="disease_detection_superclinical", output_format="html")
print(html)  # 可直接用于仪表板或文档
```

更喜欢快速的脚本入口？运行单文件冒烟测试：

```bash
uv run python examples/pii_model_comparison.py
```

## 3. 对 PII 去标识化

```python
from openmed import deidentify

result = deidentify("Patient John Doe, DOB 01/15/1970", method="mask")
print(result.deidentified_text)
# Patient [first_name] [last_name], DOB [date]
```

`deidentify()` 支持五种方法（`mask`、`remove`、`replace`、`hash`、`shift_dates`）。请参阅[匿名化快速入门](anonymization.md#quickstart-choosing-a-method)，其中包含每种方法的可运行示例以及使用 `reidentify()` 还原结果的方式。

## 4. 从文档复制代码片段

所有代码块均使用 Material for MkDocs 的复制按钮。打开命令面板（`/` 或 `cmd/ctrl + K`）后，可以搜索 “GLiNER”、“OpenMedConfig” 或 “token classification”，再复制预览窗格中的片段。如果你使用 AI 编程助手，请让它读取已发布的文档网址，以便它引用同一份结构化 Markdown 并给出规范答案。

## 5. 可选：固定配置

```python
from openmed.core import OpenMedConfig, ModelLoader

config = OpenMedConfig.from_env_fallback(
    cache_dir="~/.cache/openmed",
    device="cuda",
    default_org="OpenMed",
)
loader = ModelLoader(config=config)
ner = loader.create_pipeline("disease_detection_superclinical")
entities = ner("Hydroxyurea dose reduced after platelet drop.")
```

请继续阅读**配置**章节，了解完整的 YAML/ENV 架构、PHI 感知验证工具和日志设置。
