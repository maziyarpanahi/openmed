# OpenMed 文档

OpenMed 集成了精选的生物医学模型、高级去标识化、多模态输入、结构化健康数据工具以及单次调用编排，帮助你无需处理繁杂的基础设施即可交付临床自然语言处理工作流。本文档让可复制的代码片段和工作流触手可及：所有章节均以 Markdown 为基础，支持搜索，并针对快速浏览以及复制到笔记本进行了优化。

OpenMed `1.9.1` 完成了 `1.9` 版本的跨平台交付：Python、浏览器、Node.js 和 Android 共用同一份 ONNX 词元分类模型契约，同时修正 Swift 软件包，扩展临床抽取能力和 17 种语言的 PII 覆盖，并强化发布证据：

- **策略感知的去标识化**：提供签名审计报告、可复现性哈希、审查包、脱敏预览和发布门禁。
- **多模态与结构化输入**：覆盖 OCR、图像、PDF、DOCX、EPUB、vCard/iCalendar、DICOM、CSV/TSV、JSONL 聊天记录、HL7 v2、CDA/C-CDA、FHIR 操作以及 FHIR Bulk NDJSON。
- **Python、Swift、Kotlin/Android、REST、gRPC、React Native、TypeScript 和浏览器路径**：包括 OpenMedKit、类型化 REST 客户端、ONNX/WebGPU 和 Transformers.js 导出包。
- **17 个受支持的 PII 语言代码：ar、de、en、es、fr、he、hi、id、it、ja、ko、nl、pt、ro、te、th 和 tr**：模型支持列表包含区域感知的验证与替代值生成；仅提供证件号验证的区域也有额外覆盖。
- **发布证据**：包括泄漏热力图、模型评分卡、阈值扫描、k-匿名性/l-多样性/t-接近性、效用损失、SBOM、签名镜像、SLSA 来源证明、密钥扫描和可复现依赖锁。

## 你将获得什么

- **精选模型注册表** — 可发现的 Hugging Face 模型，并包含领域、大小和设备建议等元数据。
- **单行编排** — `analyze_text` 封装验证、推理和格式化，可用于脚本、笔记本或服务。
- **PII 检测与去标识化** — 兼顾 HIPAA 要求的智能实体合并、策略配置文件、签名审计报告和生产级去标识化。
- **Apple Silicon 与移动端加速** — 基于 MLX 的 Python 推理，以及通过 OpenMedKit 实现的 Swift 原生和 Android/Kotlin 应用集成。
- **REST 服务** — FastAPI 端点包括 `/livez`、`/readyz`、`/analyze`、`/pii/extract`、`/pii/deidentify`，并支持预热池、批处理、指标和类型化 Python/TypeScript 客户端。
- **浏览器与 React Native 导出** — 面向浏览器运行时中 Transformers.js 词元分类的 ONNX/WebGPU 包，以及移动应用使用的 React Native 桥接。
- **高级 NER 后处理** — 分数感知分组、适合 PHI 的过滤，以及 CSV/JSON/HTML 导出工具。
- **可组合配置** — `OpenMedConfig` 读取 YAML/ENV，使笔记本电脑和集群上的部署保持可复现。

!!! tip "便于复制的默认设置"
    本站每个页面的代码块都带有复制按钮和提示框，方便团队成员直接使用所需片段。使用搜索快捷键（`/` 或 `cmd/ctrl + K`）可直接跳转到实体、API 调用或 API 接口。

## 初次体验

```python
from openmed import analyze_text

result = analyze_text(
    "Patient started on imatinib for chronic myeloid leukemia.",
    model_name="disease_detection_superclinical",
    confidence_threshold=0.55,
)

for entity in result.entities:
    print(entity.label, entity.text, entity.confidence)
```

```bash
uv pip install "openmed[hf]"
uv run python examples/pii_model_comparison.py
```

后续文档将详细展开此示例。请先阅读**快速开始**完成端到端设置，然后探索配置、zero-shot GLiNER 工作流和高级处理工具等指南。

## 最新版本亮点

- [OpenMed 1.9.1 发布说明](./release/v1.9.1.md) — Swift 打包、Android 发布强化、当前模型示例，以及跨平台 1.9 版本的依赖安全修复。
- [OpenMed 1.8.0 发布说明](./release/v1.8.0.md) — 历史跨平台运行时与服务版本清单。
- [OpenMed v1.6-v1.7 功能覆盖](./release/v1.6-v1.7-feature-coverage.md) — 示例、文档、网站和源代码模块的历史覆盖清单。
- [示例与复制即用的配方](./examples.md) — 面向 Python、PII、批处理作业、Apple 运行时、浏览器导出、多模态输入以及 FHIR/HL7 的发布级片段。
- [Transformers.js 导出](./export-transformersjs.md) — 面向词元分类包的浏览器/WebGPU 打包。
- [FHIR 互操作工具](./fhir-interop.md)、[HL7 v2 去标识化](./hl7v2-deidentification.md)和 [OMOP/lakehouse 集成](./integrations/lakehouse-redaction.md) — 结构化健康数据工作流。
- [MLX 后端](./mlx-backend.md)、[OpenMedKit](./swift-openmedkit.md)、[Android span 一致性](./android-parity.md)和 [CoreML 打包](./coreml-export.md) — 本地移动端与运行时路径。

## 文档结构

1. [快速开始](./getting-started.md) — 最快建立可用环境并运行可复制脚本的路径。
2. [功能地图](./feature-map.md) — 查看每项能力如何映射到代码。
3. [OpenMed 1.9.1 发布说明](./release/v1.9.1.md) — 查看当前补丁修复、安装坐标和验证证据。
4. 核心指南：
   - [文本分析辅助函数](./analyze-text.md)：单次调用推理。
   - [REST 服务（MVP）](./rest-service.md)：容器化 HTTP 端点。
   - [PII 检测与智能合并](./pii-smart-merging.md)：兼顾 HIPAA 要求的去标识化。
   - [批处理](./batch-processing.md)：处理多个文本或文件。
   - [ModelLoader 与流水线](./model-loader.md)：运行长时间任务。
   - [模型注册表](./model-registry.md)：选择合适的检查点。
   - [配置文件](./profiles.md)：在开发、生产和测试设置间切换。
   - [高级 NER 与输出格式](./output-formatting.md)：完善实体范围。
   - [医疗感知分词器](./medical-tokenizer.md)：改进临床词元边界。
   - [配置与验证](./configuration.md)：保持部署可复现。
   - [Zero-shot 工具包](./zero-shot-ner.md)：构建 GLiNER 工作流。
   - [性能分析](./profiling.md)：计时与优化。
   - [示例](./examples.md)和[测试与质量保证](./testing.md)：日常操作。
5. 项目运维：
   - [贡献与发布](./contributing.md) — 如何发布版本和文档并保持 CI 通过。
   - [发布流与渠道](./release/semver-and-channels.md) — 模型制品与软件库的发布策略。
   - [生成式模型策略](./generative-model-policy.md) — 获准与禁止的模型辅助工作流。

如果你需要的内容尚未覆盖，请在 GitHub 提交 issue 并说明缺少的配方。每项新增内容都可以从一个 Markdown 文件开始。
