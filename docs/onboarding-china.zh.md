# 中国用户 OpenMed 上手指南

[English](onboarding-china.md)

在默认 PyPI 或 Hugging Face 端点速度较慢、无法稳定访问的网络环境中，
本指南帮助你完成首次 OpenMed 安装和模型下载。内容包括 Python 软件包镜像、
Hugging Face 模型镜像、可迁移的本地缓存，以及内置的 PIPL 技术策略。

以下镜像均为独立的第三方服务，并非 OpenMed 运营或背书的基础设施。投入生产前，
请遵守组织的软件供应链要求，固定软件包和模型版本，并校验下载的产物。

如需完整的合成临床病历演示，请运行
[简体中文示例](https://github.com/maziyarpanahi/openmed/blob/master/examples/deid_chinese_clinical_note.py)，
或参阅
[中文和印地语 Notebook 教程](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Chinese_Hindi_Deid_Tour.ipynb)。

## 通过 PyPI 镜像安装 OpenMed

临时使用清华大学 TUNA 镜像安装：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple openmed
```

如果当前机器需要下载或检查 Hugging Face 模型快照，请安装可选依赖：

```bash
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple "openmed[hf]"
```

地址中的 `simple` 路径不可省略。TUNA 的
[PyPI 镜像帮助](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)
还提供了等价的完整服务地址。

也可以使用阿里云公共镜像：

```bash
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ openmed
```

[阿里云 PyPI 镜像页面](https://developer.aliyun.com/mirror/pypi/)
列出了公共网络和 ECS 专用端点。

### 在 `pip.conf` 中固定镜像

让 pip 修改当前用户的配置，可以避免误改未知的系统级文件：

```bash
python -m pip config --user set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip config --user get global.index-url
```

运行 `python -m pip config debug` 可查看实际生效的 `pip.conf` 位置。等价的文件内容为：

```ini
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
```

如需默认使用阿里云，请将 URL 替换为
`https://mirrors.aliyun.com/pypi/simple/`。镜像同步可能滞后于官方索引，
因此应固定准确版本；遇到意外的版本差异时，先核查原因，再决定是否切换索引。

## 通过 HF-Mirror 下载模型

安装 `openmed[hf]` 后，必须在当前进程导入 `huggingface_hub`、Transformers
或 OpenMed **之前**设置模型镜像端点：

=== "Linux 和 macOS"

    ```bash
    export HF_ENDPOINT=https://hf-mirror.com
    ```

=== "Windows PowerShell"

    ```powershell
    $env:HF_ENDPOINT = "https://hf-mirror.com"
    ```

这是 [HF-Mirror](https://hf-mirror.com/) 文档提供的环境变量用法。
它只改变兼容的 Hugging Face 客户端获取模型文件的位置，不会把 OpenMed 推理转发到托管 API。

## 预下载一次，随后只使用本地缓存

除非通过 `HF_HOME` 或 `HF_HUB_CACHE` 指定其他位置，Hugging Face 标准缓存位于
`~/.cache/huggingface`。如果需要把缓存打包或复制到隔离网络中的机器，建议使用独立的
`HF_HOME` 目录。

### 1. 在联网机器上预热缓存

在 OpenMed 源代码检出目录中，示例脚本只有在明确允许后才会下载：

```bash
export HF_HOME="$PWD/openmed-hf-cache"
OPENMED_EXAMPLE_ALLOW_DOWNLOAD=1 \
  python examples/onboarding_china_mirrors.py
```

示例会先设置 `HF_ENDPOINT=https://hf-mirror.com`，再延迟导入 Hub 客户端，
并缓存默认的 44M PII 模型。如果只安装了软件包，也可以直接预下载：

```bash
export HF_HOME="$PWD/openmed-hf-cache"
export HF_ENDPOINT=https://hf-mirror.com
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1"))'
```

将整个 `openmed-hf-cache` 目录打包或复制到离线机器，并保留目录结构和符号链接。

### 2. 禁止所有 Hub 网络流量，只解析缓存

在离线机器上，让 `HF_HOME` 指向已传输的目录，并在启动 Python 前启用离线模式：

```bash
export HF_HOME="$PWD/openmed-hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python examples/onboarding_china_mirrors.py
```

示例还会传入 `local_files_only=True`。如果快照不存在，它只报告缓存未命中，
不会尝试网络请求。模型缓存完成后，同样的变量可以包裹实际应用：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python your_openmed_app.py
```

关于 `HF_HOME`、`HF_HUB_CACHE` 和 `HF_HUB_OFFLINE` 的准确行为，请参阅
[Hugging Face Hub 环境变量文档](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables)。

## 面向《个人信息保护法》的策略快速入门

《个人信息保护法》（PIPL）合规取决于处理目的、同意或其他合法性基础、数据分类、
保存期限、跨境传输、访问控制和人工治理。软件策略配置只是其中一项技术控制；
本节不是法律意见，也不表示使用某个配置即可实现 PIPL 合规。

OpenMed 已提供面向敏感个人信息的 `china_pipl` 技术策略。该策略使用高召回率的
`strict_no_leak` 阈值配置，强制执行安全扫描，替换直接标识符，并遮蔽准标识符和
临床概念：

```python
from openmed.core.policy import list_policies, load_policy

policy_name = "china_pipl"
assert policy_name in list_policies()

policy = load_policy(policy_name)
assert policy.name == "china_pipl"
assert policy.threshold_profile == "strict_no_leak"
assert policy.safety_sweep_mandatory
assert policy.default_action == "replace"
assert policy.keep_mapping is True
assert policy.reversible_id is True

print(policy.name, policy.default_action)
```

这段代码完全从已安装的软件包中加载
`openmed/core/policies/china_pipl.json`，不依赖模型或网络。
在常规去标识化调用中，可以把同一个 `policy_name` 传给
`deidentify(synthetic_text, policy=policy_name)`。该策略会为替换流程保留可逆映射，
因此其输出在 PIPL 下仍属于个人信息；只有不可逆匿名化才可能使数据脱离 PIPL 的适用范围。
在生产环境处理个人信息或受保护健康信息 (PHI) 前，请重新评估技术策略和法律治理措施。

## 隐私与网络边界

!!! important
    **镜像只影响软件包和模型下载。OpenMed 推理始终完全在本地运行，OpenMed
    库不会发送遥测数据。** 所需产物缓存完成后，`HF_HUB_OFFLINE=1` 和
    `TRANSFORMERS_OFFLINE=1` 会为兼容的 Hub 和 Transformers 加载器建立明确的
    仅缓存边界。

软件包安装器和模型下载客户端属于第三方工具，有各自的配置与政策。请勿在安装命令、
模型仓库名称、日志或镜像请求中放入真实的个人身份信息 (PII) 或受保护健康信息 (PHI)。
验证配置时只使用合成数据。

## 故障排查

- `No matching distribution found`：确认镜像已同步所需的 OpenMed 版本，
  再与 PyPI 官方发布版本比较。
- 离线模式出现 `LocalEntryNotFoundError`：传输的缓存中缺少所选模型或修订版本，
  需要在联网机器上预热该准确快照。
- 机器仍产生出站请求：确保所有环境变量都在 Python 启动前设置，并审计应用中的
  其他依赖是否自带网络客户端。
- 受限或私有模型无法通过镜像获取：遵守模型所有者的访问条款和组织的凭据政策；
  不要把令牌写入 shell 历史、源代码或文档。
