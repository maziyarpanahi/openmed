<div align="center">

<img src="docs/brand/openmed-readme-banner.png" alt="Banner README do OpenMed com o mascote gato, a marca nominativa em minúsculas, Open Cross e o texto IA de saúde de código aberto, 340 M+ de downloads e 10 M+ de instalações" width="1280" />

<h3>Seus dados. Seu modelo. Seu hardware.</h3>

<p><b>Transforme texto clínico em informação estruturada e desidentificada no hardware que você controla.</b><br/>
O runtime local principal do OpenMed realiza extração e desidentificação depois que os artefatos de modelo necessários estão disponíveis. Downloads de modelos, adaptadores de provedores remotos, caminhos com telemetria e integrações configuradas pelo usuário podem usar a rede; revise os termos de cada modelo e conjunto de dados.</p>

<p>
  <a href="https://pypi.org/project/openmed/">PyPI package</a> ·
  <a href="https://www.python.org/downloads/">Python 3.10+</a> ·
  <a href="https://huggingface.co/OpenMed">Model catalog</a> ·
  <a href="https://arxiv.org/abs/2508.01630">Research paper</a> ·
  <a href="LICENSE">Apache-2.0 SDK source</a>
</p>

<p>
  <a href="swift/OpenMedKit">OpenMedKit</a> ·
  <a href="docs/mlx-backend.md">Apple Silicon / MLX</a> ·
  <a href="docs/export-onnx-android.md">Android / ONNX Runtime Mobile</a> ·
  <a href="docs/export-transformersjs.md">Browser / Transformers.js</a> ·
  <a href="https://openmed.life/docs">Documentation</a>
</p>

<p>
  <b>Execução local em primeiro lugar</b> &nbsp;·&nbsp; <b>34 idiomas PII com suporte de modelos</b> &nbsp;·&nbsp; <b>Apache-2.0 SDK</b>
</p>

<p>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.it.md">Italiano</a> ·
  <b>Português</b> ·
  <a href="README.nl.md">Nederlands</a> ·
  <a href="README.ar.md">العربية</a> ·
  <a href="README.hi.md">हिन्दी</a> ·
  <a href="README.te.md">తెలుగు</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.tr.md">Türkçe</a> ·
  <a href="README.fa.md">فارسی</a>
</p>

</div>

---

## Veja em ação

<div align="center">
  <img src="docs/brand/openmed-pii-demo.gif" alt="OpenMed des-identificando PII de um relatório de alta clínica em tempo real" width="760" />
  <br/>
  <sub><b>Des-identificação de PII em tempo real</b>: o Privacy Filter Nemotron oculta nomes, endereços, identificadores e dados de faturamento de um relatório de alta clínica, totalmente no dispositivo. <i>(Todos os valores exibidos são sintéticos.)</i></sub>
</div>

---

## Exemplo em 30 segundos

```python
from openmed import analyze_text

result = analyze_text(
    "Patient started on imatinib for chronic myeloid leukemia.",
    model_name="disease_detection_superclinical",
)

for entity in result.entities:
    print(f"{entity.label:<12} {entity.text:<28} {entity.confidence:.2f}")
# DISEASE      chronic myeloid leukemia     0.98
# DRUG         imatinib                     0.95
```

Um modelo de NER clínico usa o runtime local depois que os artefatos necessários estão disponíveis.

---

## Por que OpenMed?

| Consideração de implantação | Limite do SDK OpenMed |
| --- | --- |
| Runtime principal | Processa localmente após os artefatos necessários estarem disponíveis |
| Caminhos de rede opcionais | Downloads, adaptadores remotos, telemetria e integrações podem usar a rede |
| Validação | O responsável valida termos de modelos e dados, privacidade e adequação clínica |
| Interfaces | Python, Swift, Android, navegador e serviços quando compatíveis |

- **Catálogo de modelos selecionado**: valide cada modelo, licença e conjunto de dados para seu caso de uso.
- **Configuração alinhada ao Safe Harbor**: pode abranger as 18 categorias de identificadores; a revisão especializada da implantação continua necessária e o uso do SDK, isoladamente, não comprova conformidade com HIPAA.
- **Caminhos de execução compatíveis**: adaptadores de CPU, CUDA, MLX, dispositivo móvel, serviço e navegador variam conforme o ambiente e o artefato.
- **Interfaces de implantação**: Python, contêineres, serviços e fluxos em lote exigem configuração e validação.
- **Código-fonte do SDK**: publicado sob Apache-2.0 License; os termos de modelos e conjuntos de dados variam.

---

## No dispositivo, na Apple: Swift, MLX e iOS

Em hardware Apple compatível, o OpenMed pode usar **MLX** e **[OpenMedKit](swift/OpenMedKit)** para processamento local após os artefatos necessários estarem disponíveis. A obtenção de modelos e as integrações remotas configuradas pelo usuário continuam sendo limites de rede separados.

```swift
// Add OpenMedKit to your app
dependencies: [
    .package(url: "https://github.com/maziyarpanahi/openmed.git", from: "2.0.0"),
]
```

- **Runtime MLX** para classificação de tokens de PII, a família Privacy Filter e tarefas zero-shot experimentais da família GLiNER, com um caminho de fallback em CoreML.
- **Um nome de modelo, todas as plataformas**: em hardware que não é Apple, os nomes de modelo MLX recorrem automaticamente ao checkpoint PyTorch correspondente.
- **Python no Apple Silicon** também: `pip install --upgrade "openmed[mlx]"`.

Guias: [Backend MLX](docs/mlx-backend.md) · [OpenMedKit (Swift)](docs/swift-openmedkit.md) · [Exportação CoreML](docs/coreml-export.md)

---

## Como funciona

```mermaid
flowchart LR
    A["Texto clínico"] --> B["OpenMed<br/>(local primeiro)"]
    B --> C["Entidades médicas"]
    B --> D["PII detectada"]
    B --> E["Texto des-identificado"]
    style B fill:#0D6E6E,stroke:#0A5656,stroke-width:2px,color:#ffffff
    style C fill:#D6EBEB,stroke:#0D6E6E,color:#0E1116
    style D fill:#F7DCD8,stroke:#C5453A,color:#0E1116
    style E fill:#F5E27A,stroke:#A9A088,color:#0E1116
```

---

## Início rápido

```bash
# Core + Hugging Face runtime (Linux, macOS, Windows; CPU or CUDA)
pip install --upgrade "openmed[hf]"

# Add the REST service
pip install --upgrade "openmed[hf,service]"

# Apple Silicon acceleration (MLX)
pip install --upgrade "openmed[mlx]"
```

<table>
<tr>
<td width="33%" valign="top">

**API Python**

```python
from openmed import analyze_text

analyze_text(
  "Patient received 75mg "
  "clopidogrel for NSTEMI.",
  model_name=
  "pharma_detection_superclinical",
)
```

</td>
<td width="33%" valign="top">

**Serviço REST**

```bash
uvicorn openmed.service.app:app \
  --host 0.0.0.0 --port 8080
```

`GET /health`
`POST /analyze`
`POST /pii/extract`
`POST /pii/deidentify`

</td>
<td width="33%" valign="top">

**Em lote**

```python
from openmed import BatchProcessor

p = BatchProcessor(
  model_name=
  "disease_detection_superclinical",
  group_entities=True,
)
p.process_texts([...])
```

</td>
</tr>
</table>

**Offline / isolado?** Aponte `model_name` (ou `model_id`) para um diretório local e o OpenMed o carrega sem contatar o Hugging Face Hub:

```python
from openmed import OpenMedConfig, analyze_text

result = analyze_text(
    "Patient presents with chronic myeloid leukemia and Type 2 diabetes.",
    model_id="./models/OpenMed-NER-DiseaseDetect-SuperClinical-434M",
    config=OpenMedConfig(device="cpu"),
)
```

---

## Modelos

Um registro curado de modelos de NER médico especializados: explore o [catálogo completo](https://openmed.life/docs/model-registry).

| Modelo | Especialização | Tipos de entidade | Tamanho |
|--------|----------------|-------------------|---------|
| `disease_detection_superclinical` | Doenças e condições | DISEASE, CONDITION, DIAGNOSIS | 434M |
| `pharma_detection_superclinical`  | Fármacos e medicamentos | DRUG, MEDICATION, TREATMENT   | 434M |
| `pii_superclinical_large`     | PII e des-identificação | NAME, DATE, SSN, PHONE, EMAIL, ADDRESS | 434M |
| `anatomy_detection_electramed`    | Anatomia e partes do corpo | ANATOMY, ORGAN, BODY_PART     | 109M |
| `gene_detection_genecorpus`       | Genes e proteínas | GENE, PROTEIN                 | 109M |

---

## Privacidade: detecção e des-identificação de PII

```python
from openmed import extract_pii, deidentify

text = "Patient: John Doe, DOB: 01/15/1970, SSN: 123-45-6789"

# Extract PII with smart merging (prevents tokenization fragmentation)
result = extract_pii(text, model_name="pii_superclinical_large", use_smart_merging=True)

# De-identify with the method you need
deidentify(text, method="mask")     # [NAME], [DATE]
deidentify(text, method="replace")  # Faker-backed, locale-aware, format-preserving fakes
deidentify(text, method="hash")     # Cryptographic hashing
deidentify(text, method="shift_dates", date_shift_days=180)
```

- **A mesclagem inteligente de entidades** mantém `01/15/1970` inteiro em vez de fragmentá-lo.
- **Ofuscação baseada em Faker** com provedores personalizados de identificadores clínicos (CPF, CNPJ, BSN, NIR, Codice Fiscale, NIE, Aadhaar, Steuer-ID, NPI).
- **Limite da HIPAA**: categorias alinhadas ao Safe Harbor e limiares configuráveis auxiliam a implementação; a revisão especializada da implantação continua necessária e o uso isolado do SDK não comprova conformidade.

[Notebook completo de PII](examples/notebooks/PII_Detection_Complete_Guide.ipynb) · [Mesclagem inteligente](docs/pii-smart-merging.md) · [Anonimização](docs/anonymization.md)

<details>
<summary><b>Família Privacy Filter</b>: três famílias de modelos sobre a arquitetura OpenAI Privacy Filter</summary>

<br/>

O código do modelo é o mesmo (transformer MoE esparso no estilo gpt-oss com atenção local, tokens sink, RoPE+YaRN, tokenização tiktoken `o200k_base`); apenas os dados de treinamento mudam. Todos usam a **mesma** API `extract_pii()` / `deidentify()`: só muda o argumento `model_name=`.

| Variante | PyTorch (CPU + CUDA) | MLX (Apple Silicon) | MLX 8-bit |
| --- | --- | --- | --- |
| **OpenAI Privacy Filter** | [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) | [`OpenMed/privacy-filter-mlx`](https://huggingface.co/OpenMed/privacy-filter-mlx) | [`…-mlx-8bit`](https://huggingface.co/OpenMed/privacy-filter-mlx-8bit) |
| **Nemotron-PII fine-tune** | [`OpenMed/privacy-filter-nemotron`](https://huggingface.co/OpenMed/privacy-filter-nemotron) | [`…-nemotron-mlx`](https://huggingface.co/OpenMed/privacy-filter-nemotron-mlx) | [`…-nemotron-mlx-8bit`](https://huggingface.co/OpenMed/privacy-filter-nemotron-mlx-8bit) |
| **OpenMed Multilingual** | [`OpenMed/privacy-filter-multilingual`](https://huggingface.co/OpenMed/privacy-filter-multilingual) | [`…-multilingual-mlx`](https://huggingface.co/OpenMed/privacy-filter-multilingual-mlx) | [`…-multilingual-mlx-8bit`](https://huggingface.co/OpenMed/privacy-filter-multilingual-mlx-8bit) |

```python
from openmed import extract_pii

text = "Patient Sarah Connor (DOB: 03/15/1985) at MRN 4471882."

extract_pii(text, model_name="openai/privacy-filter")              # PyTorch baseline
extract_pii(text, model_name="OpenMed/privacy-filter-nemotron")    # same code, different weights
extract_pii(text, model_name="OpenMed/privacy-filter-mlx")         # Apple Silicon (MLX)
```

Em hosts que não são Apple Silicon, os nomes de modelo MLX são substituídos automaticamente pelo checkpoint PyTorch correspondente (com um aviso único): escreva um nome de modelo e rode em qualquer lugar. Veja [Arquitetura do Privacy Filter e roteamento de backend](docs/anonymization.md#privacy-filter-family).

</details>

---

## PII multilíngue (35 rotas suportadas; 34 com suporte de modelos)

Extração e des-identificação em `en`, `fr`, `de`, `it`, `es`, `nl`, `hi`, `te`, `pt`, `ar`, `ja` e `tr`, **o catálogo registrado de modelos PII** no total.

```bash
python -c "from openmed import extract_pii; print([(e.label, e.text) for e in extract_pii('Dr. Pedro Almeida, CPF: 123.456.789-09, email: pedro@hospital.pt', lang='pt').entities])"
```

<details>
<summary>Ver exemplos por idioma (português, holandês, hindi, árabe, japonês, turco)</summary>

<br/>

```python
from openmed import extract_pii

portuguese = extract_pii("Paciente: Pedro Almeida, CPF: 123.456.789-09, telefone: +351 912 345 678", lang="pt", use_smart_merging=True)
dutch      = extract_pii("Patiënt: Eva de Vries, BSN: 123456782, telefoon: +31 6 12345678", lang="nl", use_smart_merging=True)
hindi      = extract_pii("रोगी: अनीता शर्मा, फोन: +91 9876543210, पता: नई दिल्ली 110001", lang="hi", use_smart_merging=True)
arabic     = extract_pii("المريضة ليلى حسن، الهاتف +20 10 1234 5678، الرقم القومي 29801011234567.", lang="ar", use_smart_merging=True)
japanese   = extract_pii("患者 佐藤 花子、電話 +81 90 1234 5678、マイナンバー 1234 5678 9012.", lang="ja", use_smart_merging=True)
turkish    = extract_pii("Hasta Ayşe Yılmaz, telefon +90 532 123 45 67, TCKN 10000000146.", lang="tr", use_smart_merging=True)

for r in (portuguese, dutch, hindi, arabic, japanese, turkish):
    print([(e.label, e.text) for e in r.entities])
```

</details>

---

## REST API

Um serviço FastAPI amigável ao Docker, com validação de requisições, pré-carregamento de pipeline compartilhado e envelopes de erro unificados.

```bash
pip install --upgrade "openmed[hf,service]"
uvicorn openmed.service.app:app --host 0.0.0.0 --port 8080

# or with Docker
docker build -t openmed:local .
docker run --rm -p 8080:8080 -e OPENMED_PROFILE=prod openmed:local
```

```bash
curl -X POST http://127.0.0.1:8080/pii/extract \
  -H "Content-Type: application/json" \
  -d '{"text":"Paciente: Maria Garcia, DNI: 12345678Z","lang":"es"}'
```

Veja o [guia completo do serviço REST](docs/rest-service.md).

---

## Documentação

Guias completos em **[openmed.life/docs](https://openmed.life/docs/)**.

| | | |
|---|---|---|
| [Primeiros passos](https://openmed.life/docs/) | [Analisar texto](https://openmed.life/docs/analyze-text) | [Registro de modelos](https://openmed.life/docs/model-registry) |
| [Guia de detecção de PII](examples/notebooks/PII_Detection_Complete_Guide.ipynb) | [Anonimização](docs/anonymization.md) | [Processamento em lote](https://openmed.life/docs/batch-processing) |
| [Perfis de configuração](https://openmed.life/docs/profiles) | [Serviço REST](docs/rest-service.md) | [Backend MLX](docs/mlx-backend.md) |

---

## Conheça o mascote

<img src="docs/brand/openmed-mascot-icon.png" alt="Mascote do OpenMed" width="104" align="left" />

O guardião do OpenMed é um gato persa fofo caracterizado como um pequeno **Avicena (Ibn Sina)**, o grande
médico persa cujo *Cânone da Medicina* foi o texto médico de referência no mundo todo por cerca de 600 anos.
Ele cuida do livro aberto do conhecimento médico, com uma paleta inspirada na **turquesa persa (fīrūza)**: um
guardião local-first para os seus dados mais privados.

<br clear="left"/>

---

## Contribuir

Contribuições são bem-vindas: relatórios de bugs, pedidos de recursos e PRs.

- [Abrir uma issue](https://github.com/maziyarpanahi/openmed/issues)
- **Traduções são bem-vindas**: ajude a completar os README em outros idiomas vinculados no seletor no topo.

---

## Créditos

O OpenMed se baseia em excelente trabalho open source: agradecimento especial à **OpenAI** (a arquitetura [Privacy Filter](https://huggingface.co/openai/privacy-filter)), à **NVIDIA** (o [conjunto de dados Nemotron PII](https://huggingface.co/datasets/nvidia/Nemotron-PII-v1)), à **Hugging Face** (`transformers` e o ecossistema de modelos), à **Apple** ([MLX](https://github.com/ml-explore/mlx)) e aos mantenedores do **[Faker](https://faker.readthedocs.io/)**.

## Licença

O código-fonte do SDK OpenMed é publicado sob a [Apache-2.0 License](LICENSE).

## Citação

Se o OpenMed for útil na sua pesquisa, por favor, cite:

```bibtex
@misc{panahi2025openmedneropensourcedomainadapted,
      title={OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art Transformers for Biomedical NER Across 12 Public Datasets},
      author={Maziyar Panahi},
      year={2025},
      eprint={2508.01630},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2508.01630},
}
```

---

## Histórico de estrelas

Se o OpenMed for útil para você, uma estrela ajuda outros a descobri-lo.

[4,700+ GitHub stars · 29 Jul 2026 snapshot](https://github.com/maziyarpanahi/openmed/stargazers)

---

<div align="center">

Feito pela equipe OpenMed

<a href="https://openmed.life">Site</a> ·
<a href="https://openmed.life/docs">Documentação</a> ·
<a href="https://x.com/openmed_ai">X / Twitter</a> ·
<a href="https://www.linkedin.com/company/openmed-ai/">LinkedIn</a>

</div>
