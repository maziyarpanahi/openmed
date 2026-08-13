<div align="center">

<img src="docs/brand/openmed-readme-banner.png" alt="Banner README di OpenMed con la mascotte gatto, il marchio in minuscolo, Open Cross e il testo IA sanitaria open source, oltre 340 milioni di download e oltre 10 milioni di installazioni" width="1280" />

<h3>I tuoi dati. Il tuo modello. Il tuo hardware.</h3>

<p><b>Trasforma il testo clinico in informazioni strutturate e de-identificate sull’hardware che controlli.</b><br/>
Il runtime locale principale di OpenMed esegue estrazione e de-identificazione dopo che gli artefatti di modello richiesti sono disponibili. Download dei modelli, adattatori per provider remoti, percorsi con telemetria e integrazioni configurate dall’utente possono usare la rete; verifica i termini di ogni modello e set di dati.</p>

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
  <b>Esecuzione locale prioritaria</b> &nbsp;·&nbsp; <b>33 lingue PII supportate da modelli</b> &nbsp;·&nbsp; <b>Apache-2.0 SDK</b>
</p>

<p>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <b>Italiano</b> ·
  <a href="README.pt.md">Português</a> ·
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

## Guardalo in azione

<div align="center">
  <img src="docs/brand/openmed-pii-demo.gif" alt="OpenMed de-identifica i PII da una lettera di dimissione clinica in tempo reale" width="760" />
  <br/>
  <sub><b>De-identificazione dei PII in tempo reale</b>: il Privacy Filter Nemotron oscura nomi, indirizzi, identificativi e dati di fatturazione da una lettera di dimissione clinica, interamente sul dispositivo. <i>(Tutti i valori mostrati sono sintetici.)</i></sub>
</div>

---

## Esempio in 30 secondi

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

Un modello NER clinico usa il runtime locale dopo che i suoi artefatti richiesti sono disponibili.

---

## Perché OpenMed?

| Aspetto di distribuzione | Perimetro dell’SDK OpenMed |
| --- | --- |
| Runtime principale | Elabora localmente dopo la disponibilità degli artefatti richiesti |
| Percorsi di rete opzionali | Download, adattatori remoti, telemetria e integrazioni possono usare la rete |
| Validazione | Il responsabile verifica termini di modelli e dati, privacy e idoneità clinica |
| Interfacce | Python, Swift, Android, browser e servizi dove supportati |

- **Catalogo di modelli selezionato**: convalida ogni modello, licenza e set di dati per il tuo caso d’uso.
- **Configurazione allineata a Safe Harbor**: può individuare le 18 categorie di identificatori; resta necessaria una revisione esperta del deployment e l’uso dell’SDK non dimostra da solo la conformità HIPAA.
- **Percorsi di esecuzione supportati**: gli adattatori CPU, CUDA, MLX, mobile, servizio e browser variano in base ad ambiente e artefatto.
- **Interfacce di deployment**: Python, container, servizi e flussi batch richiedono configurazione e convalida.
- **Codice sorgente dell’SDK**: distribuito con Apache-2.0 License; i termini di modelli e set di dati variano.

---

## Sul dispositivo, su Apple: Swift, MLX e iOS

Sull’hardware Apple supportato, OpenMed può usare **MLX** e **[OpenMedKit](swift/OpenMedKit)** per l’elaborazione locale dopo la disponibilità degli artefatti richiesti. L’acquisizione dei modelli e le integrazioni remote configurate dall’utente restano confini di rete separati.

```swift
// Add OpenMedKit to your app
dependencies: [
    .package(url: "https://github.com/maziyarpanahi/openmed.git", from: "2.1.0"),
]
```

- **Runtime MLX** per la classificazione dei token PII, la famiglia Privacy Filter e le attività zero-shot sperimentali della famiglia GLiNER, con un percorso di fallback CoreML.
- **Un solo nome di modello, tutte le piattaforme**: su hardware non Apple, i nomi dei modelli MLX ricadono automaticamente sul checkpoint PyTorch corrispondente.
- **Python su Apple Silicon** anche: `pip install --upgrade "openmed[mlx]"`.

Guide: [Backend MLX](docs/mlx-backend.md) · [OpenMedKit (Swift)](docs/swift-openmedkit.md) · [Esportazione CoreML](docs/coreml-export.md)

---

## Come funziona

```mermaid
flowchart LR
    A["Testo clinico"] --> B["OpenMed<br/>(locale per impostazione)"]
    B --> C["Entità mediche"]
    B --> D["PII rilevati"]
    B --> E["Testo de-identificato"]
    style B fill:#0D6E6E,stroke:#0A5656,stroke-width:2px,color:#ffffff
    style C fill:#D6EBEB,stroke:#0D6E6E,color:#0E1116
    style D fill:#F7DCD8,stroke:#C5453A,color:#0E1116
    style E fill:#F5E27A,stroke:#A9A088,color:#0E1116
```

---

## Avvio rapido

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

**Servizio REST**

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

**Batch**

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

**Offline / isolato?** Punta `model_name` (o `model_id`) a una directory locale e OpenMed la carica senza contattare l'Hub di Hugging Face:

```python
from openmed import OpenMedConfig, analyze_text

result = analyze_text(
    "Patient presents with chronic myeloid leukemia and Type 2 diabetes.",
    model_id="./models/OpenMed-NER-DiseaseDetect-SuperClinical-434M",
    config=OpenMedConfig(device="cpu"),
)
```

---

## Modelli

Un registro curato di modelli NER medici specializzati: esplora il [catalogo completo](https://openmed.life/docs/model-registry).

| Modello | Specializzazione | Tipi di entità | Dimensione |
|---------|------------------|----------------|------------|
| `disease_detection_superclinical` | Malattie e condizioni | DISEASE, CONDITION, DIAGNOSIS | 434M |
| `pharma_detection_superclinical`  | Farmaci e terapie | DRUG, MEDICATION, TREATMENT   | 434M |
| `pii_superclinical_large`     | PII e de-identificazione | NAME, DATE, SSN, PHONE, EMAIL, ADDRESS | 434M |
| `anatomy_detection_electramed`    | Anatomia e parti del corpo | ANATOMY, ORGAN, BODY_PART     | 109M |
| `gene_detection_genecorpus`       | Geni e proteine | GENE, PROTEIN                 | 109M |

---

## Privacy: rilevamento e de-identificazione dei PII

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

- **La fusione intelligente delle entità** mantiene `01/15/1970` intero invece di frammentarlo.
- **Offuscamento basato su Faker** con provider personalizzati di identificativi clinici (CPF, CNPJ, BSN, NIR, Codice Fiscale, NIE, Aadhaar, Steuer-ID, NPI).
- **Perimetro HIPAA**: categorie allineate a Safe Harbor e soglie configurabili sono strumenti di implementazione; resta necessaria una revisione esperta del deployment e l’uso del solo SDK non dimostra la conformità.

[Notebook PII completo](examples/notebooks/PII_Detection_Complete_Guide.ipynb) · [Fusione intelligente](docs/pii-smart-merging.md) · [Anonimizzazione](docs/anonymization.md)

<details>
<summary><b>Famiglia Privacy Filter</b>: tre famiglie di modelli sull'architettura OpenAI Privacy Filter</summary>

<br/>

Il codice del modello è identico (transformer MoE sparso in stile gpt-oss con attenzione locale, token sink, RoPE+YaRN, tokenizzazione tiktoken `o200k_base`); cambiano solo i dati di addestramento. Tutti usano la **stessa** API `extract_pii()` / `deidentify()`: cambia solo l'argomento `model_name=`.

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

Sugli host non Apple Silicon, i nomi dei modelli MLX vengono sostituiti automaticamente con il checkpoint PyTorch corrispondente (con un avviso una tantum): scrivi un solo nome di modello, eseguilo ovunque. Vedi [Architettura Privacy Filter e routing del backend](docs/anonymization.md#privacy-filter-family).

</details>

---

## PII multilingue (35 route supportate; 33 supportate da modelli)

Estrazione e de-identificazione in `en`, `fr`, `de`, `it`, `es`, `nl`, `hi`, `te`, `pt`, `ar`, `ja` e `tr`: **il catalogo registrato dei modelli PII** in totale.

```bash
python -c "from openmed import extract_pii; print([(e.label, e.text) for e in extract_pii('Dr. Pedro Almeida, CPF: 123.456.789-09, email: pedro@hospital.pt', lang='pt').entities])"
```

<details>
<summary>Mostra esempi per lingua (portoghese, olandese, hindi, arabo, giapponese, turco)</summary>

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

Un servizio FastAPI compatibile con Docker, con validazione delle richieste, precaricamento della pipeline condivisa e involucri di errore unificati.

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

Consulta la [guida completa al servizio REST](docs/rest-service.md).

---

## Documentazione

Guide complete su **[openmed.life/docs](https://openmed.life/docs/)**.

| | | |
|---|---|---|
| [Per iniziare](https://openmed.life/docs/) | [Analizza testo](https://openmed.life/docs/analyze-text) | [Registro dei modelli](https://openmed.life/docs/model-registry) |
| [Guida al rilevamento PII](examples/notebooks/PII_Detection_Complete_Guide.ipynb) | [Anonimizzazione](docs/anonymization.md) | [Elaborazione batch](https://openmed.life/docs/batch-processing) |
| [Profili di configurazione](https://openmed.life/docs/profiles) | [Servizio REST](docs/rest-service.md) | [Backend MLX](docs/mlx-backend.md) |

---

## Conosci la mascotte

<img src="docs/brand/openmed-mascot-icon.png" alt="Mascotte di OpenMed" width="104" align="left" />

Il guardiano di OpenMed è un soffice gatto persiano nei panni di un piccolo **Avicenna (Ibn Sina)**, il grande
medico persiano il cui *Canone della medicina* fu il testo medico di riferimento nel mondo per circa 600 anni.
Veglia sul libro aperto del sapere medico, con una palette ispirata al **turchese persiano (fīrūza)**: un
guardiano local-first per i tuoi dati più riservati.

<br clear="left"/>

---

## Contribuire

I contributi sono benvenuti: segnalazioni di bug, richieste di funzionalità e PR.

- [Apri una issue](https://github.com/maziyarpanahi/openmed/issues)
- **Traduzioni benvenute**: aiuta a completare i README nelle altre lingue collegati nel selettore in alto.

---

## Ringraziamenti

OpenMed si basa su eccellente lavoro open source: un ringraziamento speciale a **OpenAI** (l'architettura [Privacy Filter](https://huggingface.co/openai/privacy-filter)), **NVIDIA** (il [dataset Nemotron PII](https://huggingface.co/datasets/nvidia/Nemotron-PII-v1)), **Hugging Face** (`transformers` e l'ecosistema di modelli), **Apple** ([MLX](https://github.com/ml-explore/mlx)) e i manutentori di **[Faker](https://faker.readthedocs.io/)**.

## Licenza

Il codice sorgente dell’SDK OpenMed è distribuito con [Apache-2.0 License](LICENSE).

## Citazione

Se OpenMed ti è utile nella tua ricerca, ti preghiamo di citarlo:

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

## Cronologia delle stelle

Se OpenMed ti è utile, una stella aiuta altri a scoprirlo.

[4,700+ GitHub stars · 29 Jul 2026 snapshot](https://github.com/maziyarpanahi/openmed/stargazers)

---

<div align="center">

Realizzato dal team OpenMed

<a href="https://openmed.life">Sito web</a> ·
<a href="https://openmed.life/docs">Documentazione</a> ·
<a href="https://x.com/openmed_ai">X / Twitter</a> ·
<a href="https://www.linkedin.com/company/openmed-ai/">LinkedIn</a>

</div>
