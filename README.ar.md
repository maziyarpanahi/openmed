<div align="center">

<img src="docs/brand/openmed-readme-banner.png" alt="لافتة README من OpenMed مع تميمة القط والشعار النصي بأحرف صغيرة وعلامة Open Cross، وتعرض النص ذكاء اصطناعي مفتوح المصدر للرعاية الصحية، وأكثر من 340 مليون تنزيل، وأكثر من 10 ملايين تثبيت" width="1280" />

<h3>بياناتك. نموذجك. عتادك.</h3>

<p><b>حوِّل النص السريري إلى رؤى منظَّمة ومجرَّدة من الهوية على عتاد تتحكم به.</b><br/>
ينفّذ وقت التشغيل المحلي الأساسي في OpenMed الاستخراج وإزالة الهوية بعد توفر عناصر النماذج المطلوبة. وقد تستخدم تنزيلات النماذج وموائمات المزوّدات البعيدة والمسارات المفعّل فيها القياس عن بُعد والتكاملات التي يهيئها المستخدم الشبكة؛ راجع شروط كل نموذج ومجموعة بيانات.</p>

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
  <b>تشغيل محلي أولاً</b> &nbsp;·&nbsp; <b>34 لغة PII مدعومة بالنماذج</b> &nbsp;·&nbsp; <b>Apache-2.0 SDK</b>
</p>

<p>
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.es.md">Español</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.de.md">Deutsch</a> ·
  <a href="README.it.md">Italiano</a> ·
  <a href="README.pt.md">Português</a> ·
  <a href="README.nl.md">Nederlands</a> ·
  <b>العربية</b> ·
  <a href="README.hi.md">हिन्दी</a> ·
  <a href="README.te.md">తెలుగు</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.tr.md">Türkçe</a> ·
  <a href="README.fa.md">فارسی</a>
</p>

</div>

---

<div dir="rtl">

## شاهده أثناء العمل

</div>

<div align="center">
  <img src="docs/brand/openmed-pii-demo.gif" alt="OpenMed يزيل معرّفات الهوية من تقرير خروج سريري في الوقت الفعلي" width="760" />
  <br/>
  <sub><b>إزالة معرّفات الهوية في الوقت الفعلي</b>: يقوم Nemotron Privacy Filter بإخفاء الأسماء والعناوين والمعرّفات وبيانات الفوترة من تقرير خروج سريري، بالكامل على الجهاز. <i>(جميع القيم المعروضة اصطناعية.)</i></sub>
</div>

---

<div dir="rtl">

## مثال في 30 ثانية

</div>

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

<div dir="rtl">

نموذج NER سريري يستخدم وقت التشغيل المحلي بعد توفر العناصر المطلوبة.

</div>

---

<div dir="rtl">

## لماذا OpenMed؟

| اعتبار النشر | حدود OpenMed SDK |
| --- | --- |
| وقت التشغيل الأساسي | يعالج محليًا بعد توفر العناصر المطلوبة |
| مسارات الشبكة الاختيارية | قد تستخدم التنزيلات والموائمات البعيدة ومسارات القياس والتكاملات الشبكة |
| التحقق | يتحقق مالك النشر من شروط النماذج والبيانات وسلوك الخصوصية والملاءمة السريرية |
| الواجهات | Python وSwift وAndroid والمتصفح والخدمات حيث تكون مدعومة |

- **فهرس نماذج منسّق**: تحقّق من كل نموذج وترخيص ومجموعة بيانات لحالة الاستخدام.
- **تهيئة متوافقة مع فئات Safe Harbor**: يمكنها استهداف فئات المعرّفات الثماني عشرة؛ تبقى مراجعة خبير للنشر مطلوبة، واستخدام SDK لا يثبت بحد ذاته الامتثال لـ HIPAA.
- **مسارات تنفيذ مدعومة**: تختلف محوّلات CPU وCUDA وMLX والهاتف والخدمة والمتصفح حسب البيئة والعنصر.
- **واجهات النشر**: تتطلب Python والحاويات والخدمات ومسارات الدُفعات تهيئةً وتحققًا.
- **مصدر SDK**: منشور بموجب Apache-2.0 License؛ وتختلف شروط النماذج ومجموعات البيانات.

---

<div dir="rtl">

## على الجهاز، على Apple: Swift وMLX وiOS

على عتاد Apple المدعوم، يمكن لـ OpenMed استخدام **MLX** و**[OpenMedKit](swift/OpenMedKit)** للمعالجة المحلية بعد توفر العناصر المطلوبة. ويظل الحصول على النماذج وأي تكاملات بعيدة يهيئها المستخدم حدودًا شبكية منفصلة.

```swift
// Add OpenMedKit to your app
dependencies: [
    .package(url: "https://github.com/maziyarpanahi/openmed.git", from: "2.0.0"),
]
```

<div dir="rtl">

- **زمن تشغيل MLX** لتصنيف رموز PII، وعائلة Privacy Filter، ومهام zero-shot التجريبية لعائلة GLiNER، مع مسار احتياطي عبر CoreML.
- **اسم نموذج واحد، كل المنصات**: على العتاد غير التابع لـ Apple، تعود أسماء نماذج MLX تلقائيًا إلى نقطة تحقق PyTorch المقابلة.
- **Python على Apple Silicon** أيضًا: `pip install --upgrade "openmed[mlx]"`.

الأدلة: [خلفية MLX](docs/mlx-backend.md) · [OpenMedKit (Swift)](docs/swift-openmedkit.md) · [تصدير CoreML](docs/coreml-export.md)

</div>

---

<div dir="rtl">

## كيف يعمل

</div>

```mermaid
flowchart LR
    A["نص سريري"] --> B["OpenMed<br/>(محلي أولاً)"]
    B --> C["كيانات طبية"]
    B --> D["تم كشف PII"]
    B --> E["نص مُزال الهوية"]
    style B fill:#0D6E6E,stroke:#0A5656,stroke-width:2px,color:#ffffff
    style C fill:#D6EBEB,stroke:#0D6E6E,color:#0E1116
    style D fill:#F7DCD8,stroke:#C5453A,color:#0E1116
    style E fill:#F5E27A,stroke:#A9A088,color:#0E1116
```

---

<div dir="rtl">

## البدء السريع

</div>

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

**واجهة Python**

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

**خدمة REST**

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

**دفعي**

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

<div dir="rtl">

**دون اتصال / معزول؟** وجِّه `model_name` (أو `model_id`) إلى دليل محلي، وسيحمّله OpenMed محليًا دون الاتصال بـ Hugging Face Hub:

</div>

```python
from openmed import OpenMedConfig, analyze_text

result = analyze_text(
    "Patient presents with chronic myeloid leukemia and Type 2 diabetes.",
    model_id="./models/OpenMed-NER-DiseaseDetect-SuperClinical-434M",
    config=OpenMedConfig(device="cpu"),
)
```

---

<div dir="rtl">

## النماذج

سجل منتقى من نماذج NER الطبية المتخصصة، تصفّح [الكتالوج الكامل](https://openmed.life/docs/model-registry).

</div>

| النموذج | التخصص | أنواع الكيانات | الحجم |
|---------|--------|----------------|-------|
| `disease_detection_superclinical` | الأمراض والحالات | DISEASE, CONDITION, DIAGNOSIS | 434M |
| `pharma_detection_superclinical`  | الأدوية والعلاجات | DRUG, MEDICATION, TREATMENT   | 434M |
| `pii_superclinical_large`     | PII وإزالة الهوية | NAME, DATE, SSN, PHONE, EMAIL, ADDRESS | 434M |
| `anatomy_detection_electramed`    | التشريح وأجزاء الجسم | ANATOMY, ORGAN, BODY_PART     | 109M |
| `gene_detection_genecorpus`       | الجينات والبروتينات | GENE, PROTEIN                 | 109M |

---

<div dir="rtl">

## الخصوصية: كشف PII وإزالة الهوية

</div>

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

<div dir="rtl">

- **الدمج الذكي للكيانات** يبقي `01/15/1970` كاملاً بدلاً من تجزئته.
- **تمويه قائم على Faker** مع موفّرين مخصّصين لمعرّفات سريرية (CPF، CNPJ، BSN، NIR، Codice Fiscale، NIE، Aadhaar، Steuer-ID، NPI).
- **حدود HIPAA**: فئات متوافقة مع Safe Harbor وعتبات قابلة للتهيئة هي أدوات مساعدة للتنفيذ؛ تبقى مراجعة خبير للنشر مطلوبة ولا يثبت استخدام SDK وحده الامتثال.

[دفتر PII الكامل](examples/notebooks/PII_Detection_Complete_Guide.ipynb) · [الدمج الذكي](docs/pii-smart-merging.md) · [إخفاء الهوية](docs/anonymization.md)

</div>

<details>
<summary><b>عائلة Privacy Filter</b>: ثلاث عائلات نماذج على معمارية OpenAI Privacy Filter</summary>

<br/>

<div dir="rtl">

شِفرة النموذج واحدة (محوّل Sparse-MoE بأسلوب gpt-oss مع انتباه محلي، ورموز sink، وRoPE+YaRN، وتجزئة tiktoken `o200k_base`)؛ تختلف بيانات التدريب فقط. تمرّ جميعها عبر **نفس** واجهة `extract_pii()` / `deidentify()`، يتغيّر فقط الوسيط `model_name=`.

</div>

| المتغيّر | PyTorch (CPU + CUDA) | MLX (Apple Silicon) | MLX 8-bit |
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

<div dir="rtl">

على المضيفات غير العاملة بـ Apple Silicon، تُستبدل أسماء نماذج MLX تلقائيًا بنقطة تحقق PyTorch المقابلة (مع تحذير لمرة واحدة): اكتب اسم نموذج واحد، وشغّله في أي مكان. راجع [معمارية Privacy Filter وتوجيه الخلفية](docs/anonymization.md#privacy-filter-family).

</div>

</details>

---

<div dir="rtl">

## PII متعدد اللغات (35 مسارًا مدعومًا؛ 34 مدعومًا بالنماذج)

الاستخراج وإزالة الهوية في `en`، `fr`، `de`، `it`، `es`، `nl`، `hi`، `te`، `pt`، `ar`، `ja` و`tr`، **فهرس نماذج PII المسجّل** إجمالاً.

</div>

```bash
python -c "from openmed import extract_pii; print([(e.label, e.text) for e in extract_pii('Dr. Pedro Almeida, CPF: 123.456.789-09, email: pedro@hospital.pt', lang='pt').entities])"
```

<details>
<summary>عرض أمثلة لكل لغة (البرتغالية، الهولندية، الهندية، العربية، اليابانية، التركية)</summary>

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

<div dir="rtl">

## REST API

خدمة FastAPI صديقة لـ Docker، مع التحقق من الطلبات، والتحميل المسبق لخط أنابيب مشترك، وأغلفة أخطاء موحّدة.

</div>

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

<div dir="rtl">

راجع [دليل خدمة REST الكامل](docs/rest-service.md).

</div>

---

<div dir="rtl">

## التوثيق

الأدلة الكاملة على **[openmed.life/docs](https://openmed.life/docs/)**.

</div>

| | | |
|---|---|---|
| [البدء](https://openmed.life/docs/) | [تحليل النص](https://openmed.life/docs/analyze-text) | [سجل النماذج](https://openmed.life/docs/model-registry) |
| [دليل كشف PII](examples/notebooks/PII_Detection_Complete_Guide.ipynb) | [إخفاء الهوية](docs/anonymization.md) | [المعالجة الدفعية](https://openmed.life/docs/batch-processing) |
| [ملفات الإعداد](https://openmed.life/docs/profiles) | [خدمة REST](docs/rest-service.md) | [خلفية MLX](docs/mlx-backend.md) |

---

<div dir="rtl">

## تعرّف على التميمة

<img src="docs/brand/openmed-mascot-icon.png" alt="تميمة OpenMed" width="104" align="right" />

حارس OpenMed قطة فارسية كثيفة الفرو في هيئة **ابن سينا (Avicenna)** الصغير، الطبيب الفارسي العظيم الذي ظل
كتابه «القانون في الطب» المرجع الطبي القياسي في العالم نحو 600 عام. يحرس كتاب المعرفة الطبية المفتوح، بلوحة
ألوان مستوحاة من **الفيروز الفارسي (fīrūza)**: حارس محلي أولاً لأكثر بياناتك خصوصية.

<br clear="right"/>

</div>

---

<div dir="rtl">

## المساهمة

المساهمات مُرحَّب بها: تقارير الأخطاء وطلبات الميزات وطلبات السحب (PR).

- [افتح مشكلة (issue)](https://github.com/maziyarpanahi/openmed/issues)
- **الترجمات مُرحَّب بها**: ساعد في إكمال ملفات README بلغات أخرى المرتبطة في مبدّل اللغات بالأعلى.

## شكر وتقدير

يعتمد OpenMed على أعمال مفتوحة المصدر ممتازة: شكر خاص لـ **OpenAI** (معمارية [Privacy Filter](https://huggingface.co/openai/privacy-filter))، و**NVIDIA** (مجموعة بيانات [Nemotron PII](https://huggingface.co/datasets/nvidia/Nemotron-PII-v1))، و**Hugging Face** (`transformers` ومنظومة النماذج)، و**Apple** ([MLX](https://github.com/ml-explore/mlx))، ومشرفي **[Faker](https://faker.readthedocs.io/)**.

## الترخيص

يُنشر الكود المصدري لحزمة OpenMed SDK بموجب [Apache-2.0 License](LICENSE).

## الاستشهاد

إذا كان OpenMed مفيدًا في بحثك، فيرجى الاستشهاد به:

</div>

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

<div dir="rtl">

## سجل النجوم

إذا كان OpenMed مفيدًا لك، فإن نجمة تساعد الآخرين على اكتشافه.

</div>

[4,700+ GitHub stars · 29 Jul 2026 snapshot](https://github.com/maziyarpanahi/openmed/stargazers)

---

<div align="center">

صُنع بواسطة فريق OpenMed

<a href="https://openmed.life">الموقع</a> ·
<a href="https://openmed.life/docs">التوثيق</a> ·
<a href="https://x.com/openmed_ai">X / تويتر</a> ·
<a href="https://www.linkedin.com/company/openmed-ai/">LinkedIn</a>

</div>
