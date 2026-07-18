# त्वरित शुरुआत

यह गाइड आपको कुछ ही मिनटों में खाली वर्कस्टेशन से दस्तावेज़ के परिणाम चलाने और कॉपी करने तक ले जाता है। यह डिपेंडेंसी प्रबंधन के लिए [uv](https://github.com/astral-sh/uv) का उपयोग करता है, लेकिन कोई भी Python 3.11+ वातावरण काम करेगा।

## 1. वातावरण तैयार करें

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh  # uv इंस्टॉल करें (पहले से हो तो छोड़ें)
uv venv --python 3.11                           # समर्पित वर्चुअल वातावरण बनाएँ
source .venv/bin/activate                       # या सीधे `uv python` का उपयोग करें

# Hugging Face extras और दस्तावेज़ टूलिंग के साथ OpenMed इंस्टॉल करें
uv pip install ".[hf]"
```

zero-shot GLiNER स्टैक या डेवलपमेंट टूल चाहिए? आवश्यकतानुसार extras जोड़ें:

```bash
uv pip install ".[hf,gliner]"      # GLiNER और transformers जोड़ें
uv pip install ".[dev]"            # pytest, कवरेज और linting
```

स्कैन की गई इमेज और दस्तावेज़ OCR के लिए, मल्टीमॉडल extra तथा सिस्टम Tesseract बाइनरी इंस्टॉल करें:

```bash
uv pip install ".[multimodal]"
brew install tesseract             # macOS
sudo apt-get install tesseract-ocr # Debian/Ubuntu
```

PaddleOCR एक अधिक भारी, वैकल्पिक OCR बैकएंड के रूप में उपलब्ध है:

```bash
uv pip install ".[ocr-paddle]"
```

CDA/C-CDA XML डी-आइडेंटिफिकेशन मुख्य इंस्टॉलेशन में उपलब्ध है। यह संरचित हेडर PHI को रिडैक्ट करता है, CDA अनुभागों के वर्णनात्मक टेक्स्ट की जाँच करता है, XML को पार्स करने योग्य रखता है और केवल उन्हीं `.xml` फ़ाइलों को संसाधित करता है जो CDA दस्तावेज़ जैसी दिखती हैं:

```python
from openmed.interop.cda import redact_cda

redacted_xml = redact_cda("synthetic_ccda.xml", date_shift_days=30)
```

Apple Silicon Mac पर आप नए MLX पथ से सीधे शुरू कर सकते हैं:

```bash
uv pip install ".[mlx]"            # Python MLX रनटाइम तथा tokenizer/artifact डिपेंडेंसी
uv run python -c "from openmed.core.backends import get_backend; print(type(get_backend()).__name__)"
```

यदि एक ही मशीन पर पूरी लॉन्च सतह चाहिए, तो extras को मिलाएँ:

```bash
uv pip install ".[hf,mlx,docs]"
```

## 2. `analyze_text` चलाएँ

```python
from openmed import analyze_text

text = "Metastatic breast cancer treated with paclitaxel and trastuzumab."

resp = analyze_text(text, model_name="disease_detection_superclinical")
print(resp.entities[0])

# एम्बेड करने योग्य HTML चाहिए तो "html" आउटपुट फ़ॉर्मैट चुनें
html = analyze_text(text, model_name="disease_detection_superclinical", output_format="html")
print(html)  # डैशबोर्ड या दस्तावेज़ के लिए तैयार
```

त्वरित स्क्रिप्ट एंट्रीपॉइंट पसंद है? एक-फ़ाइल smoke script चलाएँ:

```bash
uv run python examples/pii_model_comparison.py
```

## 3. PII का डी-आइडेंटिफिकेशन करें

```python
from openmed import deidentify

result = deidentify("Patient John Doe, DOB 01/15/1970", method="mask")
print(result.deidentified_text)
# Patient [first_name] [last_name], DOB [date]
```

`deidentify()` पाँच विधियों (`mask`, `remove`, `replace`, `hash`, `shift_dates`) का समर्थन करता है। प्रत्येक का चलने योग्य उदाहरण और `reidentify()` से परिणाम वापस पाने का तरीका [अनामिकरण त्वरित शुरुआत](anonymization.md#quickstart-choosing-a-method) में देखें।

## 4. दस्तावेज़ से कोड स्निपेट कॉपी करें

सभी कोड ब्लॉक में Material for MkDocs के कॉपी बटन उपलब्ध हैं। कमांड पैलेट (`/` या `cmd/ctrl + K`) खोलकर “GLiNER,” “OpenMedConfig,” या “token classification” खोजें और पूर्वावलोकन में दिखा स्निपेट कॉपी करें। यदि आप AI कोडिंग सहायक का उपयोग करते हैं, तो उसे प्रकाशित दस्तावेज़ URL दें, ताकि वह इसी संरचित Markdown को पढ़े और प्रामाणिक उत्तर दिखाए।

## 5. वैकल्पिक: कॉन्फ़िगरेशन पिन करें

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

संपूर्ण YAML/ENV स्कीमा, PHI-सचेत सत्यापन सहायक और लॉगिंग सेटअप के लिए **कॉन्फ़िगरेशन** अनुभाग पर जाएँ।
