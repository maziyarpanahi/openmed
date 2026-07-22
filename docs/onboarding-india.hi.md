# भारत के उपयोगकर्ताओं के लिए OpenMed ऑनबोर्डिंग

[English version](onboarding-india.md)

यह गाइड भारत के लिए लोकल-फर्स्ट डी-आइडेंटिफिकेशन सेटअप दिखाती है: आज उपलब्ध
OpenMed पॉलिसी से शुरू करें, सिंथेटिक Aadhaar और ABHA फ़ॉर्मैट पहचानकर्ताओं को
डी-आइडेंटिफ़ाई करें, सिंथेटिक कोड-मिश्रित Hinglish क्लिनिकल नोट प्रोसेस करें और
CPU-only कम-RAM मशीन के लिए छोटा हिन्दी चेकपॉइंट चुनें।

पूरे सिंथेटिक Hindi और code-mixed Hinglish walkthrough के लिए
[end-to-end उदाहरण](https://github.com/maziyarpanahi/openmed/blob/master/examples/deid_hindi_hinglish_note.py)
चलाएँ या
[Chinese और Hindi Notebook tour](https://github.com/maziyarpanahi/openmed/blob/master/examples/notebooks/Chinese_Hindi_Deid_Tour.ipynb)
देखें।

!!! important "केवल सिंथेटिक वॉकथ्रू"
    इस पेज और `examples/onboarding_india_dpdp.py` में दिए गए सभी नाम और
    पहचानकर्ता बनाए गए टेस्ट डेटा हैं। ये केवल फ़ॉर्मैट उदाहरण हैं; इन्हें किसी
    वास्तविक व्यक्ति के लिए जारी या सत्यापित नहीं किया गया है। ट्यूटोरियल में
    कभी भी वास्तविक व्यक्तिगत या स्वास्थ्य डेटा पेस्ट न करें।

## लोकल रनटाइम इंस्टॉल करें

वर्चुअल एनवायरनमेंट बनाएँ और Hugging Face मॉडल डिपेंडेंसी के साथ OpenMed
इंस्टॉल करें:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install "openmed[hf]"
```

मॉडल को पहली बार डाउनलोड करने के लिए नेटवर्क चाहिए। चेकपॉइंट लोकल Hugging Face
कैश में आने के बाद OpenMed इनफ़रेंस उसी मशीन पर चलता है और क्लिनिकल नोट को किसी
होस्टेड API पर नहीं भेजता।

## DPDP-उन्मुख पॉलिसी क्विकस्टार्ट

Digital Personal Data Protection (DPDP) ढाँचा टेक्स्ट मास्किंग से कहीं व्यापक
है। उद्देश्य और वैध प्रोसेसिंग, जहाँ लागू हो वहाँ नोटिस और सहमति, डेटा प्रतिधारण,
ऐक्सेस कंट्रोल, घटना प्रतिक्रिया, अनुबंध और मानवीय प्रशासन संगठन की जिम्मेदारियाँ
हैं। आधिकारिक
[DPDP Act, 2023](https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf)
और [DPDP Rules, 2025](https://www.meity.gov.in/documents/act-and-policies/digital-personal-data-protection-rules-2025-gDOxUjMtQWa?pageTitle=Digital-Personal-Data-Protection-Rules-2025686cadad39.pdf)
देखें और उचित कानूनी सलाह लें। सॉफ़्टवेयर पॉलिसी प्रोफ़ाइल केवल एक तकनीकी
कंट्रोल है, DPDP अनुपालन का दावा नहीं।

OpenMed में assist-only `india_dpdp_act` प्रोफ़ाइल उपलब्ध है। यह direct
identifiers को replace, quasi-identifiers को mask और clinical concepts को keep
करती है। यह safety sweep अनिवार्य रखती है और reversible IDs व retained mappings
दोनों बंद करती है:

```python
from openmed.core.policy import list_policies, load_policy

policy_name = "india_dpdp_act"
assert policy_name in list_policies()

policy = load_policy(policy_name)
assert policy.default_action == "replace"
assert policy.action_for("PERSON") == "replace"
assert policy.action_for("ID_NUM") == "replace"
assert policy.safety_sweep_mandatory
assert policy.keep_mapping is False
assert policy.reversible_id is False

print(policy.name, policy.default_action)
```

यह कोड उपलब्ध `openmed/core/policies/india_dpdp_act.json` प्रोफ़ाइल लोड करता
है। इसकी metadata आधिकारिक sources दर्ज करती है और साफ़ करती है कि यह profile
केवल सहायक है, कानूनी सलाह या autonomous compliance determination नहीं।

## सिंथेटिक Aadhaar और ABHA पहचानकर्ता डी-आइडेंटिफ़ाई करें

[UIDAI के अनुसार Aadhaar](https://www.uidai.gov.in/en/my-aadhaar/about-your-aadhaar.html)
12 अंकों का नंबर है, जबकि
[Ayushman Bharat Digital Mission के अनुसार ABHA](https://abdm.gov.in/FAQ)
14 अंकों का नंबर है। उदाहरण में जानबूझकर बनाए गए ये डिस्प्ले-फ़ॉर्म मान हैं:

- सिंथेटिक Aadhaar-फ़ॉर्मैट मान: `2467 7832 5484` (यह OpenMed के Verhoeff
  checksum validator को भी पास करता है)
- सिंथेटिक ABHA-फ़ॉर्मैट मान: `91-0000-0000-0000`

`india_dpdp_act` profile OpenMed के ABDM recognizer को अपने-आप चालू करती है। वह
इन Aadhaar और ABHA display forms को validate करके canonical `ID_NUM` label में
normalize करता है। नीचे का छोटा custom recognizer केवल इस offline tutorial के
बनाए गए व्यक्ति के नाम को deterministic बनाता है। इसके बाद policy replacement
action तय करती है:

```python
from openmed import deidentify

synthetic_note = (
    "Synthetic Hinglish note: Patient Asha Verma ka follow-up aaj hai. "
    "Aadhaar 2467 7832 5484 aur ABHA 91-0000-0000-0000 record mein hain. "
    "Patient ko do din se halka bukhar hai."
)

india_recognizer = {
    "case_sensitive": False,
    "deny": {
        "terms": [{"term": "Asha Verma", "label": "PERSON"}],
    },
}

result = deidentify(
    synthetic_note,
    method="replace",
    model_name="OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1",
    lang="hi",
    locale="en_IN",
    policy="india_dpdp_act",
    use_safety_sweep=True,
    custom_recognizer=india_recognizer,
    consistent=True,
    seed=707,
)

assert "2467 7832 5484" not in result.deidentified_text
assert "91-0000-0000-0000" not in result.deidentified_text
assert "Asha Verma" not in result.deidentified_text
print(result.deidentified_text)
```

Hindi language hint हिन्दी-अवेयर normalization और patterns चुनता है। छोटा
Hindi model कोड-मिश्रित नोट संभालता है और साफ़ स्थानीय rules सिंथेटिक नाम व
पहचानकर्ता फ़ॉर्मैट के लिए ट्यूटोरियल को deterministic बनाते हैं। वास्तविक
deployment में documented data contracts से custom rules बनाएँ, प्रतिनिधि
सिंथेटिक fixtures पर direct-identifier recall जाँचें और release से पहले हर
residual leak की समीक्षा करें।

Source checkout से पूरा उदाहरण चलाएँ:

```bash
python examples/onboarding_india_dpdp.py
```

## कम-RAM, CPU-only सेटअप

OpenMed model manifest में मौजूद छोटे Hindi PII checkpoints में से शुरुआत करें।
Raw FP32 weights के लिए प्रति parameter लगभग चार bytes लगते हैं; Python
runtime, tokenizer, temporary tensors और input length अतिरिक्त memory लेते हैं।

| Checkpoint | Parameters | अनुमानित raw FP32 weights | उपयोग |
| --- | ---: | ---: | --- |
| `OpenMed/OpenMed-PII-Hindi-ClinicalE5-Small-33M-v1` | 33M | 132 MB | सबसे कम weight footprint; केवल Latin-script evaluation |
| `OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1` | 44M | 176 MB | छोटा default; Devanagari समर्थित |
| `OpenMed/OpenMed-PII-Hindi-LiteClinical-Small-66M-v1` | 66M | 264 MB | केवल Latin-script evaluation |

ये छोटे checkpoints OpenMed की **Tiny** device tier में आते हैं, जिसका release
target अधिकतम 350 MB resident RAM है। मौजूदा tokenizer-script coverage में केवल
44M checkpoint Devanagari समर्थित है। नया passing script audit न हो तो 33M और
66M checkpoints को Devanagari notes के लिए न चुनें। Memory target को हर Python
environment की guarantee न मानें; अपनी deployment machine और वास्तविक input
lengths पर peak RSS मापें। [Device Tiers and SLOs](tiers.md) और
[Tokenizer Script Coverage](model-tokenizer-script-coverage.md) देखें।

कम संसाधन वाले clinic workstation या startup VM पर:

1. Notes में Devanagari हो सकता है तो 44M checkpoint, CPU execution और
   `batch_size=1` से शुरू करें। 33M checkpoint केवल tested Latin-script-only
   Hinglish corpus के लिए विचार करें।
2. लंबे notes को batch करने से पहले sentence boundaries पर विभाजित करें। पास के
   sentences का क्रम बनाए रखें और structured identifier को chunks के बीच न काटें।
3. प्रतिनिधि notes के साथ पर्याप्त peak-memory headroom मिलने पर ही batch 2 और
   फिर 4 आज़माएँ। बड़ा batch throughput बढ़ाता है, पर temporary tensor memory भी
   बढ़ाता है।
4. हर sentence के लिए नया loader बनाने के बजाय एक loaded model या एक
   `BatchProcessor` reuse करें। कम-RAM host पर कई model-worker processes न चलाएँ।
5. छोटा model चुनते समय `india_dpdp_act`, safety sweep और स्थानीय identifier
   rules चालू रखें। छोटे checkpoint को स्वीकार करने से पहले leakage tests फिर
   चलाएँ।

कई छोटे notes के लिए `BatchProcessor(operation="deidentify")` document
`batch_size` control देता है और underlying loader reuse करता है। API के लिए
[Batch Processing](batch-processing.md) देखें।

## Production checklist

- Inference और cached artifacts को approved device या network boundary के भीतर
  रखें; OpenMed de-identification path में telemetry नहीं जोड़ता।
- Logs या audit exports में `result.original_text`, raw entity values या
  reversible mappings न रखें।
- स्थानीय पहचानकर्ता patterns और synthetic regression fixtures जोड़ें; यह
  walkthrough भारत के सभी identifiers की सूची नहीं है।
- Hindi, Latin-script Hinglish और code-mixed notes को अलग-अलग validate करें।
- DPDP governance और ABDM participation requirements को चुनी हुई OpenMed policy
  profile से स्वतंत्र मानें।
