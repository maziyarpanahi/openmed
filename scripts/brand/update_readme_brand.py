#!/usr/bin/env python3
"""Synchronize brand art and governed claim wording across all root READMEs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIMS_PATH = REPO_ROOT / "docs/brand/system/claims.yml"
README_FILES = (
    "README.md",
    "README.ar.md",
    "README.de.md",
    "README.es.md",
    "README.fa.md",
    "README.fr.md",
    "README.hi.md",
    "README.it.md",
    "README.ja.md",
    "README.nl.md",
    "README.pt.md",
    "README.sw.md",
    "README.te.md",
    "README.tr.md",
    "README.zh-CN.md",
)

ALT_TEXT = {
    "README.md": (
        "OpenMed README banner with the cat mascot, lowercase wordmark, "
        "Open Cross, and the text Open-source healthcare AI, 340M+ "
        "downloads, and 10M+ installs"
    ),
    "README.ar.md": (
        "لافتة README من OpenMed مع تميمة القط والشعار النصي بأحرف صغيرة "
        "وعلامة Open Cross، وتعرض النص ذكاء اصطناعي مفتوح المصدر للرعاية "
        "الصحية، وأكثر من 340 مليون تنزيل، وأكثر من 10 ملايين تثبيت"
    ),
    "README.de.md": (
        "OpenMed-README-Banner mit Katzenmaskottchen, kleingeschriebener "
        "Wortmarke, Open Cross und dem Text Open-Source-KI für das "
        "Gesundheitswesen, 340 Mio.+ Downloads und 10 Mio.+ Installationen"
    ),
    "README.es.md": (
        "Banner README de OpenMed con la mascota del gato, la marca "
        "denominativa en minúsculas, Open Cross y el texto IA sanitaria "
        "de código abierto, 340 M+ de descargas y 10 M+ de instalaciones"
    ),
    "README.fa.md": (
        "بنر README اپن‌مد با نماد گربه، نوشتار نشان با حروف کوچک، "
        "Open Cross و متن هوش مصنوعی متن‌باز سلامت، بیش از ۳۴۰ میلیون "
        "دانلود و بیش از ۱۰ میلیون نصب"
    ),
    "README.fr.md": (
        "Bannière README d’OpenMed avec la mascotte chat, le nom en "
        "minuscules, l’Open Cross et le texte IA de santé open source, "
        "plus de 340 M de téléchargements et plus de 10 M d’installations"
    ),
    "README.hi.md": (
        "बिल्ली शुभंकर, छोटे अक्षरों वाले वर्डमार्क, Open Cross और "
        "ओपन-सोर्स हेल्थकेयर AI, 340M+ डाउनलोड और 10M+ इंस्टॉल टेक्स्ट "
        "वाला OpenMed README बैनर"
    ),
    "README.it.md": (
        "Banner README di OpenMed con la mascotte gatto, il marchio in "
        "minuscolo, Open Cross e il testo IA sanitaria open source, oltre "
        "340 milioni di download e oltre 10 milioni di installazioni"
    ),
    "README.ja.md": (
        "猫のマスコット、小文字のワードマーク、Open Cross、"
        "オープンソースのヘルスケア AI・3億4,000万回以上のダウンロード・"
        "1,000万回以上のインストールという文字を配した OpenMed README バナー"
    ),
    "README.nl.md": (
        "OpenMed README-banner met de kattenmascotte, het woordmerk in "
        "kleine letters, Open Cross en de tekst Open-source AI voor de "
        "gezondheidszorg, 340M+ downloads en 10M+ installaties"
    ),
    "README.pt.md": (
        "Banner README do OpenMed com o mascote gato, a marca nominativa "
        "em minúsculas, Open Cross e o texto IA de saúde de código aberto, "
        "340 M+ de downloads e 10 M+ de instalações"
    ),
    "README.sw.md": (
        "Bango la README la OpenMed lenye kinyago cha paka, nembo ya "
        "maandishi yenye herufi ndogo, Open Cross, na maandishi AI ya "
        "huduma za afya ya chanzo huria, vipakuliwa milioni 340+ na "
        "usakinishaji milioni 10+"
    ),
    "README.te.md": (
        "పిల్లి మస్కట్, చిన్న అక్షరాల వర్డ్‌మార్క్, Open Cross, "
        "ఓపెన్ సోర్స్ హెల్త్‌కేర్ AI, 340M+ డౌన్‌లోడ్‌లు మరియు 10M+ "
        "ఇన్‌స్టాల్‌లు అనే వచనంతో OpenMed README బ్యానర్"
    ),
    "README.tr.md": (
        "Kedi maskotu, küçük harfli sözcük markası, Open Cross ve açık "
        "kaynak sağlık hizmetleri yapay zekâsı, 340M+ indirme ve 10M+ "
        "kurulum metni bulunan OpenMed README afişi"
    ),
    "README.zh-CN.md": (
        "OpenMed README 横幅，包含猫咪吉祥物、小写文字标识、Open Cross，"
        "以及开源医疗 AI、下载量 3.4 亿+ 和安装量 1000 万+ 文字"
    ),
}

RESOURCE_LINKS = """<p>
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
</p>"""

HERO_CLAIMS = {
    "README.md": (
        "  <b>Local-first runtime</b> &nbsp;·&nbsp; "
        "<b>{model_backed} model-backed PII languages</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.ar.md": (
        "  <b>تشغيل محلي أولاً</b> &nbsp;·&nbsp; "
        "<b>{model_backed} لغة PII مدعومة بالنماذج</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.de.md": (
        "  <b>Local-First-Laufzeit</b> &nbsp;·&nbsp; "
        "<b>{model_backed} modellgestützte PII-Sprachen</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.es.md": (
        "  <b>Ejecución local primero</b> &nbsp;·&nbsp; "
        "<b>{model_backed} idiomas PII respaldados por modelos</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.fa.md": (
        "  <b>اجرای محلی‌محور</b> &nbsp;·&nbsp; "
        "<b>{model_backed} زبان PII با پشتیبانی مدل</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.fr.md": (
        "  <b>Exécution locale en priorité</b> &nbsp;·&nbsp; "
        "<b>{model_backed} langues PII prises en charge par modèle</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.hi.md": (
        "  <b>स्थानीय-प्रथम रनटाइम</b> &nbsp;·&nbsp; "
        "<b>{model_backed} मॉडल-समर्थित PII भाषाएँ</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.it.md": (
        "  <b>Esecuzione locale prioritaria</b> &nbsp;·&nbsp; "
        "<b>{model_backed} lingue PII supportate da modelli</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.ja.md": (
        "  <b>ローカルファーストのランタイム</b> &nbsp;·&nbsp; "
        "<b>モデル対応 PII 言語 {model_backed} 種</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.nl.md": (
        "  <b>Lokale uitvoering voorop</b> &nbsp;·&nbsp; "
        "<b>{model_backed} modelondersteunde PII-talen</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.pt.md": (
        "  <b>Execução local em primeiro lugar</b> &nbsp;·&nbsp; "
        "<b>{model_backed} idiomas PII com suporte de modelos</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.sw.md": (
        "  <b>Uendeshaji unaotanguliza matumizi ya ndani</b> &nbsp;·&nbsp; "
        "<b>Lugha {model_backed} za PII zinazotumia modeli</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.te.md": (
        "  <b>స్థానిక అమలుకు ప్రాధాన్యం</b> &nbsp;·&nbsp; "
        "<b>మోడల్ మద్దతు గల {model_backed} PII భాషలు</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.tr.md": (
        "  <b>Yerel öncelikli çalışma</b> &nbsp;·&nbsp; "
        "<b>Model destekli {model_backed} PII dili</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
    "README.zh-CN.md": (
        "  <b>本地优先运行</b> &nbsp;·&nbsp; "
        "<b>{model_backed} 种模型支持的 PII 语言</b> &nbsp;·&nbsp; "
        "<b>Apache-2.0 SDK</b>"
    ),
}

INTRO_COPY = {
    "README.md": (
        "<p><b>Turn clinical text into structured, de-identified insight on "
        "hardware you control.</b><br/>\n"
        "OpenMed's core local runtime performs extraction and de-identification "
        "after required model artifacts are available. Model downloads, "
        "remote-provider adapters, telemetry-enabled paths, and user-configured "
        "integrations may use a network; review each model and dataset's "
        "terms.</p>"
    ),
    "README.ar.md": (
        "<p><b>حوِّل النص السريري إلى رؤى منظَّمة ومجرَّدة من الهوية على عتاد "
        "تتحكم به.</b><br/>\n"
        "ينفّذ وقت التشغيل المحلي الأساسي في OpenMed الاستخراج وإزالة الهوية "
        "بعد توفر عناصر النماذج المطلوبة. وقد تستخدم تنزيلات النماذج وموائمات "
        "المزوّدات البعيدة والمسارات المفعّل فيها القياس عن بُعد والتكاملات "
        "التي يهيئها المستخدم الشبكة؛ راجع شروط كل نموذج ومجموعة بيانات.</p>"
    ),
    "README.de.md": (
        "<p><b>Klinische Texte auf kontrollierter Hardware in strukturierte, "
        "de-identifizierte Erkenntnisse umwandeln.</b><br/>\n"
        "Die lokale Kernlaufzeit von OpenMed führt Extraktion und "
        "De-Identifikation aus, sobald die benötigten Modellartefakte "
        "vorliegen. Modelldownloads, Adapter für entfernte Anbieter, "
        "telemetrieaktivierte Pfade und nutzerkonfigurierte Integrationen "
        "können das Netzwerk verwenden; prüfen Sie die Bedingungen jedes "
        "Modells und Datensatzes.</p>"
    ),
    "README.es.md": (
        "<p><b>Convierta texto clínico en información estructurada y "
        "desidentificada en hardware bajo su control.</b><br/>\n"
        "El entorno local principal de OpenMed realiza la extracción y la "
        "desidentificación una vez disponibles los artefactos de modelo "
        "necesarios. Las descargas de modelos, los adaptadores de proveedores "
        "remotos, las rutas con telemetría y las integraciones configuradas "
        "por el usuario pueden usar la red; revise las condiciones de cada "
        "modelo y conjunto de datos.</p>"
    ),
    "README.fa.md": (
        "<p><b>متن بالینی را روی سخت‌افزار تحت کنترل خود به بینش ساختارمند و "
        "حذف‌هویت‌شده تبدیل کنید.</b><br/>\n"
        "زمان اجرای محلی اصلی OpenMed پس از فراهم شدن مصنوعات مدل موردنیاز، "
        "استخراج و حذف هویت را انجام می‌دهد. دانلود مدل، رابط‌های ارائه‌دهندهٔ "
        "راه‌دور، مسیرهای دارای تله‌متری و یکپارچه‌سازی‌های پیکربندی‌شده توسط "
        "کاربر ممکن است از شبکه استفاده کنند؛ شرایط هر مدل و مجموعه‌داده را "
        "بررسی کنید.</p>"
    ),
    "README.fr.md": (
        "<p><b>Transformez le texte clinique en informations structurées et "
        "dé-identifiées sur le matériel que vous contrôlez.</b><br/>\n"
        "Le moteur local principal d’OpenMed effectue l’extraction et la "
        "dé-identification une fois les artefacts de modèle requis disponibles. "
        "Les téléchargements de modèles, les adaptateurs de fournisseurs "
        "distants, les parcours avec télémétrie et les intégrations configurées "
        "par l’utilisateur peuvent utiliser le réseau ; vérifiez les conditions "
        "de chaque modèle et jeu de données.</p>"
    ),
    "README.hi.md": (
        "<p><b>अपने नियंत्रण वाले हार्डवेयर पर क्लिनिकल टेक्स्ट को संरचित, "
        "डी-आइडेंटिफ़ाइड जानकारी में बदलें।</b><br/>\n"
        "आवश्यक मॉडल आर्टिफ़ैक्ट उपलब्ध होने के बाद OpenMed का मुख्य स्थानीय "
        "रनटाइम निष्कर्षण और डी-आइडेंटिफ़िकेशन करता है। मॉडल डाउनलोड, रिमोट "
        "प्रोवाइडर अडैप्टर, टेलीमेट्री-सक्षम पथ और उपयोगकर्ता द्वारा कॉन्फ़िगर "
        "किए गए एकीकरण नेटवर्क का उपयोग कर सकते हैं; हर मॉडल और डेटासेट की "
        "शर्तें जाँचें।</p>"
    ),
    "README.it.md": (
        "<p><b>Trasforma il testo clinico in informazioni strutturate e "
        "de-identificate sull’hardware che controlli.</b><br/>\n"
        "Il runtime locale principale di OpenMed esegue estrazione e "
        "de-identificazione dopo che gli artefatti di modello richiesti sono "
        "disponibili. Download dei modelli, adattatori per provider remoti, "
        "percorsi con telemetria e integrazioni configurate dall’utente possono "
        "usare la rete; verifica i termini di ogni modello e set di dati.</p>"
    ),
    "README.ja.md": (
        "<p><b>管理下のハードウェアで、臨床テキストを構造化・非識別化された"
        "インサイトに変換します。</b><br/>\n"
        "OpenMed の中核ローカルランタイムは、必要なモデル成果物が利用可能に"
        "なった後に抽出と非識別化を実行します。モデルのダウンロード、リモート"
        "プロバイダー用アダプター、テレメトリ有効経路、ユーザー設定の統合は"
        "ネットワークを利用する場合があります。各モデルとデータセットの条件を"
        "確認してください。</p>"
    ),
    "README.nl.md": (
        "<p><b>Zet klinische tekst om in gestructureerd, "
        "ge-de-identificeerd inzicht op hardware die je beheert.</b><br/>\n"
        "De lokale kernruntime van OpenMed voert extractie en de-identificatie "
        "uit nadat de vereiste modelartefacten beschikbaar zijn. "
        "Modeldownloads, adapters voor externe providers, paden met telemetrie "
        "en door de gebruiker ingestelde integraties kunnen het netwerk "
        "gebruiken; controleer de voorwaarden van elk model en elke dataset.</p>"
    ),
    "README.pt.md": (
        "<p><b>Transforme texto clínico em informação estruturada e "
        "desidentificada no hardware que você controla.</b><br/>\n"
        "O runtime local principal do OpenMed realiza extração e "
        "desidentificação depois que os artefatos de modelo necessários estão "
        "disponíveis. Downloads de modelos, adaptadores de provedores remotos, "
        "caminhos com telemetria e integrações configuradas pelo usuário podem "
        "usar a rede; revise os termos de cada modelo e conjunto de dados.</p>"
    ),
    "README.sw.md": (
        "<p><b>Badili matini ya kliniki kuwa maarifa yaliyopangwa na "
        "yasiyotambulisha mtu kwenye maunzi unayodhibiti.</b><br/>\n"
        "Runtime kuu ya ndani ya OpenMed hufanya uchimbaji na uondoaji "
        "utambulisho baada ya vipengee vya modeli vinavyohitajika kupatikana. "
        "Upakuaji wa modeli, adapta za watoa huduma wa mbali, njia zenye "
        "telemetria na miunganisho iliyosanidiwa na mtumiaji zinaweza kutumia "
        "mtandao; kagua masharti ya kila modeli na mkusanyiko wa data.</p>"
    ),
    "README.te.md": (
        "<p><b>మీ నియంత్రణలోని హార్డ్‌వేర్‌పై క్లినికల్ టెక్స్ట్‌ను నిర్మాణాత్మక, "
        "డీ-ఐడెంటిఫై చేసిన అంతర్దృష్టిగా మార్చండి.</b><br/>\n"
        "అవసరమైన మోడల్ ఆర్టిఫాక్ట్‌లు అందుబాటులోకి వచ్చిన తర్వాత OpenMed ప్రధాన "
        "స్థానిక రన్‌టైమ్ వెలికితీత మరియు డీ-ఐడెంటిఫికేషన్‌ను నిర్వహిస్తుంది. "
        "మోడల్ డౌన్‌లోడ్‌లు, రిమోట్ ప్రొవైడర్ అడాప్టర్‌లు, టెలిమెట్రీ ప్రారంభించిన "
        "మార్గాలు మరియు వినియోగదారు కాన్ఫిగర్ చేసిన అనుసంధానాలు నెట్‌వర్క్‌ను "
        "ఉపయోగించవచ్చు; ప్రతి మోడల్ మరియు డేటాసెట్ నిబంధనలను పరిశీలించండి.</p>"
    ),
    "README.tr.md": (
        "<p><b>Klinik metni kontrolünüzdeki donanımda yapılandırılmış ve "
        "kimliksizleştirilmiş içgörüye dönüştürün.</b><br/>\n"
        "OpenMed’in temel yerel çalışma zamanı, gerekli model yapıtları hazır "
        "olduktan sonra çıkarım ve kimliksizleştirme yapar. Model indirmeleri, "
        "uzak sağlayıcı bağdaştırıcıları, telemetri etkin yollar ve kullanıcı "
        "tarafından yapılandırılan entegrasyonlar ağı kullanabilir; her modelin "
        "ve veri kümesinin koşullarını inceleyin.</p>"
    ),
    "README.zh-CN.md": (
        "<p><b>在你掌控的硬件上，将临床文本转化为结构化、去标识化的洞见。"
        "</b><br/>\n"
        "OpenMed 的核心本地运行时会在所需模型制品就绪后执行抽取和去标识化。"
        "模型下载、远程提供商适配器、启用遥测的路径及用户配置的集成可能使用"
        "网络；请核查每个模型和数据集的条款。</p>"
    ),
}

DEPLOYMENT_TABLES = {
    "README.md": """| Deployment consideration | OpenMed SDK boundary |
| --- | --- |
| Core runtime | Processes locally after required artifacts are available |
| Optional network paths | Downloads, remote adapters, telemetry-enabled paths, and user integrations may use a network |
| Validation | Deployment owners validate model and dataset terms, privacy behavior, and clinical fitness |
| Interfaces | Python, Swift, Android, browser, and service surfaces where supported |""",
    "README.ar.md": """| اعتبار النشر | حدود OpenMed SDK |
| --- | --- |
| وقت التشغيل الأساسي | يعالج محليًا بعد توفر العناصر المطلوبة |
| مسارات الشبكة الاختيارية | قد تستخدم التنزيلات والموائمات البعيدة ومسارات القياس والتكاملات الشبكة |
| التحقق | يتحقق مالك النشر من شروط النماذج والبيانات وسلوك الخصوصية والملاءمة السريرية |
| الواجهات | Python وSwift وAndroid والمتصفح والخدمات حيث تكون مدعومة |""",
    "README.de.md": """| Bereitstellungsaspekt | Grenze des OpenMed SDK |
| --- | --- |
| Kernlaufzeit | Verarbeitet lokal, sobald die benötigten Artefakte vorliegen |
| Optionale Netzwerkpfade | Downloads, entfernte Adapter, Telemetriepfade und Integrationen können das Netzwerk nutzen |
| Validierung | Betreiber prüfen Modell- und Datenbedingungen, Datenschutzverhalten und klinische Eignung |
| Schnittstellen | Python, Swift, Android, Browser und Dienste, soweit unterstützt |""",
    "README.es.md": """| Consideración de despliegue | Límite del SDK de OpenMed |
| --- | --- |
| Entorno principal | Procesa localmente cuando están disponibles los artefactos necesarios |
| Rutas de red opcionales | Descargas, adaptadores remotos, telemetría e integraciones pueden usar la red |
| Validación | El responsable valida términos de modelos y datos, privacidad y aptitud clínica |
| Interfaces | Python, Swift, Android, navegador y servicios cuando son compatibles |""",
    "README.fa.md": """| ملاحظهٔ استقرار | مرز OpenMed SDK |
| --- | --- |
| زمان اجرای اصلی | پس از فراهم شدن مصنوعات لازم، محلی پردازش می‌کند |
| مسیرهای اختیاری شبکه | دانلودها، رابط‌های راه‌دور، تله‌متری و یکپارچه‌سازی‌ها ممکن است از شبکه استفاده کنند |
| اعتبارسنجی | مالک استقرار شرایط مدل و داده، رفتار حریم خصوصی و تناسب بالینی را بررسی می‌کند |
| رابط‌ها | Python، Swift، Android، مرورگر و سرویس‌ها در صورت پشتیبانی |""",
    "README.fr.md": """| Point de déploiement | Périmètre du SDK OpenMed |
| --- | --- |
| Moteur principal | Traite localement une fois les artefacts requis disponibles |
| Parcours réseau facultatifs | Téléchargements, adaptateurs distants, télémétrie et intégrations peuvent utiliser le réseau |
| Validation | Le responsable valide les conditions des modèles et données, la confidentialité et l’aptitude clinique |
| Interfaces | Python, Swift, Android, navigateur et services lorsqu’ils sont pris en charge |""",
    "README.hi.md": """| परिनियोजन विचार | OpenMed SDK की सीमा |
| --- | --- |
| मुख्य रनटाइम | आवश्यक आर्टिफ़ैक्ट उपलब्ध होने के बाद स्थानीय रूप से प्रोसेस करता है |
| वैकल्पिक नेटवर्क पथ | डाउनलोड, रिमोट अडैप्टर, टेलीमेट्री पथ और एकीकरण नेटवर्क उपयोग कर सकते हैं |
| सत्यापन | परिनियोजन स्वामी मॉडल व डेटा शर्तें, गोपनीयता व्यवहार और क्लिनिकल उपयुक्तता जाँचता है |
| इंटरफ़ेस | जहाँ समर्थित हों वहाँ Python, Swift, Android, ब्राउज़र और सेवाएँ |""",
    "README.it.md": """| Aspetto di distribuzione | Perimetro dell’SDK OpenMed |
| --- | --- |
| Runtime principale | Elabora localmente dopo la disponibilità degli artefatti richiesti |
| Percorsi di rete opzionali | Download, adattatori remoti, telemetria e integrazioni possono usare la rete |
| Validazione | Il responsabile verifica termini di modelli e dati, privacy e idoneità clinica |
| Interfacce | Python, Swift, Android, browser e servizi dove supportati |""",
    "README.ja.md": """| デプロイ時の考慮事項 | OpenMed SDK の境界 |
| --- | --- |
| 中核ランタイム | 必要な成果物が利用可能になった後、ローカルで処理 |
| 任意のネットワーク経路 | ダウンロード、リモートアダプター、テレメトリ経路、統合はネットワークを使う場合あり |
| 検証 | デプロイ責任者がモデル・データ条件、プライバシー動作、臨床適合性を検証 |
| インターフェース | 対応範囲で Python、Swift、Android、ブラウザ、サービス |""",
    "README.nl.md": """| Implementatieaspect | Grens van de OpenMed SDK |
| --- | --- |
| Kernruntime | Verwerkt lokaal nadat de vereiste artefacten beschikbaar zijn |
| Optionele netwerkpaden | Downloads, externe adapters, telemetriepaden en integraties kunnen het netwerk gebruiken |
| Validatie | De beheerder valideert model- en datavoorwaarden, privacygedrag en klinische geschiktheid |
| Interfaces | Python, Swift, Android, browser en diensten waar ondersteund |""",
    "README.pt.md": """| Consideração de implantação | Limite do SDK OpenMed |
| --- | --- |
| Runtime principal | Processa localmente após os artefatos necessários estarem disponíveis |
| Caminhos de rede opcionais | Downloads, adaptadores remotos, telemetria e integrações podem usar a rede |
| Validação | O responsável valida termos de modelos e dados, privacidade e adequação clínica |
| Interfaces | Python, Swift, Android, navegador e serviços quando compatíveis |""",
    "README.sw.md": """| Jambo la kuzingatia katika usambazaji | Mpaka wa OpenMed SDK |
| --- | --- |
| Runtime kuu | Huchakata ndani baada ya vipengee vinavyohitajika kupatikana |
| Njia za hiari za mtandao | Upakuaji, adapta za mbali, telemetria na miunganisho zinaweza kutumia mtandao |
| Uthibitishaji | Mmiliki hukagua masharti ya modeli na data, faragha na ufaafu wa kliniki |
| Violesura | Python, Swift, Android, kivinjari na huduma pale zinapoungwa mkono |""",
    "README.te.md": """| అమలు పరిశీలన | OpenMed SDK పరిధి |
| --- | --- |
| ప్రధాన రన్‌టైమ్ | అవసరమైన ఆర్టిఫాక్ట్‌లు అందుబాటులోకి వచ్చిన తర్వాత స్థానికంగా ప్రాసెస్ చేస్తుంది |
| ఐచ్ఛిక నెట్‌వర్క్ మార్గాలు | డౌన్‌లోడ్‌లు, రిమోట్ అడాప్టర్‌లు, టెలిమెట్రీ మరియు అనుసంధానాలు నెట్‌వర్క్‌ను ఉపయోగించవచ్చు |
| ధృవీకరణ | అమలు యజమాని మోడల్, డేటా నిబంధనలు, గోప్యత ప్రవర్తన మరియు క్లినికల్ అనుకూలతను ధృవీకరిస్తారు |
| ఇంటర్‌ఫేస్‌లు | మద్దతు ఉన్న చోట Python, Swift, Android, బ్రౌజర్ మరియు సేవలు |""",
    "README.tr.md": """| Dağıtım değerlendirmesi | OpenMed SDK sınırı |
| --- | --- |
| Temel çalışma zamanı | Gerekli yapıtlar hazır olduktan sonra yerel olarak işler |
| İsteğe bağlı ağ yolları | İndirmeler, uzak bağdaştırıcılar, telemetri ve entegrasyonlar ağı kullanabilir |
| Doğrulama | Dağıtım sahibi model ve veri koşullarını, gizlilik davranışını ve klinik uygunluğu doğrular |
| Arayüzler | Desteklendiği yerlerde Python, Swift, Android, tarayıcı ve hizmetler |""",
    "README.zh-CN.md": """| 部署考量 | OpenMed SDK 边界 |
| --- | --- |
| 核心运行时 | 所需制品就绪后在本地处理 |
| 可选网络路径 | 下载、远程适配器、遥测路径和用户集成可能使用网络 |
| 验证 | 部署方验证模型与数据条款、隐私行为及临床适用性 |
| 接口 | 在受支持的范围内提供 Python、Swift、Android、浏览器和服务接口 |""",
}

CAPABILITY_BULLETS = {
    "README.md": """- **Curated model catalog**: validate each model, license, and dataset for your use case.
- **Safe Harbor-aligned configuration**: can target the 18 identifier categories; expert deployment review remains required, and use of the SDK does not itself establish HIPAA compliance.
- **Supported execution paths**: CPU, CUDA, MLX, mobile, service, and browser adapters vary by environment and artifact.
- **Deployment interfaces**: Python, containers, services, and batch workflows require configuration and validation.""",
    "README.ar.md": """- **فهرس نماذج منسّق**: تحقّق من كل نموذج وترخيص ومجموعة بيانات لحالة الاستخدام.
- **تهيئة متوافقة مع فئات Safe Harbor**: يمكنها استهداف فئات المعرّفات الثماني عشرة؛ تبقى مراجعة خبير للنشر مطلوبة، واستخدام SDK لا يثبت بحد ذاته الامتثال لـ HIPAA.
- **مسارات تنفيذ مدعومة**: تختلف محوّلات CPU وCUDA وMLX والهاتف والخدمة والمتصفح حسب البيئة والعنصر.
- **واجهات النشر**: تتطلب Python والحاويات والخدمات ومسارات الدُفعات تهيئةً وتحققًا.""",
    "README.de.md": """- **Kuratierter Modellkatalog**: Jedes Modell, jede Lizenz und jeder Datensatz muss für den Einsatzzweck geprüft werden.
- **An Safe Harbor ausgerichtete Konfiguration**: Kann die 18 Identifikatorkategorien adressieren; fachliche Bereitstellungsprüfung bleibt erforderlich, und die SDK-Nutzung belegt für sich keine HIPAA-Konformität.
- **Unterstützte Ausführungspfade**: CPU-, CUDA-, MLX-, Mobil-, Dienst- und Browseradapter hängen von Umgebung und Artefakt ab.
- **Bereitstellungsschnittstellen**: Python, Container, Dienste und Stapelabläufe erfordern Konfiguration und Validierung.""",
    "README.es.md": """- **Catálogo de modelos seleccionado**: valide cada modelo, licencia y conjunto de datos para su caso de uso.
- **Configuración alineada con Safe Harbor**: puede dirigirse a las 18 categorías de identificadores; sigue siendo necesaria la revisión experta del despliegue y usar el SDK no demuestra por sí solo el cumplimiento de HIPAA.
- **Rutas de ejecución compatibles**: los adaptadores de CPU, CUDA, MLX, móvil, servicio y navegador varían según el entorno y el artefacto.
- **Interfaces de despliegue**: Python, contenedores, servicios y flujos por lotes requieren configuración y validación.""",
    "README.fa.md": """- **فهرست گزینش‌شدهٔ مدل‌ها**: هر مدل، مجوز و مجموعه‌داده را برای کاربرد خود اعتبارسنجی کنید.
- **پیکربندی هم‌راستا با Safe Harbor**: می‌تواند ۱۸ دستهٔ شناسه را هدف بگیرد؛ بازبینی تخصصی استقرار همچنان لازم است و استفاده از SDK به‌تنهایی انطباق HIPAA را اثبات نمی‌کند.
- **مسیرهای اجرای پشتیبانی‌شده**: رابط‌های CPU، CUDA، MLX، موبایل، سرویس و مرورگر به محیط و مصنوع بستگی دارند.
- **رابط‌های استقرار**: Python، کانتینرها، سرویس‌ها و جریان‌های دسته‌ای به پیکربندی و اعتبارسنجی نیاز دارند.""",
    "README.fr.md": """- **Catalogue de modèles sélectionné** : validez chaque modèle, licence et jeu de données pour votre cas d’usage.
- **Configuration alignée sur Safe Harbor** : peut cibler les 18 catégories d’identifiants ; une revue experte du déploiement reste requise et l’utilisation du SDK n’établit pas à elle seule la conformité HIPAA.
- **Parcours d’exécution pris en charge** : les adaptateurs CPU, CUDA, MLX, mobile, service et navigateur varient selon l’environnement et l’artefact.
- **Interfaces de déploiement** : Python, conteneurs, services et traitements par lots nécessitent configuration et validation.""",
    "README.hi.md": """- **चयनित मॉडल कैटलॉग**: अपने उपयोग के लिए हर मॉडल, लाइसेंस और डेटासेट को सत्यापित करें।
- **Safe Harbor-संरेखित कॉन्फ़िगरेशन**: 18 पहचानकर्ता श्रेणियों को लक्षित कर सकता है; विशेषज्ञ परिनियोजन समीक्षा आवश्यक रहती है और केवल SDK का उपयोग HIPAA अनुपालन सिद्ध नहीं करता।
- **समर्थित निष्पादन पथ**: CPU, CUDA, MLX, मोबाइल, सेवा और ब्राउज़र अडैप्टर वातावरण व आर्टिफ़ैक्ट के अनुसार बदलते हैं।
- **परिनियोजन इंटरफ़ेस**: Python, कंटेनर, सेवाएँ और बैच वर्कफ़्लो कॉन्फ़िगरेशन व सत्यापन माँगते हैं।""",
    "README.it.md": """- **Catalogo di modelli selezionato**: convalida ogni modello, licenza e set di dati per il tuo caso d’uso.
- **Configurazione allineata a Safe Harbor**: può individuare le 18 categorie di identificatori; resta necessaria una revisione esperta del deployment e l’uso dell’SDK non dimostra da solo la conformità HIPAA.
- **Percorsi di esecuzione supportati**: gli adattatori CPU, CUDA, MLX, mobile, servizio e browser variano in base ad ambiente e artefatto.
- **Interfacce di deployment**: Python, container, servizi e flussi batch richiedono configurazione e convalida.""",
    "README.ja.md": """- **精選モデルカタログ**：用途ごとに各モデル、ライセンス、データセットを検証してください。
- **Safe Harbor に沿った設定**：18 の識別子カテゴリを対象にできますが、専門家による導入レビューが必要です。SDK の利用だけで HIPAA 準拠を示すものではありません。
- **対応する実行経路**：CPU、CUDA、MLX、モバイル、サービス、ブラウザのアダプターは環境と成果物によって異なります。
- **導入インターフェース**：Python、コンテナ、サービス、バッチ処理には設定と検証が必要です。""",
    "README.nl.md": """- **Samengestelde modelcatalogus**: valideer elk model, elke licentie en dataset voor je gebruikssituatie.
- **Op Safe Harbor afgestemde configuratie**: kan de 18 identificatorcategorieën adresseren; deskundige implementatiebeoordeling blijft vereist en gebruik van de SDK toont op zichzelf geen HIPAA-naleving aan.
- **Ondersteunde uitvoerpaden**: CPU-, CUDA-, MLX-, mobiele, service- en browseradapters verschillen per omgeving en artefact.
- **Implementatie-interfaces**: Python, containers, services en batchworkflows vereisen configuratie en validatie.""",
    "README.pt.md": """- **Catálogo de modelos selecionado**: valide cada modelo, licença e conjunto de dados para seu caso de uso.
- **Configuração alinhada ao Safe Harbor**: pode abranger as 18 categorias de identificadores; a revisão especializada da implantação continua necessária e o uso do SDK, isoladamente, não comprova conformidade com HIPAA.
- **Caminhos de execução compatíveis**: adaptadores de CPU, CUDA, MLX, dispositivo móvel, serviço e navegador variam conforme o ambiente e o artefato.
- **Interfaces de implantação**: Python, contêineres, serviços e fluxos em lote exigem configuração e validação.""",
    "README.sw.md": """- **Katalogi ya modeli iliyochaguliwa**: thibitisha kila modeli, leseni na mkusanyiko wa data kwa matumizi yako.
- **Usanidi unaolingana na Safe Harbor**: unaweza kulenga makundi 18 ya vitambulishi; ukaguzi wa kitaalamu wa utekelezaji bado unahitajika, na kutumia SDK pekee hakuthibitishi utiifu wa HIPAA.
- **Njia za utekelezaji zinazotumika**: adapta za CPU, CUDA, MLX, simu, huduma na kivinjari hutegemea mazingira na kielelezo.
- **Violesura vya uwekaji**: Python, kontena, huduma na michakato ya bechi huhitaji usanidi na uthibitishaji.""",
    "README.te.md": """- **ఎంపిక చేసిన మోడల్ కేటలాగ్**: మీ వినియోగానికి ప్రతి మోడల్, లైసెన్స్ మరియు డేటాసెట్‌ను ధృవీకరించండి.
- **Safe Harbor‌కు సరిపోలే కాన్ఫిగరేషన్**: 18 ఐడెంటిఫయర్ వర్గాలను లక్ష్యంగా చేసుకోగలదు; నిపుణుల అమలు సమీక్ష ఇంకా అవసరం, SDK వాడకం మాత్రమే HIPAA అనుసరణను నిరూపించదు.
- **మద్దతు ఉన్న అమలు మార్గాలు**: CPU, CUDA, MLX, మొబైల్, సేవ మరియు బ్రౌజర్ అడాప్టర్‌లు పర్యావరణం, ఆర్టిఫాక్ట్‌ను బట్టి మారుతాయి.
- **అమలు ఇంటర్‌ఫేస్‌లు**: Python, కంటైనర్‌లు, సేవలు మరియు బ్యాచ్ వర్క్‌ఫ్లోలకు కాన్ఫిగరేషన్, ధృవీకరణ అవసరం.""",
    "README.tr.md": """- **Seçilmiş model kataloğu**: Her modeli, lisansı ve veri kümesini kullanım amacınız için doğrulayın.
- **Safe Harbor ile uyumlu yapılandırma**: 18 tanımlayıcı kategorisini hedefleyebilir; uzman dağıtım incelemesi yine gereklidir ve SDK kullanımı tek başına HIPAA uyumluluğunu kanıtlamaz.
- **Desteklenen yürütme yolları**: CPU, CUDA, MLX, mobil, hizmet ve tarayıcı bağdaştırıcıları ortama ve yapıta göre değişir.
- **Dağıtım arayüzleri**: Python, kapsayıcılar, hizmetler ve toplu iş akışları yapılandırma ve doğrulama gerektirir.""",
    "README.zh-CN.md": """- **精选模型目录**：请针对你的用例验证每个模型、许可证和数据集。
- **与 Safe Harbor 对齐的配置**：可面向 18 类标识符；仍需专家开展部署审查，使用 SDK 本身并不能证明符合 HIPAA。
- **受支持的执行路径**：CPU、CUDA、MLX、移动端、服务和浏览器适配器会因环境与制品而异。
- **部署接口**：Python、容器、服务和批处理工作流需要配置与验证。""",
}

HIPAA_BOUNDARY_LINES = {
    "README.md": "- **HIPAA boundary**: Safe Harbor-aligned categories and configurable thresholds are implementation aids; expert deployment review remains required, and SDK use alone does not establish compliance.",
    "README.ar.md": "- **حدود HIPAA**: فئات متوافقة مع Safe Harbor وعتبات قابلة للتهيئة هي أدوات مساعدة للتنفيذ؛ تبقى مراجعة خبير للنشر مطلوبة ولا يثبت استخدام SDK وحده الامتثال.",
    "README.de.md": "- **HIPAA-Grenze**: An Safe Harbor ausgerichtete Kategorien und konfigurierbare Schwellen sind Implementierungshilfen; eine fachliche Bereitstellungsprüfung bleibt erforderlich und die SDK-Nutzung allein belegt keine Konformität.",
    "README.es.md": "- **Límite de HIPAA**: las categorías alineadas con Safe Harbor y los umbrales configurables ayudan a la implementación; sigue siendo necesaria la revisión experta del despliegue y usar el SDK no demuestra por sí solo el cumplimiento.",
    "README.fa.md": "- **مرز HIPAA**: دسته‌های هم‌راستا با Safe Harbor و آستانه‌های قابل‌تنظیم ابزارهای پیاده‌سازی‌اند؛ بازبینی تخصصی استقرار همچنان لازم است و استفاده از SDK به‌تنهایی انطباق را اثبات نمی‌کند.",
    "README.fr.md": "- **Périmètre HIPAA** : les catégories alignées sur Safe Harbor et les seuils configurables sont des aides à l’implémentation ; une revue experte du déploiement reste requise et l’utilisation du SDK seule n’établit pas la conformité.",
    "README.hi.md": "- **HIPAA सीमा**: Safe Harbor-संरेखित श्रेणियाँ और कॉन्फ़िगर करने योग्य सीमाएँ कार्यान्वयन सहायक हैं; विशेषज्ञ परिनियोजन समीक्षा आवश्यक रहती है और केवल SDK उपयोग अनुपालन सिद्ध नहीं करता।",
    "README.it.md": "- **Perimetro HIPAA**: categorie allineate a Safe Harbor e soglie configurabili sono strumenti di implementazione; resta necessaria una revisione esperta del deployment e l’uso del solo SDK non dimostra la conformità.",
    "README.ja.md": "- **HIPAA の境界**：Safe Harbor に沿ったカテゴリと設定可能なしきい値は実装支援です。専門家による導入レビューが必要であり、SDK の利用だけで準拠を示すものではありません。",
    "README.nl.md": "- **HIPAA-grens**: op Safe Harbor afgestemde categorieën en instelbare drempels zijn implementatiehulpmiddelen; deskundige implementatiebeoordeling blijft vereist en alleen SDK-gebruik toont geen naleving aan.",
    "README.pt.md": "- **Limite da HIPAA**: categorias alinhadas ao Safe Harbor e limiares configuráveis auxiliam a implementação; a revisão especializada da implantação continua necessária e o uso isolado do SDK não comprova conformidade.",
    "README.sw.md": "- **Mpaka wa HIPAA**: makundi yanayolingana na Safe Harbor na vizingiti vinavyosanidiwa ni vifaa vya utekelezaji; ukaguzi wa kitaalamu bado unahitajika na kutumia SDK pekee hakuthibitishi utiifu.",
    "README.te.md": "- **HIPAA పరిధి**: Safe Harbor‌కు సరిపోలే వర్గాలు, కాన్ఫిగర్ చేయగల థ్రెష్‌హోల్డ్‌లు అమలు సహాయకాలు; నిపుణుల అమలు సమీక్ష ఇంకా అవసరం, SDK వాడకం మాత్రమే అనుసరణను నిరూపించదు.",
    "README.tr.md": "- **HIPAA sınırı**: Safe Harbor ile uyumlu kategoriler ve yapılandırılabilir eşikler uygulama yardımcılarıdır; uzman dağıtım incelemesi yine gereklidir ve yalnızca SDK kullanımı uyumluluğu kanıtlamaz.",
    "README.zh-CN.md": "- **HIPAA 边界**：与 Safe Harbor 对齐的类别和可配置阈值属于实现辅助；仍需专家开展部署审查，仅使用 SDK 并不能证明合规。",
}

LANGUAGE_HEADINGS = {
    "README.md": (
        "## Multilingual PII ({supported} supported routes; "
        "{model_backed} model-backed)"
    ),
    "README.ar.md": (
        "## PII متعدد اللغات ({supported} مسارًا مدعومًا؛ {model_backed} مدعومًا بالنماذج)"
    ),
    "README.de.md": (
        "## Mehrsprachige PII ({supported} unterstützte Routen; "
        "{model_backed} modellgestützt)"
    ),
    "README.es.md": (
        "## PII multilingüe ({supported} rutas admitidas; "
        "{model_backed} respaldadas por modelos)"
    ),
    "README.fa.md": (
        "## PII چندزبانه ({supported} مسیر پشتیبانی‌شده؛ {model_backed} مسیر با مدل)"
    ),
    "README.fr.md": (
        "## PII multilingue ({supported} routes prises en charge ; "
        "{model_backed} prises en charge par modèle)"
    ),
    "README.hi.md": (
        "## बहुभाषी PII ({supported} समर्थित रूट; {model_backed} मॉडल-समर्थित)"
    ),
    "README.it.md": (
        "## PII multilingue ({supported} route supportate; "
        "{model_backed} supportate da modelli)"
    ),
    "README.ja.md": (
        "## 多言語 PII（対応ルート {supported}、モデル対応 {model_backed}）"
    ),
    "README.nl.md": (
        "## Meertalige PII ({supported} ondersteunde routes; "
        "{model_backed} modelondersteund)"
    ),
    "README.pt.md": (
        "## PII multilíngue ({supported} rotas suportadas; "
        "{model_backed} com suporte de modelos)"
    ),
    "README.sw.md": (
        "## PII ya lugha nyingi (njia {supported} zinazotumika; "
        "{model_backed} zikitumia modeli)"
    ),
    "README.te.md": (
        "## బహుభాషా PII ({supported} మద్దతు ఉన్న మార్గాలు; {model_backed} మోడల్ మద్దతుతో)"
    ),
    "README.tr.md": (
        "## Çok dilli PII ({supported} desteklenen yol; {model_backed} model destekli)"
    ),
    "README.zh-CN.md": (
        "## 多语言 PII（{supported} 条支持的路由；{model_backed} 条由模型支持）"
    ),
}

PRIVACY_HEADINGS = {
    "README.md": "## Privacy: PII detection & de-identification",
    "README.ar.md": "## الخصوصية: كشف PII وإزالة الهوية",
    "README.de.md": "## Datenschutz: PII-Erkennung & De-Identifikation",
    "README.es.md": "## Privacidad: detección y des-identificación de PII",
    "README.fa.md": "## حریمِ خصوصی: تشخیص و حذفِ PII",
    "README.fr.md": "## Confidentialité : détection et dé-identification des PII",
    "README.hi.md": "## गोपनीयता: PII पहचान और डी-आइडेंटिफिकेशन",
    "README.it.md": "## Privacy: rilevamento e de-identificazione dei PII",
    "README.ja.md": "## プライバシー：PII 検出と非識別化",
    "README.nl.md": "## Privacy: PII-detectie & de-identificatie",
    "README.pt.md": "## Privacidade: detecção e des-identificação de PII",
    "README.sw.md": "## Faragha: utambuzi wa PII na uondoaji utambulisho",
    "README.te.md": "## గోప్యత: PII గుర్తింపు & డీ-ఐడెంటిఫికేషన్",
    "README.tr.md": "## Gizlilik: PII tespiti ve kimliksizleştirme",
    "README.zh-CN.md": "## 隐私：PII 检测与去标识化",
}

LICENSE_LINES = {
    "README.md": (
        "The OpenMed SDK source is released under the "
        "[Apache-2.0 License](LICENSE). Third-party asset notices are recorded "
        "in [NOTICE](NOTICE)."
    ),
    "README.ar.md": (
        "يُنشر الكود المصدري لحزمة OpenMed SDK بموجب [Apache-2.0 License](LICENSE)."
    ),
    "README.de.md": (
        "Der Quellcode des OpenMed SDK wird unter der "
        "[Apache-2.0 License](LICENSE) veröffentlicht."
    ),
    "README.es.md": (
        "El código fuente del SDK de OpenMed se publica bajo la "
        "[Apache-2.0 License](LICENSE)."
    ),
    "README.fa.md": (
        "کد منبع OpenMed SDK تحت [Apache-2.0 License](LICENSE) منتشر می‌شود."
    ),
    "README.fr.md": (
        "Le code source du SDK OpenMed est publié sous [Apache-2.0 License](LICENSE)."
    ),
    "README.hi.md": (
        "OpenMed SDK का स्रोत [Apache-2.0 License](LICENSE) के अंतर्गत जारी "
        "किया गया है। Third-party asset notices [NOTICE](NOTICE) में दर्ज हैं।"
    ),
    "README.it.md": (
        "Il codice sorgente dell’SDK OpenMed è distribuito con "
        "[Apache-2.0 License](LICENSE)."
    ),
    "README.ja.md": (
        "OpenMed SDK のソースは [Apache-2.0 License](LICENSE) の下で公開されています。"
    ),
    "README.nl.md": (
        "De broncode van de OpenMed SDK wordt uitgebracht onder de "
        "[Apache-2.0 License](LICENSE)."
    ),
    "README.pt.md": (
        "O código-fonte do SDK OpenMed é publicado sob a [Apache-2.0 License](LICENSE)."
    ),
    "README.sw.md": (
        "Msimbo chanzo wa OpenMed SDK umetolewa chini ya "
        "[Apache-2.0 License](LICENSE). Taarifa za mali za watu wengine "
        "zimeandikwa katika [NOTICE](NOTICE)."
    ),
    "README.te.md": (
        "OpenMed SDK సోర్స్ [Apache-2.0 License](LICENSE) క్రింద విడుదల చేయబడింది."
    ),
    "README.tr.md": (
        "OpenMed SDK kaynak kodu [Apache-2.0 License](LICENSE) altında yayımlanmıştır."
    ),
    "README.zh-CN.md": (
        "OpenMed SDK 源代码基于 [Apache-2.0 License](LICENSE) 发布。"
        "第三方资源声明记录在 [NOTICE](NOTICE) 中。"
    ),
}

EXAMPLE_COPY = {
    "README.md": (
        "A clinical NER model using the local runtime after its required "
        "artifacts are available."
    ),
    "README.ar.md": (
        '<div dir="rtl">\n\n'
        "نموذج NER سريري يستخدم وقت التشغيل المحلي بعد توفر العناصر المطلوبة."
        "\n\n</div>"
    ),
    "README.de.md": (
        "Ein klinisches NER-Modell nutzt die lokale Laufzeit, nachdem die "
        "benötigten Artefakte verfügbar sind."
    ),
    "README.es.md": (
        "Un modelo de NER clínico usa el entorno local después de que sus "
        "artefactos necesarios están disponibles."
    ),
    "README.fa.md": (
        '<div dir="rtl">\n\n'
        "یک مدل NER بالینی پس از فراهم شدن مصنوعات موردنیاز از زمان اجرای "
        "محلی استفاده می‌کند.\n\n</div>"
    ),
    "README.fr.md": (
        "Un modèle de NER clinique utilise le moteur local une fois ses "
        "artefacts requis disponibles."
    ),
    "README.hi.md": (
        "आवश्यक आर्टिफ़ैक्ट उपलब्ध होने के बाद एक क्लिनिकल NER मॉडल स्थानीय "
        "रनटाइम का उपयोग करता है।"
    ),
    "README.it.md": (
        "Un modello NER clinico usa il runtime locale dopo che i suoi "
        "artefatti richiesti sono disponibili."
    ),
    "README.ja.md": (
        "臨床 NER モデルは、必要な成果物が利用可能になった後にローカル"
        "ランタイムを使用します。"
    ),
    "README.nl.md": (
        "Een klinisch NER-model gebruikt de lokale runtime nadat de vereiste "
        "artefacten beschikbaar zijn."
    ),
    "README.pt.md": (
        "Um modelo de NER clínico usa o runtime local depois que os artefatos "
        "necessários estão disponíveis."
    ),
    "README.sw.md": (
        "Modeli ya NER ya kliniki hutumia runtime ya ndani baada ya vipengee "
        "vinavyohitajika kupatikana."
    ),
    "README.te.md": (
        "అవసరమైన ఆర్టిఫాక్ట్‌లు అందుబాటులోకి వచ్చిన తర్వాత క్లినికల్ NER మోడల్ స్థానిక రన్‌టైమ్‌ను ఉపయోగిస్తుంది."
    ),
    "README.tr.md": (
        "Bir klinik NER modeli, gerekli yapıtları hazır olduktan sonra yerel "
        "çalışma zamanını kullanır."
    ),
    "README.zh-CN.md": ("临床 NER 模型会在所需制品就绪后使用本地运行时。"),
}

APPLE_COPY = {
    "README.md": (
        "On supported Apple hardware, OpenMed can use **MLX** and "
        "**[OpenMedKit](swift/OpenMedKit)** for local processing after "
        "required artifacts are available. Model acquisition and any "
        "user-configured remote integrations remain separate network "
        "boundaries."
    ),
    "README.ar.md": (
        "على عتاد Apple المدعوم، يمكن لـ OpenMed استخدام **MLX** و"
        "**[OpenMedKit](swift/OpenMedKit)** للمعالجة المحلية بعد توفر العناصر "
        "المطلوبة. ويظل الحصول على النماذج وأي تكاملات بعيدة يهيئها المستخدم "
        "حدودًا شبكية منفصلة."
    ),
    "README.de.md": (
        "Auf unterstützter Apple-Hardware kann OpenMed **MLX** und "
        "**[OpenMedKit](swift/OpenMedKit)** für lokale Verarbeitung nutzen, "
        "sobald die benötigten Artefakte vorliegen. Modellbezug und "
        "nutzerkonfigurierte entfernte Integrationen bleiben getrennte "
        "Netzwerkgrenzen."
    ),
    "README.es.md": (
        "En hardware Apple compatible, OpenMed puede usar **MLX** y "
        "**[OpenMedKit](swift/OpenMedKit)** para el procesamiento local una "
        "vez disponibles los artefactos necesarios. La obtención de modelos y "
        "las integraciones remotas configuradas por el usuario son límites de "
        "red independientes."
    ),
    "README.fa.md": (
        "روی سخت‌افزار پشتیبانی‌شدهٔ Apple، OpenMed می‌تواند پس از فراهم شدن "
        "مصنوعات لازم از **MLX** و **[OpenMedKit](swift/OpenMedKit)** برای "
        "پردازش محلی استفاده کند. دریافت مدل و یکپارچه‌سازی‌های راه‌دور "
        "پیکربندی‌شده توسط کاربر، مرزهای شبکه‌ای جداگانه هستند."
    ),
    "README.fr.md": (
        "Sur le matériel Apple pris en charge, OpenMed peut utiliser **MLX** "
        "et **[OpenMedKit](swift/OpenMedKit)** pour le traitement local une "
        "fois les artefacts requis disponibles. L’acquisition des modèles et "
        "les intégrations distantes configurées par l’utilisateur restent des "
        "frontières réseau distinctes."
    ),
    "README.hi.md": (
        "समर्थित Apple हार्डवेयर पर, आवश्यक आर्टिफ़ैक्ट उपलब्ध होने के बाद "
        "OpenMed स्थानीय प्रोसेसिंग के लिए **MLX** और "
        "**[OpenMedKit](swift/OpenMedKit)** का उपयोग कर सकता है। मॉडल प्राप्ति "
        "और उपयोगकर्ता द्वारा कॉन्फ़िगर किए गए रिमोट एकीकरण अलग नेटवर्क सीमाएँ "
        "हैं।"
    ),
    "README.it.md": (
        "Sull’hardware Apple supportato, OpenMed può usare **MLX** e "
        "**[OpenMedKit](swift/OpenMedKit)** per l’elaborazione locale dopo la "
        "disponibilità degli artefatti richiesti. L’acquisizione dei modelli e "
        "le integrazioni remote configurate dall’utente restano confini di "
        "rete separati."
    ),
    "README.ja.md": (
        "対応する Apple ハードウェアでは、必要な成果物が利用可能になった後、"
        "OpenMed は **MLX** と **[OpenMedKit](swift/OpenMedKit)** を使って"
        "ローカル処理できます。モデル取得とユーザー設定のリモート統合は別の"
        "ネットワーク境界です。"
    ),
    "README.nl.md": (
        "Op ondersteunde Apple-hardware kan OpenMed **MLX** en "
        "**[OpenMedKit](swift/OpenMedKit)** gebruiken voor lokale verwerking "
        "nadat de vereiste artefacten beschikbaar zijn. Modelverwerving en "
        "door de gebruiker ingestelde externe integraties blijven afzonderlijke "
        "netwerkgrenzen."
    ),
    "README.pt.md": (
        "Em hardware Apple compatível, o OpenMed pode usar **MLX** e "
        "**[OpenMedKit](swift/OpenMedKit)** para processamento local após os "
        "artefatos necessários estarem disponíveis. A obtenção de modelos e "
        "as integrações remotas configuradas pelo usuário continuam sendo "
        "limites de rede separados."
    ),
    "README.sw.md": (
        "Kwenye maunzi ya Apple yanayoungwa mkono, OpenMed inaweza kutumia "
        "**MLX** na **[OpenMedKit](swift/OpenMedKit)** kwa uchakataji wa ndani "
        "baada ya vipengee vinavyohitajika kupatikana. Upatikanaji wa modeli na "
        "miunganisho ya mbali iliyosanidiwa na mtumiaji hubaki mipaka tofauti "
        "ya mtandao."
    ),
    "README.te.md": (
        "మద్దతు ఉన్న Apple హార్డ్‌వేర్‌పై, అవసరమైన ఆర్టిఫాక్ట్‌లు అందుబాటులోకి "
        "వచ్చిన తర్వాత OpenMed స్థానిక ప్రాసెసింగ్ కోసం **MLX** మరియు "
        "**[OpenMedKit](swift/OpenMedKit)**ను ఉపయోగించగలదు. మోడల్ సేకరణ మరియు "
        "వినియోగదారు కాన్ఫిగర్ చేసిన రిమోట్ అనుసంధానాలు వేర్వేరు నెట్‌వర్క్ "
        "పరిధులుగా ఉంటాయి."
    ),
    "README.tr.md": (
        "Desteklenen Apple donanımında OpenMed, gerekli yapıtlar hazır olduktan "
        "sonra yerel işleme için **MLX** ve "
        "**[OpenMedKit](swift/OpenMedKit)** kullanabilir. Model edinme ve "
        "kullanıcı tarafından yapılandırılan uzak entegrasyonlar ayrı ağ "
        "sınırlarıdır."
    ),
    "README.zh-CN.md": (
        "在受支持的 Apple 硬件上，所需制品就绪后，OpenMed 可使用 **MLX** 和 "
        "**[OpenMedKit](swift/OpenMedKit)** 进行本地处理。模型获取和用户配置"
        "的远程集成仍是独立的网络边界。"
    ),
}

LICENSE_BULLETS = {
    "README.md": (
        "- **SDK source**: released under the Apache-2.0 License; model and "
        "dataset terms vary."
    ),
    "README.ar.md": (
        "- **مصدر SDK**: منشور بموجب Apache-2.0 License؛ وتختلف شروط النماذج "
        "ومجموعات البيانات."
    ),
    "README.de.md": (
        "- **SDK-Quellcode**: unter der Apache-2.0 License veröffentlicht; "
        "Modell- und Datenbedingungen unterscheiden sich."
    ),
    "README.es.md": (
        "- **Código fuente del SDK**: publicado bajo Apache-2.0 License; los "
        "términos de modelos y datos varían."
    ),
    "README.fa.md": (
        "- **کد منبع SDK**: تحت Apache-2.0 License منتشر شده است؛ شرایط مدل و "
        "مجموعه‌داده متفاوت است."
    ),
    "README.fr.md": (
        "- **Code source du SDK** : publié sous Apache-2.0 License ; les "
        "conditions des modèles et jeux de données varient."
    ),
    "README.hi.md": (
        "- **SDK स्रोत**: Apache-2.0 License के अंतर्गत जारी; मॉडल और डेटासेट "
        "की शर्तें अलग-अलग होती हैं।"
    ),
    "README.it.md": (
        "- **Codice sorgente dell’SDK**: distribuito con Apache-2.0 License; i "
        "termini di modelli e set di dati variano."
    ),
    "README.ja.md": (
        "- **SDK ソース**：Apache-2.0 License で公開。モデルとデータセットの"
        "条件は異なります。"
    ),
    "README.nl.md": (
        "- **SDK-broncode**: uitgebracht onder Apache-2.0 License; voorwaarden "
        "van modellen en datasets verschillen."
    ),
    "README.pt.md": (
        "- **Código-fonte do SDK**: publicado sob Apache-2.0 License; os termos "
        "de modelos e conjuntos de dados variam."
    ),
    "README.sw.md": (
        "- **Msimbo chanzo wa SDK**: umetolewa chini ya Apache-2.0 License; "
        "masharti ya modeli na mikusanyiko ya data hutofautiana."
    ),
    "README.te.md": (
        "- **SDK సోర్స్**: Apache-2.0 License క్రింద విడుదల చేయబడింది; మోడల్ "
        "మరియు డేటాసెట్ నిబంధనలు మారుతాయి."
    ),
    "README.tr.md": (
        "- **SDK kaynak kodu**: Apache-2.0 License altında yayımlanır; model ve "
        "veri kümesi koşulları değişir."
    ),
    "README.zh-CN.md": (
        "- **SDK 源代码**：基于 Apache-2.0 License 发布；模型和数据集条款各不相同。"
    ),
}

DEMO_COPY = {
    "README.md": (
        "This iPhone example uses OpenMed's local runtime after the required "
        "model artifacts are available:"
    ),
    "README.hi.md": (
        "यह iPhone उदाहरण आवश्यक मॉडल आर्टिफ़ैक्ट उपलब्ध होने के बाद OpenMed "
        "के स्थानीय रनटाइम का उपयोग करता है:"
    ),
    "README.sw.md": (
        "Mfano huu wa iPhone hutumia runtime ya ndani ya OpenMed baada ya "
        "vipengee vya modeli vinavyohitajika kupatikana:"
    ),
    "README.zh-CN.md": (
        "此 iPhone 示例会在所需模型制品就绪后使用 OpenMed 的本地运行时："
    ),
}

DEMO_CAPTIONS = {
    "README.md": (
        '  <sub><b>On iPhone via <a href="swift/OpenMedKit">OpenMedKit</a></b>: '
        "scan a clinical note, de-identify it, and extract clinical signals "
        "with Apple MLX processing locally in this configuration.</sub>",
        "  <sub><b>Real-time PII de-identification</b>: in this configured "
        "local workflow, the Nemotron Privacy Filter redacts names, addresses, "
        "IDs, and billing data from a synthetic clinical discharge packet. "
        "<i>(All values shown are synthetic.)</i></sub>",
    ),
    "README.hi.md": (
        '  <sub><b><a href="swift/OpenMedKit">OpenMedKit</a> के माध्यम से '
        "iPhone पर</b>: इस कॉन्फ़िगरेशन में Apple MLX स्थानीय रूप से क्लिनिकल "
        "नोट को स्कैन, डी-आइडेंटिफ़ाई और संकेतों को निकालता है।</sub>",
        "  <sub><b>रीयल-टाइम PII डी-आइडेंटिफिकेशन</b>: इस कॉन्फ़िगर किए गए "
        "स्थानीय वर्कफ़्लो में Nemotron Privacy Filter सिंथेटिक क्लिनिकल "
        "डिस्चार्ज पैकेट से नाम, पते, ID और बिलिंग डेटा छिपाता है। "
        "<i>(दिखाए गए सभी मान सिंथेटिक हैं।)</i></sub>",
    ),
    "README.sw.md": (
        '  <sub><b>Kwenye iPhone kupitia <a href="swift/OpenMedKit">'
        "OpenMedKit</a></b>: katika usanidi huu Apple MLX huchakata dokezo la "
        "kliniki ndani ya kifaa ili kuondoa utambulisho na kutoa ishara.</sub>",
        "  <sub><b>Uondoaji wa utambulisho wa PII kwa wakati halisi</b>: katika "
        "mtiririko huu wa ndani uliosanidiwa, Nemotron Privacy Filter huficha "
        "majina, anwani, vitambulisho na data ya malipo kutoka hati sintetiki. "
        "<i>(Thamani zote zinazoonekana ni za kutengenezwa.)</i></sub>",
    ),
    "README.zh-CN.md": (
        '  <sub><b>通过 <a href="swift/OpenMedKit">OpenMedKit</a> 在 iPhone '
        "上运行</b>：在此配置中，由 Apple MLX 在本地扫描临床记录、完成"
        "去标识化并抽取临床信号。</sub>",
        "  <sub><b>实时 PII 去标识化</b>：在此配置的本地工作流中，Nemotron "
        "Privacy Filter 会对合成临床出院记录中的姓名、地址、证件号和账单数据"
        "进行脱敏。<i>（图中所有数值均为合成数据。）</i></sub>",
    ),
}

REPLACEMENTS: dict[str, tuple[tuple[str, str], ...]] = {
    "README.md": (
        (
            "removes 55+ PHI types",
            "applies configurable PII entity detection and de-identification",
        ),
        ("The same 2,000+ open models", "Curated open model routes"),
        (
            "| Specialized medical models            |          2,000+          |",
            "| Specialized medical models            |     Curated catalog      |",
        ),
        (
            "| Model-backed PII languages            |            29            |",
            "| Model-backed PII languages            |            33            |",
        ),
        (
            "- **Specialized models**: 2,000+ curated biomedical & clinical "
            "models, many outperforming proprietary stacks.",
            "- **Specialized models**: a curated biomedical and clinical model "
            "catalog; validate each model for your use case.",
        ),
        ("100% on-device", "local-first"),
        (
            "**600+ PII checkpoints**",
            "**the registry-backed PII model catalog**",
        ),
    ),
    "README.ar.md": (
        (
            "ويزيل أكثر من 55 نوعًا من معرّفات الهوية الشخصية (PHI)",
            "ويطبّق كشف كيانات PII وإزالة الهوية بصورة قابلة للتهيئة",
        ),
        (
            "النماذج المفتوحة نفسها، وعددها أكثر من 2,000 نموذج",
            "مسارات نماذج مفتوحة ومنتقاة",
        ),
        (
            "| نماذج طبية متخصصة                      |          2,000+          |",
            "| نماذج طبية متخصصة                      |       فهرس منتقى          |",
        ),
        (
            "- **نماذج متخصصة**: أكثر من 2,000 نموذج طبي حيوي وسريري منتقى، "
            "يتفوق كثير منها على الحلول الاحتكارية.",
            "- **نماذج متخصصة**: فهرس منتقى من النماذج الطبية الحيوية والسريرية؛ "
            "تحقّق من كل نموذج وفق حالة الاستخدام.",
        ),
        ("100% على الجهاز", "محلي أولاً"),
        ("**600+ نقطة تحقق PII**", "**فهرس نماذج PII المسجّل**"),
    ),
    "README.de.md": (
        (
            "entfernt über 55 PHI-Typen",
            "wendet konfigurierbare PII-Erkennung und De-Identifikation an",
        ),
        ("Dieselben 2.000+ offenen Modelle", "Ausgewählte offene Modelle"),
        (
            "| Spezialisierte medizinische Modelle   |          2.000+          |",
            "| Spezialisierte medizinische Modelle   |    Kuratierter Katalog    |",
        ),
        (
            "- **Spezialisierte Modelle**: über 2.000 kuratierte biomedizinische "
            "und klinische Modelle, von denen viele proprietäre Lösungen "
            "übertreffen.",
            "- **Spezialisierte Modelle**: ein kuratierter biomedizinischer und "
            "klinischer Modellkatalog; jedes Modell für den Anwendungsfall prüfen.",
        ),
        ("100 % auf dem Gerät", "Local-First"),
        ("**600+ PII-Checkpoints**", "**der registrierte PII-Modellkatalog**"),
        (
            "**HIPAA-konforme De-Identifikation**",
            "**HIPAA-bewusste De-Identifikation**",
        ),
    ),
    "README.es.md": (
        (
            "elimina más de 55 tipos de PHI",
            "aplica detección configurable de entidades PII y desidentificación",
        ),
        (
            "Los mismos más de 2.000 modelos abiertos",
            "Las rutas seleccionadas de modelos abiertos",
        ),
        (
            "| Modelos médicos especializados        |          2.000+          |",
            "| Modelos médicos especializados        |     Catálogo seleccionado |",
        ),
        (
            "- **Modelos especializados**: más de 2.000 modelos biomédicos y "
            "clínicos seleccionados, muchos de ellos superan a las soluciones "
            "propietarias.",
            "- **Modelos especializados**: un catálogo biomédico y clínico "
            "seleccionado; valide cada modelo para su caso de uso.",
        ),
        ("100% en el dispositivo", "local primero"),
        ("**600+ checkpoints de PII**", "**el catálogo registrado de modelos PII**"),
    ),
    "README.fa.md": (
        (
            "بیش از 55 نوع اطلاعاتِ سلامتِ محافظت‌شده (PHI) را",
            "تشخیصِ قابل‌پیکربندیِ موجودیت‌های PII و حذف هویت را",
        ),
        (
            "همان بیش از ۲٬۰۰۰ مدلِ متن‌باز",
            "مسیرهای گزیدهٔ مدل‌های باز",
        ),
        (
            "| مدل‌های تخصصی پزشکی                    |          2,000+          |",
            "| مدل‌های تخصصی پزشکی                    |       فهرست گزیده          |",
        ),
        (
            "- **مدل‌های تخصصی**: بیش از 2,000 مدلِ زیست‌پزشکی و بالینیِ "
            "گزینش‌شده که بسیاری از آن‌ها از راهکارهای انحصاری بهتر عمل می‌کنند.",
            "- **مدل‌های تخصصی**: فهرستی گزیده از مدل‌های زیست‌پزشکی و بالینی؛ "
            "هر مدل را برای کاربرد خود اعتبارسنجی کنید.",
        ),
        ("100٪ روی دستگاه", "محلی‌محور"),
        ("**600+ نقطه‌بازرسیِ PII**", "**فهرست ثبت‌شدهٔ مدل‌های PII**"),
    ),
    "README.fr.md": (
        (
            "supprime plus de 55 types de PHI",
            "applique une détection configurable des entités PII et la "
            "dé-identification",
        ),
        (
            "Les mêmes 2 000+ modèles ouverts",
            "Des routes de modèles ouverts sélectionnées",
        ),
        (
            "| Modèles médicaux spécialisés          |          2 000+          |",
            "| Modèles médicaux spécialisés          |   Catalogue sélectionné   |",
        ),
        (
            "- **Modèles spécialisés** : plus de 2 000 modèles biomédicaux et "
            "cliniques sélectionnés, dont beaucoup surpassent les solutions "
            "propriétaires.",
            "- **Modèles spécialisés** : un catalogue biomédical et clinique "
            "sélectionné ; validez chaque modèle pour votre usage.",
        ),
        ("100 % sur l'appareil", "local en priorité"),
        ("**600+ checkpoints PII**", "**le catalogue enregistré de modèles PII**"),
    ),
    "README.hi.md": (
        (
            "55+ PHI प्रकार हटाता है",
            "कॉन्फ़िगर किए जा सकने वाले PII एंटिटी की पहचान और डी-आइडेंटिफ़िकेशन लागू करता है",
        ),
        ("वही 2,000+ ओपन मॉडल", "चुनिंदा ओपन मॉडल रूट"),
        (
            "| विशेष चिकित्सा मॉडल                    |          2,000+          |",
            "| विशेष चिकित्सा मॉडल                    |       चुनिंदा कैटलॉग       |",
        ),
        (
            "| मॉडल-समर्थित PII भाषाएँ               |            29            |",
            "| मॉडल-समर्थित PII भाषाएँ               |            33            |",
        ),
        (
            "- **विशेष मॉडल**: 2,000+ सावधानी से चुने गए बायोमेडिकल और "
            "क्लिनिकल मॉडल, जिनमें से कई प्रोप्राइटरी समाधानों से बेहतर प्रदर्शन "
            "करते हैं।",
            "- **विशेष मॉडल**: चुना हुआ बायोमेडिकल और क्लिनिकल मॉडल कैटलॉग; "
            "अपने उपयोग के लिए हर मॉडल को सत्यापित करें।",
        ),
        ("100% डिवाइस पर", "स्थानीय-प्रथम"),
        ("100% on-device", "local-first"),
        ("**600+ PII checkpoints**", "**रजिस्ट्री-आधारित PII मॉडल कैटलॉग**"),
    ),
    "README.it.md": (
        (
            "rimuove oltre 55 tipi di PHI",
            "applica il rilevamento configurabile delle entità PII e la "
            "de-identificazione",
        ),
        (
            "Gli stessi oltre 2.000 modelli aperti",
            "Le route selezionate di modelli aperti",
        ),
        (
            "| Modelli medici specializzati          |          2.000+          |",
            "| Modelli medici specializzati          |    Catalogo selezionato   |",
        ),
        (
            "- **Modelli specializzati**: oltre 2.000 modelli biomedici e clinici "
            "selezionati, molti dei quali superano le soluzioni proprietarie.",
            "- **Modelli specializzati**: un catalogo biomedico e clinico "
            "selezionato; valida ogni modello per il tuo caso d'uso.",
        ),
        ("100% sul dispositivo", "locale per impostazione"),
        ("**600+ checkpoint PII**", "**il catalogo registrato dei modelli PII**"),
    ),
    "README.ja.md": (
        (
            "55+ 種類の PHI を",
            "設定可能な PII エンティティ検出と非識別化を",
        ),
        (
            "同じ 2,000+ のオープンモデル",
            "厳選されたオープンモデルルート",
        ),
        (
            "| 専門医療モデル                        |          2,000+          |",
            "| 専門医療モデル                        |       厳選カタログ         |",
        ),
        (
            "- **専門モデル**：厳選された 2,000 以上の生物医学・臨床モデル。"
            "その多くは商用の専有スタックを上回ります。",
            "- **専門モデル**：厳選された生物医学・臨床モデルカタログ。"
            "用途ごとに各モデルを検証してください。",
        ),
        ("100% オンデバイス", "ローカルファースト"),
        (
            "**600+ 個の PII チェックポイント**",
            "**登録済み PII モデルカタログ**",
        ),
    ),
    "README.nl.md": (
        (
            "verwijdert 55+ PHI-typen",
            "past configureerbare PII-entiteitsdetectie en de-identificatie toe",
        ),
        ("Dezelfde 2.000+ open modellen", "Geselecteerde open modelroutes"),
        (
            "| Gespecialiseerde medische modellen    |          2.000+          |",
            "| Gespecialiseerde medische modellen    |  Geselecteerde catalogus  |",
        ),
        (
            "- **Gespecialiseerde modellen**: meer dan 2.000 zorgvuldig "
            "geselecteerde biomedische en klinische modellen, waarvan vele "
            "propriëtaire oplossingen overtreffen.",
            "- **Gespecialiseerde modellen**: een geselecteerde biomedische en "
            "klinische modelcatalogus; valideer elk model voor je toepassing.",
        ),
        ("100% op het apparaat", "lokaal eerst"),
        (
            "**600+ PII-checkpoints**",
            "**de geregistreerde PII-modelcatalogus**",
        ),
    ),
    "README.pt.md": (
        (
            "remove mais de 55 tipos de PHI",
            "aplica detecção configurável de entidades PII e desidentificação",
        ),
        (
            "Os mesmos 2.000+ modelos abertos",
            "Rotas selecionadas de modelos abertos",
        ),
        (
            "| Modelos médicos especializados        |          2.000+          |",
            "| Modelos médicos especializados        |    Catálogo selecionado   |",
        ),
        (
            "- **Modelos especializados**: mais de 2.000 modelos biomédicos e "
            "clínicos selecionados, muitos superando soluções proprietárias.",
            "- **Modelos especializados**: um catálogo biomédico e clínico "
            "selecionado; valide cada modelo para o seu caso de uso.",
        ),
        ("100% no dispositivo", "local primeiro"),
        (
            "**600+ checkpoints de PII**",
            "**o catálogo registrado de modelos PII**",
        ),
    ),
    "README.sw.md": (
        (
            "kuondoa aina 55+ za PHI",
            "kutumia utambuzi wa huluki za PII unaoweza kusanidiwa na uondoaji "
            "utambulisho",
        ),
        (
            "Modeli huria\n2,000+",
            "Njia zilizochaguliwa za modeli huria",
        ),
        (
            "| Modeli maalumu za matibabu           |            2,000+",
            "| Modeli maalumu za matibabu           |      Katalogi iliyochaguliwa",
        ),
        (
            "| Lugha za PII zinazotumia modeli      |              29",
            "| Lugha za PII zinazotumia modeli      |              33",
        ),
        (
            "- **Modeli maalumu**: modeli 2,000+ za biomedicine na kliniki "
            "zilizochaguliwa.",
            "- **Modeli maalumu**: katalogi iliyochaguliwa ya modeli za "
            "biomedicine na kliniki; thibitisha kila modeli kwa matumizi yako.",
        ),
        ("100% kwenye kifaa", "matumizi ya ndani kwanza"),
        ("600+ za PII", "katalogi iliyosajiliwa ya modeli za PII"),
    ),
    "README.te.md": (
        (
            "55+ PHI రకాలను",
            "కాన్ఫిగర్ చేయగల PII ఎంటిటీ గుర్తింపు మరియు డీ-ఐడెంటిఫికేషన్‌ను",
        ),
        ("అదే 2,000+ ఓపెన్ మోడల్‌లు", "ఎంపిక చేసిన ఓపెన్ మోడల్ మార్గాలు"),
        (
            "| ప్రత్యేక వైద్య మోడల్‌లు                  |          2,000+          |",
            "| ప్రత్యేక వైద్య మోడల్‌లు                  |       ఎంపిక చేసిన కేటలాగ్    |",
        ),
        (
            "- **ప్రత్యేక మోడల్‌లు**: జాగ్రత్తగా ఎంపిక చేసిన 2,000+ బయోమెడికల్ "
            "& క్లినికల్ మోడల్‌లు, వీటిలో చాలావి యాజమాన్య పరిష్కారాలను "
            "అధిగమిస్తాయి.",
            "- **ప్రత్యేక మోడల్‌లు**: ఎంపిక చేసిన బయోమెడికల్ మరియు క్లినికల్ "
            "మోడల్ కేటలాగ్; మీ వినియోగానికి ప్రతి మోడల్‌ను ధృవీకరించండి.",
        ),
        ("100% పరికరంలో", "స్థానిక అమలుకు ప్రాధాన్యం"),
        ("**600+ PII చెక్‌పాయింట్‌లు**", "**నమోదిత PII మోడల్ కేటలాగ్**"),
    ),
    "README.tr.md": (
        (
            "55+ PHI türünü",
            "yapılandırılabilir PII varlık tespiti ve kimliksizleştirmeyi",
        ),
        ("Aynı 2.000+ açık model", "Seçilmiş açık model yolları"),
        (
            "| Özelleşmiş tıbbi modeller             |          2.000+          |",
            "| Özelleşmiş tıbbi modeller             |      Seçilmiş katalog     |",
        ),
        (
            "- **Özelleşmiş modeller**: 2.000'den fazla özenle seçilmiş "
            "biyomedikal ve klinik model; birçoğu tescilli çözümleri geride "
            "bırakır.",
            "- **Özelleşmiş modeller**: seçilmiş biyomedikal ve klinik model "
            "kataloğu; kullanımınız için her modeli doğrulayın.",
        ),
        ("%100 cihazda", "yerel öncelikli"),
        (
            "**600+ PII denetim noktası**",
            "**kayıtlı PII model kataloğu**",
        ),
        (
            "**HIPAA uyumlu kimliksizleştirme**",
            "**HIPAA konusunda bilinçli kimliksizleştirme**",
        ),
    ),
    "README.zh-CN.md": (
        (
            "彻底移除 55+ 种 PHI 类型",
            "执行可配置的 PII 实体检测与去标识化",
        ),
        (
            "同一套 2,000+ 个开源模型",
            "精选的开放模型路由",
        ),
        (
            "| 专业医疗模型                           |          2,000+          |",
            "| 专业医疗模型                           |         精选目录          |",
        ),
        (
            "| 由模型支持的 PII 语言                  |            29            |",
            "| 由模型支持的 PII 语言                  |            33            |",
        ),
        (
            "- **专业模型**：2,000+ 个精选的生物医学与临床模型，其中许多性能"
            "超越商业专有方案。",
            "- **专业模型**：精选的生物医学与临床模型目录；请针对你的用例验证"
            "每个模型。",
        ),
        ("100% 设备本地运行", "本地优先"),
        ("100% 本地", "本地优先"),
        ("**600+ 个 PII 检查点**", "**已登记的 PII 模型目录**"),
    ),
}

LEGACY_CLAIM_PATTERNS = (
    re.compile(r"(?:2[, .]000|۲٬۰۰۰)(?:\+| 以上|'den fazla)?"),
    re.compile(r"600\+"),
    re.compile(r"55\+|(?:plus de|über|más de|mais de|oltre|أكثر من|بیش از) 55"),
    re.compile(r"Models-2%2C000\+"),
)


def _claim_numbers() -> dict[str, int]:
    registry = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    claims = registry["claims"]
    pointers = {
        "supported": "supported_pii_languages",
        "model_backed": "model_backed_pii_languages",
    }
    values: dict[str, int] = {}
    for placeholder, claim_name in pointers.items():
        claim = claims[claim_name]
        if claim["status"] != "verified" or not isinstance(claim["value"], int):
            raise RuntimeError(f"{claim_name} must be a verified integer claim")
        values[placeholder] = claim["value"]
    return values


def _claim_copy(template: str) -> str:
    return template.format(**_claim_numbers())


def _star_snapshot() -> str:
    registry = json.loads(CLAIMS_PATH.read_text(encoding="utf-8"))
    claim = registry["claims"]["github_stars_snapshot"]
    if (
        claim["status"] != "verified"
        or not isinstance(claim["value"], int)
        or not claim["display"]
        or not claim["as_of"]
    ):
        raise RuntimeError("github_stars_snapshot must be a dated verified claim")
    captured = claim["as_of"].split("-")
    date_label = (
        f"{int(captured[2])} "
        f"{('Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec')[int(captured[1]) - 1]} "
        f"{captured[0]}"
    )
    return (
        f"[{claim['display']} · {date_label} snapshot]"
        "(https://github.com/maziyarpanahi/openmed/stargazers)"
    )


def _capability_bullets(filename: str) -> str:
    return CAPABILITY_BULLETS[filename] + "\n" + LICENSE_BULLETS[filename]


def _banner(filename: str) -> str:
    return (
        '<img src="docs/brand/openmed-readme-banner.png" '
        f'alt="{ALT_TEXT[filename]}" width="1280" />'
    )


def transform(filename: str, text: str) -> str:
    """Return the governed README form for one locale."""

    text, count = re.subn(
        r'<img src="docs/brand/(?:openmed-mascot-lockup|'
        r'openmed-readme-banner)\.png" alt="[^"]*" width="(?:400|1280)" />',
        _banner(filename),
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"{filename}: expected exactly one hero brand image")

    text, count = re.subn(
        r"<p><b>.*?</p>",
        INTRO_COPY[filename],
        text,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"{filename}: expected exactly one introductory claim")

    text = re.sub(
        r'<a href="https://trendshift\.io/.*?</a>\n\n',
        "",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if RESOURCE_LINKS not in text:
        text, count = re.subn(
            r'(?s)(?:<a href="https://trendshift\.io/.*?</a>\n\n)?'
            r'<p>\n\s*<a href="https://pypi\.org/project/openmed/">.*?</p>\n\n'
            r"<p>\n\s*<a href=\"swift/OpenMedKit\">.*?</p>",
            RESOURCE_LINKS,
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(f"{filename}: expected the two legacy badge rows")

    for old, new in REPLACEMENTS[filename]:
        text = text.replace(old, new)

    text, count = re.subn(
        r"(?s)(# DRUG[^\n]*\n```\n\n).*?(\n\n---)",
        lambda match: match.group(1) + EXAMPLE_COPY[filename] + match.group(2),
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"{filename}: expected exactly one quick example")

    text, count = re.subn(
        r"(?ms)(^## [^\n]*(?:Swift|MLX|iOS)[^\n]*\n\n)"
        r".*?(?=\n\n```swift$)",
        lambda match: match.group(1) + APPLE_COPY[filename],
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"{filename}: expected exactly one Apple SDK section")

    if LICENSE_BULLETS[filename] not in text:
        text, count = re.subn(
            r"(?m)^- \*\*[^\n]*\*\* ?[：:] ?"
            r"[^\n]*Apache-2\.0 SDK[^\n]*$",
            LICENSE_BULLETS[filename],
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                f"{filename}: expected exactly one legacy lock-in bullet"
            )

    if filename in DEMO_COPY:
        text, count = re.subn(
            r"(?ms)(^## [^\n]+\n\n).*?"
            r'(?=\n\n<div align="center">\n'
            r'  <img src="docs/brand/openmed-ios-scan\.png")',
            lambda match: match.group(1) + DEMO_COPY[filename],
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                f"{filename}: expected exactly one iPhone demo introduction"
            )
        for asset, caption in zip(
            ("openmed-ios-scan.png", "openmed-pii-demo.gif"),
            DEMO_CAPTIONS[filename],
            strict=True,
        ):
            text, count = re.subn(
                rf'(?s)(<img src="docs/brand/{re.escape(asset)}"'
                rf"[^\n]*\n  <br/>\n)  <sub>.*?</sub>",
                lambda match, replacement=caption: match.group(1) + replacement,
                text,
                count=1,
            )
            if count != 1:
                raise RuntimeError(
                    f"{filename}: expected exactly one caption for {asset}"
                )

    if DEPLOYMENT_TABLES[filename] not in text:
        text, count = re.subn(
            r"(?m)^\|[^\n]*\*\*OpenMed\*\*[^\n]*\n"
            r"\|[^\n]*\n(?:\|[^\n]*\n)+",
            DEPLOYMENT_TABLES[filename] + "\n",
            text,
            count=1,
        )
        if count != 1:
            raise RuntimeError(
                f"{filename}: expected exactly one provider comparison table"
            )

    text, count = re.subn(
        rf"(?s)({re.escape(DEPLOYMENT_TABLES[filename])}\n\n).*?(?=\n\n---)",
        lambda match: match.group(1) + _capability_bullets(filename),
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"{filename}: expected one deployment capability list")

    lines = text.splitlines()
    pii_heading_indexes = [
        index
        for index, line in enumerate(lines)
        if line.startswith("## ") and "PII" in line
    ]
    if len(pii_heading_indexes) != 2:
        raise RuntimeError(
            f"{filename}: expected two PII headings, found {len(pii_heading_indexes)}"
        )
    lines[pii_heading_indexes[0]] = PRIVACY_HEADINGS[filename]
    lines[pii_heading_indexes[1]] = _claim_copy(LANGUAGE_HEADINGS[filename])
    text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    text, count = re.subn(
        r"(?m)^.*\[.*Apache-2\.0.*\]\(LICENSE\).*"
        r"(?:\n(?!\n|##).*)?",
        LICENSE_LINES[filename],
        text,
        count=1,
    )
    if count != 1:
        raise RuntimeError(f"{filename}: expected exactly one legal license line")

    lines = text.splitlines()
    claim_line_indexes = [
        index
        for index, line in enumerate(lines)
        if line.startswith("  <b>")
        and "&nbsp;·&nbsp;" in line
        and (
            "Apache-2.0" in line
            or "2,000+" in line
            or "2.000+" in line
            or "2 000+" in line
        )
    ]
    if len(claim_line_indexes) != 1:
        raise RuntimeError(
            f"{filename}: expected one hero claim line, found {len(claim_line_indexes)}"
        )
    lines[claim_line_indexes[0]] = _claim_copy(HERO_CLAIMS[filename])
    text = "\n".join(lines) + ("\n" if text.endswith("\n") else "")

    text = text.replace("Models-2%2C000+", "Models-Catalog")
    text = text.replace(
        "badge/License-Apache_2.0",
        "badge/SDK_License-Apache_2.0",
    )
    text = text.replace(
        '<img alt="License" src="https://img.shields.io/badge/SDK_License',
        '<img alt="SDK License" src="https://img.shields.io/badge/SDK_License',
    )
    text = text.replace(
        '<img alt="Leseni" src="https://img.shields.io/badge/SDK_License',
        '<img alt="Leseni ya SDK" src="https://img.shields.io/badge/SDK_License',
    )
    text, count = re.subn(
        r"(?m)^- \*\*HIPAA\*\*\s*[：:].*$",
        HIPAA_BOUNDARY_LINES[filename],
        text,
        count=1,
    )
    if count != 1 and HIPAA_BOUNDARY_LINES[filename] not in text:
        raise RuntimeError(f"{filename}: expected one legacy HIPAA bullet")
    text = text.replace(
        "- **One model name, every platform**: MLX model names automatically "
        "fall back to the matching PyTorch checkpoint on non-Apple hardware.",
        "- **Portable model naming where supported**: an MLX model name can "
        "fall back to a matching PyTorch checkpoint when that mapping and "
        "artifact are available on non-Apple hardware.",
    )
    text = text.replace(
        "ship one model name, run anywhere",
        "reuse one model name where a matching backend artifact is available",
    )
    snapshot = _star_snapshot()
    if snapshot not in text:
        html_count = 0
        markdown_count = 0
        text, html_count = re.subn(
            r'(?s)<a href="https://star-history\.com/[^"]*">\s*'
            r'<img src="https://api\.star-history\.com/[^"]*"[^>]*\s*/>\s*</a>',
            snapshot,
            text,
            count=1,
        )
        text, markdown_count = re.subn(
            r"(?m)^\[!\[[^\n]*\]\(https://api\.star-history\.com/[^\n]*$",
            snapshot,
            text,
            count=1,
        )
        if html_count + markdown_count != 1:
            raise RuntimeError(f"{filename}: expected one remote star-history image")
    return text


def _validate(filename: str, text: str) -> list[str]:
    errors: list[str] = []
    if text.count("docs/brand/openmed-readme-banner.png") != 1:
        errors.append("canonical banner must appear exactly once")
    if "openmed-mascot-lockup.png" in text:
        errors.append("old mascot lockup remains")
    if ALT_TEXT[filename] not in text:
        errors.append("localized canonical alt text is missing")
    if text.count(INTRO_COPY[filename]) != 1:
        errors.append("governed local-runtime boundary must appear exactly once")
    if text.count(RESOURCE_LINKS) != 1:
        errors.append("governed static resource links must appear exactly once")
    if text.count(DEPLOYMENT_TABLES[filename]) != 1:
        errors.append("provider-neutral deployment table must appear exactly once")
    if text.count(_capability_bullets(filename)) != 1:
        errors.append("qualified capability list must appear exactly once")
    if text.count(HIPAA_BOUNDARY_LINES[filename]) != 1:
        errors.append("qualified HIPAA boundary must appear exactly once")
    if text.count(_claim_copy(HERO_CLAIMS[filename])) != 1:
        errors.append("registry-backed hero claim must appear exactly once")
    if text.count(_claim_copy(LANGUAGE_HEADINGS[filename])) != 1:
        errors.append("registry-backed language heading must appear exactly once")
    if text.count(PRIVACY_HEADINGS[filename]) != 1:
        errors.append("privacy heading must appear exactly once")
    if text.count(LICENSE_LINES[filename]) != 1:
        errors.append("SDK source license line must appear exactly once")
    if text.count(EXAMPLE_COPY[filename]) != 1:
        errors.append("governed quick-example boundary must appear exactly once")
    if text.count(APPLE_COPY[filename]) != 1:
        errors.append("governed Apple runtime boundary must appear exactly once")
    if text.count(LICENSE_BULLETS[filename]) != 1:
        errors.append("governed SDK source bullet must appear exactly once")
    if filename in DEMO_COPY:
        if text.count(DEMO_COPY[filename]) != 1:
            errors.append("governed iPhone demo boundary must appear exactly once")
        for caption in DEMO_CAPTIONS[filename]:
            if text.count(caption) != 1:
                errors.append("governed demo caption must appear exactly once")
    if text.count("[Apache-2.0 License](LICENSE)") != 1:
        errors.append("canonical Apache-2.0 legal link must appear exactly once")
    if re.search(r"(?m)^\|[^\n]*\*\*OpenMed\*\*[^\n]*$", text):
        errors.append("provider comparison table remains")
    if re.search(r"(?m)^## [^\n]*PII[^\n]*12[^\n]*$", text):
        errors.append("stale 12-language PII heading remains")
    if "trendshift.io" in text or "img.shields.io" in text:
        errors.append("remote dynamic badge remains")
    if "api.star-history.com" in text:
        errors.append("remote dynamic star-history image remains")
    if text.count(_star_snapshot()) != 1:
        errors.append("registry-backed GitHub stars snapshot must appear once")
    if "%F0%9F%A4%97" in text:
        errors.append("emoji-encoded brand chrome remains")
    if re.search(
        r"(?i)\b(?:runs everywhere|one-line deployment|every platform|run anywhere)\b",
        text,
    ):
        errors.append("absolute deployment wording remains")
    for pattern in LEGACY_CLAIM_PATTERNS:
        if pattern.search(text):
            errors.append(f"legacy claim matches {pattern.pattern!r}")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if (
            "Apache-2.0" in line
            and "SDK" not in line
            and line != LICENSE_LINES[filename]
        ):
            errors.append(f"line {line_number}: Apache-2.0 is not SDK-scoped")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    failures: list[str] = []
    for filename in README_FILES:
        path = REPO_ROOT / filename
        current = path.read_text(encoding="utf-8")
        expected = transform(filename, current)
        if args.write and expected != current:
            path.write_text(expected, encoding="utf-8")
            print(f"updated {filename}")
            current = expected
        elif args.check and expected != current:
            failures.append(f"{filename}: brand/claim wording is stale")
        failures.extend(
            f"{filename}: {error}" for error in _validate(filename, current)
        )

    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("all 15 READMEs use governed brand art and claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
