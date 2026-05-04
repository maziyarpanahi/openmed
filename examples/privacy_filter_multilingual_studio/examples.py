"""Pre-defined examples for the multilingual side-by-side comparison studio.

One language tab per supported language (16 total). Clicking a tab loads its
example into the editor; the user can also cycle through extra examples per
language via the chip strip.

All names, IDs, and contact details are synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StudioExample:
    id: str
    title: str
    blurb: str
    text: str

    def to_public_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "blurb": self.blurb, "text": self.text}


# Display order for the language tab strip.
LANGUAGE_META: list[dict[str, str]] = [
    {"code": "en", "label": "English",   "native": "English"},
    {"code": "es", "label": "Spanish",   "native": "Español"},
    {"code": "fr", "label": "French",    "native": "Français"},
    {"code": "de", "label": "German",    "native": "Deutsch"},
    {"code": "it", "label": "Italian",   "native": "Italiano"},
    {"code": "nl", "label": "Dutch",     "native": "Nederlands"},
    {"code": "pt", "label": "Portuguese","native": "Português"},
    {"code": "tr", "label": "Turkish",   "native": "Türkçe"},
    {"code": "ar", "label": "Arabic",    "native": "العربية"},
    {"code": "hi", "label": "Hindi",     "native": "हिन्दी"},
    {"code": "bn", "label": "Bengali",   "native": "বাংলা"},
    {"code": "te", "label": "Telugu",    "native": "తెలుగు"},
    {"code": "vi", "label": "Vietnamese","native": "Tiếng Việt"},
    {"code": "zh", "label": "Chinese",   "native": "中文"},
    {"code": "ja", "label": "Japanese",  "native": "日本語"},
    {"code": "ko", "label": "Korean",    "native": "한국어"},
]


LANGUAGE_EXAMPLES: dict[str, tuple[StudioExample, ...]] = {
    "en": (
        StudioExample(
            id="en-clinical",
            title="Clinical encounter",
            blurb="Discharge summary with names, MRN, DOB, address, contact, insurance.",
            text=(
                "Patient Sarah Johnson (DOB 03/15/1985), MRN 4872910, was discharged on "
                "April 22 2026 after a three-day admission for community-acquired pneumonia. "
                "Follow-up scheduled with Dr. Michael Chen on May 6 2026 at Cedars Medical Center, "
                "8700 Beverly Blvd, Los Angeles CA 90048. Reach the patient at (415) 555-7012 "
                "or sarah.johnson@example.com. Insurance member ID BLU-2284-118-A."
            ),
        ),
        StudioExample(
            id="en-business",
            title="Business onboarding",
            blurb="Enterprise mix: emails, phone, address, company, financial.",
            text=(
                "Welcome to Harbor Logistics. Your contact is Linda Park (linda.park@harborlog.com, "
                "+1 312-555-0144). Mail packages to 1428 W Roosevelt Rd, Suite 6B, Chicago IL 60608. "
                "Direct deposit to account 902418737, routing 021000089, BIC HARBLOGUS. "
                "Employee ID EMP-447128 — please confirm by November 14 2026."
            ),
        ),
    ),
    "es": (
        StudioExample(
            id="es-clinical",
            title="Encuentro clínico",
            blurb="Informe de alta con nombre, fecha de nacimiento, dirección y contacto.",
            text=(
                "El paciente Juan García (fecha de nacimiento 15/03/1985) fue dado de alta el "
                "22/04/2026 tras un ingreso de tres días por neumonía. Cita de seguimiento "
                "con la Dra. María López el 6 de mayo de 2026 en el Hospital Universitario "
                "Vall d'Hebron, Passeig de la Vall d'Hebron 119, 08035 Barcelona. "
                "Teléfono: +34 93 489 30 00, correo: juan.garcia@correo.es."
            ),
        ),
        StudioExample(
            id="es-business",
            title="Alta empresarial",
            blurb="Datos de contacto, dirección, tarjeta y cuenta bancaria.",
            text=(
                "Estimada Carmen Ruiz, su cuenta de cliente CL-99821 ha sido activada. "
                "Para cualquier consulta, contáctenos en soporte@empresa.es o al "
                "+34 91 555 01 23. La dirección de facturación es Calle Mayor 25, 28013 Madrid. "
                "Tarjeta terminada en 4111 1111 1111 7392, IBAN ES91 2100 0418 4502 0005 1332."
            ),
        ),
    ),
    "fr": (
        StudioExample(
            id="fr-clinical",
            title="Consultation clinique",
            blurb="Compte rendu avec patient, médecin, adresse et coordonnées.",
            text=(
                "Madame Marie Dupont, née le 15/03/1985, a été hospitalisée du 19 au 22 avril 2026 "
                "à l'Hôpital Saint-Louis, 1 avenue Claude-Vellefaux, 75010 Paris, pour une "
                "pneumonie communautaire. Le suivi est prévu avec le Dr Jean-Luc Moreau le "
                "6 mai 2026. Pour toute question : marie.dupont@exemple.fr ou +33 1 42 49 49 49. "
                "Numéro de sécurité sociale : 2 85 03 75 116 002 89."
            ),
        ),
        StudioExample(
            id="fr-business",
            title="Inscription entreprise",
            blurb="Coordonnées professionnelles, IBAN, code BIC.",
            text=(
                "Cher Pierre Lefèvre, votre compte client n° 22184 chez Atelier du Marais est "
                "actif. Adresse de livraison : 12 rue de la Paix, 75002 Paris. Téléphone : "
                "+33 1 47 03 56 78, courriel : pierre.lefevre@atelier-marais.fr. "
                "Pour le prélèvement automatique, IBAN FR76 3000 1007 9412 3456 7890 185, "
                "BIC BNPAFRPPXXX."
            ),
        ),
    ),
    "de": (
        StudioExample(
            id="de-clinical",
            title="Arztbericht",
            blurb="Entlassungsbrief mit Name, Geburtsdatum, Adresse und Kontakt.",
            text=(
                "Herr Hans Müller (geboren am 15.03.1985, Versichertennummer A123456789) wurde "
                "am 22.04.2026 nach dreitägigem Aufenthalt entlassen. Die Nachsorge erfolgt am "
                "06.05.2026 durch Dr. med. Karin Schmidt, Charité - Universitätsmedizin Berlin, "
                "Charitéplatz 1, 10117 Berlin. Erreichbar unter +49 30 450 50 oder "
                "hans.mueller@beispiel.de."
            ),
        ),
        StudioExample(
            id="de-business",
            title="Geschäftsanbahnung",
            blurb="Geschäftliche Kontaktdaten, IBAN und Steuernummer.",
            text=(
                "Sehr geehrte Frau Anna Becker, Ihr Kundenkonto bei der Bayrischen Spedition GmbH "
                "wurde unter der Nummer KD-44219 angelegt. Kontaktieren Sie uns per E-Mail "
                "anna.becker@bayspedition.de oder telefonisch unter +49 89 12345678. "
                "Lieferadresse: Maximilianstraße 12, 80539 München. IBAN DE89 3704 0044 0532 "
                "0130 00, USt-IdNr. DE123456789."
            ),
        ),
    ),
    "it": (
        StudioExample(
            id="it-clinical",
            title="Visita clinica",
            blurb="Lettera di dimissione con dati anagrafici e di contatto.",
            text=(
                "Il signor Mario Rossi (nato il 15/03/1985, codice fiscale RSSMRA85C15H501Z) "
                "è stato dimesso il 22/04/2026 dopo un ricovero di tre giorni. Il follow-up "
                "è previsto il 06/05/2026 con la dottoressa Giulia Bianchi presso Ospedale "
                "San Raffaele, Via Olgettina 60, 20132 Milano. Contatti: "
                "mario.rossi@esempio.it, +39 02 2643 1."
            ),
        ),
        StudioExample(
            id="it-business",
            title="Onboarding aziendale",
            blurb="Recapiti aziendali e coordinate bancarie.",
            text=(
                "Gentile Lucia Greco, il suo account cliente n. CL-77821 presso Atelier Romano "
                "è stato attivato. Indirizzo di spedizione: Via del Corso 45, 00186 Roma. "
                "Telefono: +39 06 678 9012, email: lucia.greco@atelierromano.it. "
                "Per i pagamenti automatici, IBAN IT60 X054 2811 1010 0000 0123 456, "
                "BIC UNCRITM1XXX."
            ),
        ),
    ),
    "nl": (
        StudioExample(
            id="nl-clinical",
            title="Medisch verslag",
            blurb="Ontslagbrief met naam, geboortedatum, adres en contact.",
            text=(
                "Mevrouw Anneke de Vries (geboren op 15-03-1985, BSN 123456789) is op "
                "22-04-2026 ontslagen na drie dagen opname. De controleafspraak is op "
                "06-05-2026 met dr. Pieter Jansen, AMC Amsterdam, Meibergdreef 9, "
                "1105 AZ Amsterdam. Bereikbaar op +31 20 566 9111 of "
                "anneke.devries@voorbeeld.nl."
            ),
        ),
        StudioExample(
            id="nl-business",
            title="Zakelijke aanmelding",
            blurb="Zakelijke contactgegevens en bankrekening.",
            text=(
                "Beste Tom Bakker, uw klantnummer bij Rotterdam Logistiek is RT-44210. "
                "Stuur correspondentie naar t.bakker@rotterdamlog.nl of bel +31 10 555 12 34. "
                "Bezorgadres: Coolsingel 75, 3012 AD Rotterdam. Voor automatische incasso, "
                "IBAN NL91 ABNA 0417 1643 00, BIC ABNANL2A."
            ),
        ),
    ),
    "pt": (
        StudioExample(
            id="pt-clinical",
            title="Atendimento clínico",
            blurb="Resumo de alta com nome, data de nascimento, endereço e contato.",
            text=(
                "O paciente João Silva (nascido em 15/03/1985, CPF 123.456.789-09) recebeu "
                "alta em 22/04/2026 após três dias de internação por pneumonia. O retorno "
                "está agendado para 06/05/2026 com a Dra. Ana Costa no Hospital Israelita "
                "Albert Einstein, Avenida Albert Einstein 627, São Paulo SP 05652-900. "
                "Contato: joao.silva@exemplo.com.br, +55 11 2151 1233."
            ),
        ),
        StudioExample(
            id="pt-business",
            title="Cadastro empresarial",
            blurb="Dados de contato e bancários para cobrança.",
            text=(
                "Prezada Mariana Souza, sua conta no Atelier Ipanema foi ativada com o número "
                "CL-77821. Para suporte: mariana.souza@atelieripanema.com.br ou +55 21 2522 0000. "
                "Endereço de entrega: Rua Visconde de Pirajá 414, Ipanema, Rio de Janeiro RJ "
                "22410-002. Para débito automático, IBAN BR97 0036 0305 0000 1000 9795 493P 1."
            ),
        ),
    ),
    "tr": (
        StudioExample(
            id="tr-clinical",
            title="Klinik kayıt",
            blurb="Taburcu özeti — ad, doğum tarihi, adres ve iletişim.",
            text=(
                "Hasta Ayşe Yılmaz (doğum tarihi 15.03.1985, T.C. Kimlik No 12345678901) "
                "üç günlük yatıştan sonra 22.04.2026 tarihinde taburcu edildi. Kontrol "
                "randevusu 06.05.2026 tarihinde Dr. Mehmet Demir ile Acıbadem Maslak Hastanesi, "
                "Büyükdere Caddesi No:40, 34457 Sarıyer İstanbul'da. İletişim: "
                "ayse.yilmaz@ornek.com.tr, +90 212 304 4444."
            ),
        ),
        StudioExample(
            id="tr-business",
            title="Kurumsal kayıt",
            blurb="Kurumsal iletişim ve banka bilgileri.",
            text=(
                "Sayın Emre Kaya, Ankara Lojistik'teki müşteri numaranız ML-44219 olarak "
                "oluşturuldu. İletişim: emre.kaya@ankaralojistik.com.tr veya +90 312 456 78 90. "
                "Teslimat adresi: Atatürk Bulvarı 175, 06680 Çankaya Ankara. "
                "Otomatik ödeme için IBAN TR33 0006 1005 1978 6457 8413 26."
            ),
        ),
    ),
    "ar": (
        StudioExample(
            id="ar-clinical",
            title="تقرير طبي",
            blurb="ملخص خروج بالاسم وتاريخ الميلاد والعنوان ووسائل التواصل.",
            text=(
                "تم خروج المريض أحمد العلي (تاريخ الميلاد 15/03/1985، رقم الهوية 1023456789) "
                "بتاريخ 22/04/2026 بعد إقامة لمدة ثلاثة أيام. موعد المتابعة في 06/05/2026 "
                "مع الدكتورة فاطمة الزهراني في مستشفى الملك فيصل التخصصي، طريق المطار، "
                "الرياض 11211. للتواصل: ahmad.alali@example.sa أو +966 11 442 7777."
            ),
        ),
        StudioExample(
            id="ar-business",
            title="تسجيل تجاري",
            blurb="بيانات الاتصال التجارية ورقم الحساب.",
            text=(
                "السيدة سارة الجابر، تم إنشاء حسابك في شركة الرياض اللوجستية برقم العميل "
                "CL-44219. للتواصل: sara.aljaber@riyadhlog.sa أو الهاتف +966 11 510 2200. "
                "عنوان التسليم: شارع العليا، حي العليا، الرياض 12241. "
                "رقم الآيبان للحوالات: SA03 8000 0000 6080 1016 7519."
            ),
        ),
    ),
    "hi": (
        StudioExample(
            id="hi-clinical",
            title="चिकित्सा रिपोर्ट",
            blurb="रोगी का नाम, जन्म तिथि, पता और संपर्क।",
            text=(
                "रोगी राजेश कुमार (जन्म तिथि 15/03/1985, आधार संख्या 1234 5678 9012) को "
                "तीन दिन की भर्ती के बाद 22/04/2026 को छुट्टी दी गई। अनुवर्ती मुलाकात "
                "06/05/2026 को डॉ. प्रिया शर्मा के साथ अपोलो अस्पताल, मथुरा रोड, सरिता विहार, "
                "नई दिल्ली 110076 में है। संपर्क: rajesh.kumar@example.in, +91 11 2692 5858।"
            ),
        ),
        StudioExample(
            id="hi-business",
            title="व्यावसायिक पंजीकरण",
            blurb="कार्यालय संपर्क और बैंक विवरण।",
            text=(
                "प्रिय अनिता वर्मा, भारत लॉजिस्टिक्स में आपका ग्राहक खाता CL-44219 के रूप में "
                "सक्रिय है। संपर्क: anita.verma@bharatlog.in या +91 22 6680 1818। "
                "वितरण पता: पाली हिल, बांद्रा वेस्ट, मुंबई 400050। ऑटो भुगतान के लिए, "
                "IFSC HDFC0000060, खाता संख्या 50100123456789।"
            ),
        ),
    ),
    "bn": (
        StudioExample(
            id="bn-clinical",
            title="চিকিৎসা প্রতিবেদন",
            blurb="রোগীর নাম, জন্ম তারিখ, ঠিকানা এবং যোগাযোগ।",
            text=(
                "রোগী আবদুল রহমান (জন্ম তারিখ 15/03/1985, জাতীয় পরিচয় 1234567890123) "
                "তিন দিনের ভর্তির পর 22/04/2026 তারিখে ছাড়পত্র পেয়েছেন। ফলো-আপ "
                "06/05/2026 তারিখে ডাঃ রুমানা আক্তার-এর সাথে স্কয়ার হাসপাতাল, "
                "১৮/এফ ওয়েস্ট পান্থপথ, ঢাকা ১২০৫-এ। যোগাযোগ: abdul.rahman@example.bd, "
                "+880 2 8159457।"
            ),
        ),
        StudioExample(
            id="bn-business",
            title="ব্যবসায়িক নিবন্ধন",
            blurb="অফিসের যোগাযোগ এবং ব্যাংক বিবরণ।",
            text=(
                "প্রিয় তানিয়া হোসেন, ঢাকা লজিস্টিকসে আপনার গ্রাহক নম্বর হল CL-44219। "
                "যোগাযোগ: tania.hossain@dhakalog.bd অথবা +880 2 9883590। "
                "ডেলিভারি ঠিকানা: গুলশান-২, ঢাকা ১২১২। স্বয়ংক্রিয় পেমেন্টের জন্য, "
                "ব্যাংক হিসাব 1052020123456 (DBBL)।"
            ),
        ),
    ),
    "te": (
        StudioExample(
            id="te-clinical",
            title="వైద్య నివేదిక",
            blurb="రోగి పేరు, పుట్టిన తేదీ, చిరునామా మరియు సంప్రదింపులు.",
            text=(
                "రోగి రామకృష్ణ రెడ్డి (పుట్టిన తేదీ 15/03/1985, ఆధార్ సంఖ్య 1234 5678 9012) "
                "మూడు రోజుల చేరిక తర్వాత 22/04/2026న డిశ్చార్జ్ చేయబడ్డారు. ఫాలో-అప్ "
                "06/05/2026న డాక్టర్ లక్ష్మీ ప్రియతో అపోలో హాస్పిటల్, రోడ్ నెం. 72, "
                "జూబ్లీ హిల్స్, హైదరాబాద్ 500033లో. సంప్రదింపులు: "
                "ramakrishna.reddy@example.in, +91 40 2360 7777."
            ),
        ),
        StudioExample(
            id="te-business",
            title="వ్యాపార నమోదు",
            blurb="వ్యాపార సంప్రదింపులు మరియు బ్యాంక్ వివరాలు.",
            text=(
                "ప్రియమైన శ్రీలత నాయుడు, తెలంగాణ లాజిస్టిక్స్‌లో మీ కస్టమర్ ఖాతా CL-44219 "
                "సక్రియం చేయబడింది. సంప్రదింపులు: srilatha.naidu@telanganalog.in లేదా "
                "+91 40 6680 1818. డెలివరీ చిరునామా: బంజారా హిల్స్, రోడ్ నెం. 12, "
                "హైదరాబాద్ 500034. ఆటో పేమెంట్ కోసం, IFSC SBIN0020001, "
                "ఖాతా సంఖ్య 30123456789."
            ),
        ),
    ),
    "vi": (
        StudioExample(
            id="vi-clinical",
            title="Báo cáo y khoa",
            blurb="Bệnh nhân — họ tên, ngày sinh, địa chỉ và liên lạc.",
            text=(
                "Bệnh nhân Nguyễn Văn An (sinh ngày 15/03/1985, CMND 012345678901) đã được "
                "xuất viện ngày 22/04/2026 sau ba ngày điều trị. Hẹn tái khám ngày "
                "06/05/2026 với BS. Trần Thị Hương tại Bệnh viện Bạch Mai, 78 Đường Giải Phóng, "
                "Đống Đa, Hà Nội. Liên hệ: nguyen.van.an@example.vn, +84 24 3869 3731."
            ),
        ),
        StudioExample(
            id="vi-business",
            title="Đăng ký doanh nghiệp",
            blurb="Liên hệ doanh nghiệp và thông tin ngân hàng.",
            text=(
                "Kính gửi anh Lê Quốc Bảo, tài khoản khách hàng của anh tại Sài Gòn Logistics "
                "đã được kích hoạt với mã CL-44219. Liên hệ: le.quoc.bao@saigonlog.vn hoặc "
                "+84 28 3829 5555. Địa chỉ giao hàng: 8 Nguyễn Huệ, Quận 1, TP. Hồ Chí Minh. "
                "Để thanh toán tự động, số tài khoản 0102345678 tại Vietcombank."
            ),
        ),
    ),
    "zh": (
        StudioExample(
            id="zh-clinical",
            title="临床报告",
            blurb="出院摘要 — 姓名、出生日期、地址和联系方式。",
            text=(
                "患者王明(出生日期 1985年3月15日,身份证号 110108198503150018)经过三天住院,"
                "于2026年4月22日出院。复诊预约2026年5月6日,由李华医生在北京协和医院,"
                "北京市东城区帅府园1号。联系方式: wang.ming@example.cn 或 "
                "+86 10 6915 6699。"
            ),
        ),
        StudioExample(
            id="zh-business",
            title="商业开户",
            blurb="商业联系方式和银行信息。",
            text=(
                "尊敬的张丽女士,您在上海物流的客户账号 CL-44219 已激活。联系方式: "
                "zhang.li@shanghailog.cn 或 +86 21 6391 8888。送货地址: "
                "上海市浦东新区世纪大道100号。自动扣款账号 6225 8801 2345 6789(招商银行),"
                "BIC CMBCCNBS。"
            ),
        ),
    ),
    "ja": (
        StudioExample(
            id="ja-clinical",
            title="医療レポート",
            blurb="退院サマリー — 氏名、生年月日、住所、連絡先。",
            text=(
                "患者の田中 健一(生年月日 1985年3月15日、保険証番号 12345678)は、3日間の入院後、"
                "2026年4月22日に退院しました。次回診察は2026年5月6日、佐藤 美咲医師による"
                "東京大学医学部附属病院、東京都文京区本郷7-3-1にて。連絡先: "
                "tanaka.kenichi@example.jp、+81 3 3815 5411。"
            ),
        ),
        StudioExample(
            id="ja-business",
            title="ビジネス登録",
            blurb="事業連絡先と銀行情報。",
            text=(
                "山田 由美様、東京物流における顧客番号 CL-44219 が有効になりました。"
                "ご連絡先: yamada.yumi@tokyolog.jp または +81 3 5562 9911。"
                "配送先住所: 東京都港区赤坂3-1-1。自動引き落としは三菱UFJ銀行、"
                "口座番号 0123456 をご利用ください。"
            ),
        ),
    ),
    "ko": (
        StudioExample(
            id="ko-clinical",
            title="진료 기록",
            blurb="환자 정보 — 성명, 생년월일, 주소, 연락처.",
            text=(
                "환자 김민수(생년월일 1985년 3월 15일, 주민등록번호 850315-1234567)는 "
                "3일간 입원 후 2026년 4월 22일에 퇴원했습니다. 후속 진료는 2026년 5월 6일, "
                "이영희 의사와 서울아산병원, 서울특별시 송파구 올림픽로 88에서 예정되어 있습니다. "
                "연락처: kim.minsu@example.kr, +82 2 3010 3114."
            ),
        ),
        StudioExample(
            id="ko-business",
            title="비즈니스 등록",
            blurb="회사 연락처와 계좌 정보.",
            text=(
                "박지영 고객님, 서울 물류의 고객 번호 CL-44219가 활성화되었습니다. "
                "연락처: park.jiyoung@seoullog.kr 또는 +82 2 3275 9999. "
                "배송지: 서울특별시 강남구 테헤란로 152. 자동 결제는 신한은행 "
                "계좌번호 110-123-456789를 이용해주세요."
            ),
        ),
    ),
}


# Sanity check: every language listed in LANGUAGE_META must have at least one example.
for entry in LANGUAGE_META:
    code = entry["code"]
    if code not in LANGUAGE_EXAMPLES or not LANGUAGE_EXAMPLES[code]:
        raise RuntimeError(f"Missing examples for language: {code}")
