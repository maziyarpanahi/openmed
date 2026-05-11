import Foundation

/// Bundled sample clinical note so "Use Sample" returns instant text without
/// requiring OCR. Mirrors the style of the image asset (`SampleClinicalDocument`)
/// but can be fed straight into the PII + clinical pipelines.
public enum SampleClinicalText {
    public enum Language: String, CaseIterable, Identifiable, Hashable {
        case en, fr, ar

        public var id: String { rawValue }

        public var buttonTitle: String { rawValue.uppercased() }

        public var displayName: String {
            switch self {
            case .en: return "English"
            case .fr: return "French"
            case .ar: return "Arabic"
            }
        }

        public var assetName: String {
            switch self {
            case .en: return "SampleClinicalDocument"
            case .fr: return "SampleClinicalDocumentFrench"
            case .ar: return "SampleClinicalDocumentArabic"
            }
        }

        public var note: String {
            switch self {
            case .en: return SampleClinicalText.note
            case .fr: return SampleClinicalText.frenchNote
            case .ar: return SampleClinicalText.arabicNote
            }
        }
    }

    public static let note: String = """
    PATIENT NOTE — Saint Mary Community Hospital

    Patient: Eleanor Ruiz
    DOB: 03/14/1981
    MRN: MRN-448271
    Phone: (415) 555-0142
    Email: eleanor.ruiz@sampleclinic.test
    Address: 1942 Harbor View Drive, Marseille, CA 92111
    Insurance ID: HMO-99318442
    Employer: Blue Harbor Foods
    Emergency contact: Martin Ruiz, (415) 555-0199
    Encounter date: 04/16/2026
    Provider: Dr. Maya Shah, MD — Emergency Department

    HISTORY OF PRESENT ILLNESS
    The patient is a 45-year-old woman with type 2 diabetes, hypertension, chronic migraine, hyperlipidemia, and GERD who presents after urgent care hydration for persistent frontal headache, dizziness, nausea, and reduced oral intake. Home finger-stick glucose this morning was 212 mg/dL.

    ALLERGIES
    Penicillin — reported rash in childhood.

    MEDICATIONS
    Metformin 500 mg twice daily, lisinopril 10 mg daily, atorvastatin 20 mg nightly, sumatriptan 50 mg as needed, ondansetron 4 mg every 8 hours as needed.

    ASSESSMENT
    Migraine flare with dehydration and mild hyperglycemia; neurologic exam stable.

    PLAN
    1. Continue oral hydration and ibuprofen 400 mg every 6 hours as needed.
    2. PCP follow-up within 48 hours and neurology follow-up within 2 weeks.
    3. Return to the emergency department for worsening headache, chest pain, repeated vomiting, syncope, confusion, or weakness.

    Signed electronically by Dr. Maya Shah, MD on 04/16/2026.
    """

    public static let frenchNote: String = """
    COMPTE RENDU PATIENT — Centre Hospitalier Saint-Martin

    Patiente : Claire Benali
    Date de naissance : 14/03/1981
    NIR : 2 81 03 75 116 002 89
    Dossier : FR-MRN-772901
    Telephone : +33 6 42 18 09 77
    Courriel : claire.benali@example.fr
    Adresse : 18 rue des Lilas, 75011 Paris, France
    Mutuelle : MGEN-4472-9910
    Employeur : Atelier Lumiere
    Contact d'urgence : Karim Benali, +33 6 11 22 33 44
    Date de consultation : 16/04/2026
    Medecin : Dre Elise Moreau — Service des urgences

    HISTOIRE DE LA MALADIE
    Patiente de 45 ans avec diabete de type 2, hypertension arterielle, migraine chronique, hyperlipidemie et reflux gastro-oesophagien. Elle consulte apres une perfusion de serum physiologique en soins non programmes pour cephalee frontale persistante, vertiges, nausees et apports reduits. Glycemie capillaire ce matin : 212 mg/dL.

    ALLERGIES
    Penicilline — eruption cutanee dans l'enfance.

    TRAITEMENTS
    Metformine 500 mg deux fois par jour, lisinopril 10 mg par jour, atorvastatine 20 mg le soir, sumatriptan 50 mg si besoin, ondansetron 4 mg toutes les 8 heures si besoin.

    EVALUATION
    Poussee migraineuse avec deshydratation et hyperglycemie moderee ; examen neurologique stable sans deficit focal.

    PLAN
    1. Hydratation orale et ibuprofene 400 mg toutes les 6 heures si besoin.
    2. Suivi avec le medecin traitant sous 48 heures et neurologie sous 2 semaines.
    3. Retour aux urgences en cas de cephalee aggravée, douleur thoracique, vomissements repetes, malaise, confusion ou faiblesse.

    Signature electronique : Dre Elise Moreau, 16/04/2026.
    """

    public static let arabicNote: String = """
    تقرير المريض — مركز النور الطبي

    المريضة: ليلى منصور
    تاريخ الميلاد: 14/03/1981
    رقم الهوية: 784-1981-4472
    رقم الملف الطبي: AR-MRN-552901
    الهاتف: +971 50 442 7719
    البريد: layla.mansour@example.ae
    العنوان: برج الندى، شارع الشيخ زايد، دبي 00000
    رقم التأمين: DUB-8842-1190
    جهة العمل: Horizon Trade LLC
    جهة الاتصال للطوارئ: سامر منصور، +971 55 119 3322
    تاريخ الزيارة: 16/04/2026
    الطبيب: د. نورة الحسن — قسم الطوارئ

    التاريخ المرضي
    مريضة تبلغ 45 عاما لديها سكري من النوع الثاني، ارتفاع ضغط الدم، صداع نصفي مزمن، ارتفاع الدهون، وارتجاع معدي مريئي. حضرت بعد تلقي سوائل وريدية في عيادة عاجلة بسبب صداع جبهي مستمر، دوخة، غثيان، وقلة تناول السوائل. قياس السكر المنزلي صباحا 212 mg/dL.

    الحساسية
    حساسية من البنسلين — طفح جلدي في الطفولة.

    الادوية
    Metformin 500 mg مرتين يوميا، lisinopril 10 mg يوميا، atorvastatin 20 mg ليلا، sumatriptan 50 mg عند اللزوم، ondansetron 4 mg كل 8 ساعات عند اللزوم.

    التقييم
    نوبة صداع نصفي مع جفاف وارتفاع بسيط في السكر؛ الفحص العصبي مستقر ولا توجد علامات بؤرية.

    الخطة
    1. الاستمرار على شرب السوائل و ibuprofen 400 mg كل 6 ساعات عند اللزوم.
    2. مراجعة طبيب الرعاية الأولية خلال 48 ساعة ومراجعة الأعصاب خلال أسبوعين.
    3. العودة للطوارئ عند زيادة الصداع، ألم الصدر، تكرر القيء، الإغماء، التشوش أو الضعف.

    تم التوقيع إلكترونيا بواسطة د. نورة الحسن في 16/04/2026.
    """
}
