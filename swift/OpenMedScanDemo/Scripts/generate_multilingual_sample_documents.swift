import AppKit
import Foundation

struct Field {
    let label: String
    let value: String
}

struct SectionBlock {
    let title: String
    let body: String
}

struct DemoDocument {
    let assetFolder: String
    let fileName: String
    let facility: String
    let facilityLine: String
    let documentTitle: String
    let visitDate: String
    let fields: [Field]
    let sections: [SectionBlock]
    let footerLeft: String
    let signature: String
    let isRTL: Bool
}

let scriptURL = URL(fileURLWithPath: #filePath)
let repoRoot = scriptURL
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
let assetsRoot = repoRoot.appending(path: "swift/OpenMedScanDemo/OpenMedScanDemo/Assets.xcassets")

let french = DemoDocument(
    assetFolder: "SampleClinicalDocumentFrench.imageset",
    fileName: "sample-clinical-document-fr.png",
    facility: "Centre Hospitalier Saint-Martin",
    facilityLine: "24 avenue de la Republique · 75011 Paris · +33 1 44 55 66 77",
    documentTitle: "Compte rendu des urgences",
    visitDate: "Consultation : 16/04/2026",
    fields: [
        .init(label: "Patiente", value: "Claire Benali"),
        .init(label: "Date de naissance", value: "14/03/1981"),
        .init(label: "NIR", value: "2 81 03 75 116 002 89"),
        .init(label: "Dossier", value: "FR-MRN-772901"),
        .init(label: "Telephone", value: "+33 6 42 18 09 77"),
        .init(label: "Courriel", value: "claire.benali@example.fr"),
        .init(label: "Adresse", value: "18 rue des Lilas, 75011 Paris"),
        .init(label: "Mutuelle", value: "MGEN-4472-9910"),
        .init(label: "Employeur", value: "Atelier Lumiere"),
        .init(label: "Urgence", value: "Karim Benali, +33 6 11 22 33 44"),
    ],
    sections: [
        .init(title: "Histoire de la maladie", body: "Patiente de 45 ans avec diabete de type 2, hypertension arterielle, migraine chronique, hyperlipidemie et reflux gastro-oesophagien. Elle consulte apres une perfusion de serum physiologique pour cephalee frontale persistante, vertiges, nausees et apports reduits."),
        .init(title: "Allergies et traitements", body: "Penicilline : eruption cutanee dans l'enfance. Traitements : metformine 500 mg deux fois par jour, lisinopril 10 mg, atorvastatine 20 mg, sumatriptan 50 mg si besoin, ondansetron 4 mg si besoin."),
        .init(title: "Evaluation", body: "Poussee migraineuse avec deshydratation et hyperglycemie moderee. Examen neurologique stable, sans deficit focal. Glycemie capillaire ce matin : 212 mg/dL."),
        .init(title: "Plan", body: "Hydratation orale, ibuprofene 400 mg toutes les 6 heures si besoin, suivi avec le medecin traitant sous 48 heures et rendez-vous neurologie sous 2 semaines. Retour aux urgences en cas de cephalee aggravée, douleur thoracique, vomissements repetes, malaise, confusion ou faiblesse."),
    ],
    footerLeft: "Prescription : ibuprofene 400 mg PO q6h PRN\nOndansetron 4 mg PO q8h PRN nausees",
    signature: "Dre Elise Moreau\nService des urgences",
    isRTL: false
)

let arabic = DemoDocument(
    assetFolder: "SampleClinicalDocumentArabic.imageset",
    fileName: "sample-clinical-document-ar.png",
    facility: "مركز النور الطبي",
    facilityLine: "شارع الشيخ زايد · دبي 00000 · +971 4 555 6677",
    documentTitle: "تقرير قسم الطوارئ",
    visitDate: "تاريخ الزيارة: 16/04/2026",
    fields: [
        .init(label: "المريضة", value: "ليلى منصور"),
        .init(label: "تاريخ الميلاد", value: "14/03/1981"),
        .init(label: "رقم الهوية", value: "784-1981-4472"),
        .init(label: "رقم الملف الطبي", value: "AR-MRN-552901"),
        .init(label: "الهاتف", value: "+971 50 442 7719"),
        .init(label: "البريد", value: "layla.mansour@example.ae"),
        .init(label: "العنوان", value: "برج الندى، شارع الشيخ زايد، دبي"),
        .init(label: "رقم التأمين", value: "DUB-8842-1190"),
        .init(label: "جهة العمل", value: "Horizon Trade LLC"),
        .init(label: "الطوارئ", value: "سامر منصور، +971 55 119 3322"),
    ],
    sections: [
        .init(title: "التاريخ المرضي", body: "مريضة تبلغ 45 عاما لديها سكري من النوع الثاني، ارتفاع ضغط الدم، صداع نصفي مزمن، ارتفاع الدهون، وارتجاع معدي مريئي. حضرت بعد تلقي سوائل وريدية في عيادة عاجلة بسبب صداع جبهي مستمر، دوخة، غثيان، وقلة تناول السوائل."),
        .init(title: "الحساسية والأدوية", body: "حساسية من البنسلين مع طفح جلدي في الطفولة. الادوية: Metformin 500 mg مرتين يوميا، lisinopril 10 mg يوميا، atorvastatin 20 mg ليلا، sumatriptan 50 mg عند اللزوم، ondansetron 4 mg كل 8 ساعات عند اللزوم."),
        .init(title: "التقييم", body: "نوبة صداع نصفي مع جفاف وارتفاع بسيط في السكر. الفحص العصبي مستقر ولا توجد علامات بؤرية. قياس السكر المنزلي صباحا 212 mg/dL."),
        .init(title: "الخطة", body: "الاستمرار على شرب السوائل و ibuprofen 400 mg كل 6 ساعات عند اللزوم. مراجعة طبيب الرعاية الأولية خلال 48 ساعة ومراجعة الأعصاب خلال أسبوعين. العودة للطوارئ عند زيادة الصداع، ألم الصدر، تكرر القيء، الإغماء، التشوش أو الضعف."),
    ],
    footerLeft: "الوصفة: ibuprofen 400 mg عند اللزوم\nondansetron 4 mg عند الغثيان",
    signature: "د. نورة الحسن\nقسم الطوارئ",
    isRTL: true
)

let documents = [french, arabic]

let canvasSize = NSSize(width: 1600, height: 2520)
let paperRect = NSRect(x: 130, y: 110, width: 1340, height: 2300)
let marginX: CGFloat = 94
let titleColor = NSColor(calibratedRed: 0.12, green: 0.25, blue: 0.42, alpha: 1.0)
let accentColor = NSColor(calibratedRed: 0.22, green: 0.53, blue: 0.74, alpha: 1.0)
let bodyColor = NSColor(calibratedWhite: 0.18, alpha: 1.0)
let mutedColor = NSColor(calibratedWhite: 0.46, alpha: 1.0)
let lineColor = NSColor(calibratedWhite: 0.89, alpha: 1.0)
let titleFont = NSFont.systemFont(ofSize: 30, weight: .bold)
let subtitleFont = NSFont.systemFont(ofSize: 17, weight: .medium)
let sectionTitleFont = NSFont.systemFont(ofSize: 15, weight: .bold)
let bodyFont = NSFont.systemFont(ofSize: 21, weight: .regular)
let labelFont = NSFont.systemFont(ofSize: 13, weight: .semibold)
let valueFont = NSFont.systemFont(ofSize: 19, weight: .regular)
let monoFont = NSFont.monospacedSystemFont(ofSize: 17, weight: .medium)

func paragraphStyle(alignment: NSTextAlignment, lineHeightMultiple: CGFloat = 1.18) -> NSParagraphStyle {
    let style = NSMutableParagraphStyle()
    style.alignment = alignment
    style.lineBreakMode = .byWordWrapping
    style.lineHeightMultiple = lineHeightMultiple
    style.baseWritingDirection = alignment == .right ? .rightToLeft : .leftToRight
    return style
}

func measure(text: String, width: CGFloat, font: NSFont, alignment: NSTextAlignment, lineHeightMultiple: CGFloat = 1.18) -> CGFloat {
    let attributes: [NSAttributedString.Key: Any] = [
        .font: font,
        .paragraphStyle: paragraphStyle(alignment: alignment, lineHeightMultiple: lineHeightMultiple),
    ]
    let rect = NSString(string: text).boundingRect(
        with: NSSize(width: width, height: .greatestFiniteMagnitude),
        options: [.usesLineFragmentOrigin, .usesFontLeading],
        attributes: attributes
    )
    return ceil(rect.height)
}

func draw(text: String, in rect: NSRect, font: NSFont, color: NSColor, alignment: NSTextAlignment, lineHeightMultiple: CGFloat = 1.18) {
    let attributes: [NSAttributedString.Key: Any] = [
        .font: font,
        .foregroundColor: color,
        .paragraphStyle: paragraphStyle(alignment: alignment, lineHeightMultiple: lineHeightMultiple),
    ]
    NSString(string: text).draw(
        with: rect,
        options: [.usesLineFragmentOrigin, .usesFontLeading],
        attributes: attributes
    )
}

func horizontalRule(y: CGFloat, color: NSColor, lineWidth: CGFloat = 1.0) {
    color.setStroke()
    let path = NSBezierPath()
    path.lineWidth = lineWidth
    path.move(to: CGPoint(x: paperRect.minX + marginX, y: y))
    path.line(to: CGPoint(x: paperRect.maxX - marginX, y: y))
    path.stroke()
}

func render(_ document: DemoDocument) throws {
    let image = NSImage(size: canvasSize)
    image.lockFocus()
    guard let context = NSGraphicsContext.current?.cgContext else {
        fatalError("Missing graphics context")
    }

    let alignment: NSTextAlignment = document.isRTL ? .right : .left
    let dateAlignment: NSTextAlignment = document.isRTL ? .left : .right

    NSColor(calibratedRed: 0.92, green: 0.94, blue: 0.96, alpha: 1.0).setFill()
    NSBezierPath(rect: NSRect(origin: .zero, size: canvasSize)).fill()

    context.setShadow(offset: CGSize(width: 0, height: -20), blur: 36, color: NSColor.black.withAlphaComponent(0.16).cgColor)
    NSColor(calibratedRed: 0.985, green: 0.982, blue: 0.968, alpha: 1.0).setFill()
    NSBezierPath(roundedRect: paperRect, xRadius: 26, yRadius: 26).fill()
    context.setShadow(offset: .zero, blur: 0, color: nil)

    NSColor(calibratedWhite: 0.86, alpha: 1.0).setStroke()
    let border = NSBezierPath(roundedRect: paperRect, xRadius: 26, yRadius: 26)
    border.lineWidth = 1.5
    border.stroke()

    let logoRect = document.isRTL
        ? NSRect(x: paperRect.maxX - marginX - 72, y: paperRect.maxY - 156, width: 72, height: 72)
        : NSRect(x: paperRect.minX + marginX, y: paperRect.maxY - 156, width: 72, height: 72)
    accentColor.setFill()
    NSBezierPath(roundedRect: logoRect, xRadius: 20, yRadius: 20).fill()
    draw(text: "OM", in: logoRect.offsetBy(dx: 0, dy: 11), font: NSFont.systemFont(ofSize: 30, weight: .bold), color: .white, alignment: .center)

    let headerX = document.isRTL ? paperRect.minX + 420 : logoRect.maxX + 24
    let headerWidth: CGFloat = 760
    draw(text: document.facility, in: NSRect(x: headerX, y: paperRect.maxY - 122, width: headerWidth, height: 42), font: titleFont, color: titleColor, alignment: alignment)
    draw(text: document.facilityLine, in: NSRect(x: headerX, y: paperRect.maxY - 160, width: headerWidth, height: 24), font: subtitleFont, color: mutedColor, alignment: alignment)

    draw(
        text: document.documentTitle,
        in: NSRect(x: paperRect.minX + marginX, y: paperRect.maxY - 250, width: 760, height: 40),
        font: NSFont.systemFont(ofSize: 33, weight: .semibold),
        color: bodyColor,
        alignment: alignment
    )
    draw(
        text: document.visitDate,
        in: NSRect(x: paperRect.maxX - 430, y: paperRect.maxY - 242, width: 330, height: 28),
        font: monoFont,
        color: mutedColor,
        alignment: dateAlignment
    )
    horizontalRule(y: paperRect.maxY - 264, color: accentColor.withAlphaComponent(0.44), lineWidth: 2.0)

    let columnGap: CGFloat = 36
    let columnWidth = (paperRect.width - (marginX * 2) - columnGap) / 2
    var currentY = paperRect.maxY - 314
    for row in stride(from: 0, to: document.fields.count, by: 2) {
        let first = document.fields[row]
        let second = row + 1 < document.fields.count ? document.fields[row + 1] : nil
        let rowHeight: CGFloat = 78
        let x1 = paperRect.minX + marginX
        let x2 = paperRect.minX + marginX + columnWidth + columnGap
        let rowFields: [(Field, CGFloat)] = [(first, x1)] + (second.map { [($0, x2)] } ?? [])
        for (field, x) in rowFields {
            draw(text: field.label.uppercased(), in: NSRect(x: x, y: currentY - 18, width: columnWidth, height: 16), font: labelFont, color: mutedColor, alignment: alignment)
            draw(text: field.value, in: NSRect(x: x, y: currentY - 56, width: columnWidth, height: 28), font: valueFont, color: bodyColor, alignment: alignment)
        }
        horizontalRule(y: currentY - rowHeight, color: lineColor)
        currentY -= (rowHeight + 12)
    }

    currentY -= 10
    let sectionWidth = paperRect.width - (marginX * 2)
    for section in document.sections {
        let titleHeight = measure(text: section.title.uppercased(), width: sectionWidth, font: sectionTitleFont, alignment: alignment, lineHeightMultiple: 1.0)
        let bodyHeight = measure(text: section.body, width: sectionWidth, font: bodyFont, alignment: alignment)
        draw(
            text: section.title.uppercased(),
            in: NSRect(x: paperRect.minX + marginX, y: currentY - titleHeight, width: sectionWidth, height: titleHeight),
            font: sectionTitleFont,
            color: accentColor,
            alignment: alignment,
            lineHeightMultiple: 1.0
        )
        let bodyTop = currentY - titleHeight - 10
        draw(
            text: section.body,
            in: NSRect(x: paperRect.minX + marginX, y: bodyTop - bodyHeight, width: sectionWidth, height: bodyHeight),
            font: bodyFont,
            color: bodyColor,
            alignment: alignment
        )
        currentY = bodyTop - bodyHeight - 28
        horizontalRule(y: currentY, color: lineColor)
        currentY -= 34
    }

    horizontalRule(y: paperRect.minY + 190, color: lineColor, lineWidth: 1.2)
    draw(text: document.footerLeft, in: NSRect(x: paperRect.minX + marginX, y: paperRect.minY + 82, width: 620, height: 86), font: monoFont, color: bodyColor, alignment: alignment)
    let signatureLabel = document.isRTL ? "تم التوقيع إلكترونيا" : "Signature electronique"
    draw(text: "\(signatureLabel)\n\(document.signature)", in: NSRect(x: paperRect.maxX - 410, y: paperRect.minY + 66, width: 310, height: 100), font: subtitleFont, color: bodyColor, alignment: .right)

    NSColor(calibratedWhite: 0.0, alpha: 0.01).setFill()
    for _ in 0..<220 {
        let x = CGFloat.random(in: paperRect.minX...(paperRect.maxX - 2))
        let y = CGFloat.random(in: paperRect.minY...(paperRect.maxY - 2))
        NSBezierPath(ovalIn: NSRect(x: x, y: y, width: 1.8, height: 1.8)).fill()
    }

    image.unlockFocus()

    guard let tiffRepresentation = image.tiffRepresentation,
          let bitmap = NSBitmapImageRep(data: tiffRepresentation),
          let pngData = bitmap.representation(using: .png, properties: [:]) else {
        fatalError("Could not create PNG representation")
    }

    let outputURL = assetsRoot
        .appending(path: document.assetFolder)
        .appending(path: document.fileName)
    try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
    try pngData.write(to: outputURL, options: .atomic)
    print(outputURL.path)
}

for document in documents {
    try render(document)
}
