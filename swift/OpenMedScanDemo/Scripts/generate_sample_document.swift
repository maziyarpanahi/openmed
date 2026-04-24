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

let fileManager = FileManager.default
let scriptURL = URL(fileURLWithPath: #filePath)
let repoRoot = scriptURL
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
    .deletingLastPathComponent()
let outputURL = repoRoot
    .appending(path: "swift/OpenMedScanDemo/OpenMedScanDemo/Assets.xcassets/SampleClinicalDocument.imageset/sample-clinical-document.png")

let canvasSize = NSSize(width: 1600, height: 2520)
let paperRect = NSRect(x: 130, y: 110, width: 1340, height: 2300)
let marginX: CGFloat = 94

let image = NSImage(size: canvasSize)
image.lockFocus()
guard let context = NSGraphicsContext.current?.cgContext else {
    fatalError("Missing graphics context")
}

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

let leftFields: [Field] = [
    .init(label: "Patient Name", value: "Eleanor Ruiz"),
    .init(label: "DOB", value: "03/14/1981"),
    .init(label: "MRN", value: "MRN-448271"),
    .init(label: "Address", value: "1942 Harbor View Drive, Marseille, CA 92111"),
    .init(label: "Phone", value: "(415) 555-0142"),
]

let rightFields: [Field] = [
    .init(label: "Email", value: "eleanor.ruiz@sampleclinic.test"),
    .init(label: "Insurance ID", value: "HMO-99318442"),
    .init(label: "Employer", value: "Blue Harbor Foods"),
    .init(label: "Emergency Contact", value: "Martin Ruiz, (415) 555-0199"),
    .init(label: "Primary Clinician", value: "Dr. Maya Shah, MD"),
]

let sections: [SectionBlock] = [
    .init(
        title: "Chief Concern",
        body: "Persistent frontal headache, dizziness, nausea, and reduced oral intake after urgent care hydration."
    ),
    .init(
        title: "History of Present Illness",
        body: "58-year-old woman with type 2 diabetes mellitus, hypertension, chronic migraine, hyperlipidemia, and GERD presents for emergency department follow-up. She reports 2 days of worsening frontal headache rated 7/10 with lightheadedness, photophobia, nausea, and poor oral intake. She received 1 L IV normal saline and ondansetron 4 mg ODT at urgent care yesterday with partial relief. Home finger-stick glucose this morning was 212 mg/dL."
    ),
    .init(
        title: "Medical History",
        body: "Type 2 diabetes mellitus, hypertension, chronic migraine, hyperlipidemia, and GERD. No prior stroke, seizure, or heart failure."
    ),
    .init(
        title: "Home Medications",
        body: "Metformin 500 mg twice daily, lisinopril 10 mg daily, atorvastatin 20 mg nightly, and sumatriptan 50 mg as needed for migraine."
    ),
    .init(
        title: "Allergy",
        body: "Penicillin causes a pruritic rash. No known latex allergy."
    ),
    .init(
        title: "Exam and Vitals",
        body: "BP 152/94 mmHg, HR 102 bpm, Temp 98.4 F, SpO2 98% on room air. Mild dry mucous membranes. Neurologically intact with no focal deficit."
    ),
    .init(
        title: "Labs and Procedures",
        body: "Point-of-care glucose 212 mg/dL, sodium 133 mmol/L, creatinine 1.1 mg/dL. Finger-stick glucose and orthostatic vitals obtained. 12-lead ECG showed normal sinus rhythm."
    ),
    .init(
        title: "Assessment",
        body: "Migraine flare with dehydration and mild hyperglycemia; lower concern for acute intracranial process given stable neurologic exam and symptom improvement after hydration."
    ),
    .init(
        title: "Plan and Follow-Up",
        body: "Continue oral hydration, ibuprofen 400 mg every 6 hours as needed, ondansetron 4 mg every 8 hours as needed for nausea, and resume home medications. PCP follow-up within 48 hours and neurology follow-up within 2 weeks. Return to the emergency department for worsening headache, chest pain, repeated vomiting, syncope, confusion, or weakness."
    ),
    .init(
        title: "Work Status",
        body: "May return to work on 04/19/2026 if headache and dizziness continue to improve."
    ),
]

func paragraphStyle(alignment: NSTextAlignment = .left, lineHeightMultiple: CGFloat = 1.18) -> NSParagraphStyle {
    let style = NSMutableParagraphStyle()
    style.alignment = alignment
    style.lineBreakMode = .byWordWrapping
    style.lineHeightMultiple = lineHeightMultiple
    return style
}

func measure(text: String, width: CGFloat, font: NSFont, alignment: NSTextAlignment = .left, lineHeightMultiple: CGFloat = 1.18) -> CGFloat {
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

func draw(text: String, in rect: NSRect, font: NSFont, color: NSColor, alignment: NSTextAlignment = .left, lineHeightMultiple: CGFloat = 1.18) {
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

func horizontalRule(y: CGFloat, color: NSColor, lineWidth: CGFloat = 1.0, leftInset: CGFloat = marginX, rightInset: CGFloat = marginX) {
    color.setStroke()
    let path = NSBezierPath()
    path.lineWidth = lineWidth
    path.move(to: CGPoint(x: paperRect.minX + leftInset, y: y))
    path.line(to: CGPoint(x: paperRect.maxX - rightInset, y: y))
    path.stroke()
}

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

let logoRect = NSRect(x: paperRect.minX + marginX, y: paperRect.maxY - 156, width: 72, height: 72)
accentColor.setFill()
NSBezierPath(roundedRect: logoRect, xRadius: 20, yRadius: 20).fill()
draw(text: "OM", in: logoRect.offsetBy(dx: 0, dy: 11), font: NSFont.systemFont(ofSize: 30, weight: .bold), color: .white, alignment: .center)

draw(
    text: "OpenMed Bayview Outpatient Center",
    in: NSRect(x: logoRect.maxX + 24, y: paperRect.maxY - 122, width: 760, height: 42),
    font: titleFont,
    color: titleColor
)
draw(
    text: "390 Harbor Clinical Plaza · San Diego, CA 92111 · (415) 555-0100",
    in: NSRect(x: logoRect.maxX + 24, y: paperRect.maxY - 160, width: 760, height: 24),
    font: subtitleFont,
    color: mutedColor
)

draw(
    text: "Emergency Department Follow-Up",
    in: NSRect(x: paperRect.minX + marginX, y: paperRect.maxY - 250, width: 760, height: 40),
    font: NSFont.systemFont(ofSize: 33, weight: .semibold),
    color: bodyColor
)
draw(
    text: "Visit Date: 04/16/2026",
    in: NSRect(x: paperRect.maxX - 350, y: paperRect.maxY - 242, width: 250, height: 28),
    font: monoFont,
    color: mutedColor,
    alignment: .right
)

horizontalRule(y: paperRect.maxY - 264, color: accentColor.withAlphaComponent(0.44), lineWidth: 2.0)

let columnGap: CGFloat = 36
let columnWidth = (paperRect.width - (marginX * 2) - columnGap) / 2
var currentY = paperRect.maxY - 314

for index in 0..<leftFields.count {
    let left = leftFields[index]
    let right = rightFields[index]

    let leftValueHeight = measure(text: left.value, width: columnWidth, font: valueFont)
    let rightValueHeight = measure(text: right.value, width: columnWidth, font: valueFont)
    let rowContentHeight = max(leftValueHeight, rightValueHeight)
    let rowHeight = max(72, rowContentHeight + 30)

    let labelY = currentY - 18
    let valueTopY = labelY - 14

    draw(
        text: left.label.uppercased(),
        in: NSRect(x: paperRect.minX + marginX, y: labelY, width: columnWidth, height: 16),
        font: labelFont,
        color: mutedColor
    )
    draw(
        text: left.value,
        in: NSRect(x: paperRect.minX + marginX, y: valueTopY - leftValueHeight, width: columnWidth, height: leftValueHeight + 4),
        font: valueFont,
        color: bodyColor
    )

    draw(
        text: right.label.uppercased(),
        in: NSRect(x: paperRect.minX + marginX + columnWidth + columnGap, y: labelY, width: columnWidth, height: 16),
        font: labelFont,
        color: mutedColor
    )
    draw(
        text: right.value,
        in: NSRect(x: paperRect.minX + marginX + columnWidth + columnGap, y: valueTopY - rightValueHeight, width: columnWidth, height: rightValueHeight + 4),
        font: valueFont,
        color: bodyColor
    )

    horizontalRule(y: currentY - rowHeight, color: lineColor)
    currentY -= (rowHeight + 12)
}

currentY -= 12

for section in sections {
    let titleHeight = measure(text: section.title.uppercased(), width: paperRect.width - (marginX * 2), font: sectionTitleFont, lineHeightMultiple: 1.0)
    let bodyHeight = measure(text: section.body, width: paperRect.width - (marginX * 2), font: bodyFont)

    let titleRect = NSRect(
        x: paperRect.minX + marginX,
        y: currentY - titleHeight,
        width: paperRect.width - (marginX * 2),
        height: titleHeight
    )
    draw(
        text: section.title.uppercased(),
        in: titleRect,
        font: sectionTitleFont,
        color: accentColor,
        lineHeightMultiple: 1.0
    )

    let bodyRect = NSRect(
        x: paperRect.minX + marginX,
        y: titleRect.minY - 10 - bodyHeight,
        width: paperRect.width - (marginX * 2),
        height: bodyHeight
    )
    draw(
        text: section.body,
        in: bodyRect,
        font: bodyFont,
        color: bodyColor
    )

    currentY = bodyRect.minY - 28
    horizontalRule(y: currentY, color: lineColor)
    currentY -= 34
}

horizontalRule(y: paperRect.minY + 190, color: lineColor, lineWidth: 1.2)
draw(
    text: "Prescribed: ibuprofen 400 mg PO q6h PRN\\nOndansetron 4 mg PO q8h PRN nausea",
    in: NSRect(x: paperRect.minX + marginX, y: paperRect.minY + 96, width: 560, height: 66),
    font: monoFont,
    color: bodyColor
)
draw(
    text: "Disposition: discharge home\\nReturn-to-work target: 04/19/2026",
    in: NSRect(x: paperRect.minX + marginX, y: paperRect.minY + 42, width: 500, height: 54),
    font: monoFont,
    color: bodyColor
)

draw(
    text: "Electronically signed",
    in: NSRect(x: paperRect.maxX - 360, y: paperRect.minY + 122, width: 250, height: 18),
    font: labelFont,
    color: mutedColor,
    alignment: .right
)
draw(
    text: "Maya Shah, MD",
    in: NSRect(x: paperRect.maxX - 360, y: paperRect.minY + 76, width: 250, height: 32),
    font: NSFont.systemFont(ofSize: 23, weight: .medium),
    color: bodyColor,
    alignment: .right
)
draw(
    text: "OpenMed Bayview",
    in: NSRect(x: paperRect.maxX - 360, y: paperRect.minY + 44, width: 250, height: 18),
    font: subtitleFont,
    color: mutedColor,
    alignment: .right
)

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

try fileManager.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
try pngData.write(to: outputURL, options: .atomic)
print(outputURL.path)
