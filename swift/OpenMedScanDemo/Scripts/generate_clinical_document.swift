import AppKit
import CoreGraphics
import Foundation

// Generates a print-ready A4 clinical discharge summary used as the demo's
// scan target. Outputs a vector PDF (for printing) and a matching PNG (for the
// bundled `SampleClinicalDocument` asset). All data is synthetic.
//
// Run:  swift Scripts/generate_clinical_document.swift

let scriptURL = URL(fileURLWithPath: #filePath)
let demoRoot = scriptURL.deletingLastPathComponent().deletingLastPathComponent()
let pdfURL = demoRoot.appending(path: "SampleClinicalDocument-A4.pdf")
let pngURL = demoRoot.appending(path: "OpenMedScanDemo/Assets.xcassets/SampleClinicalDocument.imageset/sample-clinical-document.png")

// A4 in points (72 pt/in)
let pageW: CGFloat = 595.276
let pageH: CGFloat = 841.890
let margin: CGFloat = 46
let contentW = pageW - margin * 2

let ink = NSColor(calibratedWhite: 0.10, alpha: 1)
let faint = NSColor(calibratedWhite: 0.42, alpha: 1)
let navy = NSColor(calibratedRed: 0.10, green: 0.21, blue: 0.40, alpha: 1)
let ruleC = NSColor(calibratedWhite: 0.80, alpha: 1)
let boxC = NSColor(calibratedWhite: 0.72, alpha: 1)
let stripBG = NSColor(calibratedWhite: 0.93, alpha: 1)
let divC = NSColor(calibratedWhite: 0.90, alpha: 1)

func sys(_ s: CGFloat, _ w: NSFont.Weight = .regular) -> NSFont { NSFont.systemFont(ofSize: s, weight: w) }
func mono(_ s: CGFloat, _ w: NSFont.Weight = .regular) -> NSFont { NSFont.monospacedSystemFont(ofSize: s, weight: w) }

func attr(_ s: String, _ f: NSFont, _ c: NSColor, _ align: NSTextAlignment = .left,
          lineMul: CGFloat = 1.12, kern: CGFloat = 0) -> NSAttributedString {
    let p = NSMutableParagraphStyle()
    p.alignment = align
    p.lineBreakMode = .byWordWrapping
    p.lineHeightMultiple = lineMul
    return NSAttributedString(string: s, attributes: [.font: f, .foregroundColor: c, .paragraphStyle: p, .kern: kern])
}
func measureH(_ a: NSAttributedString, _ w: CGFloat) -> CGFloat {
    ceil(a.boundingRect(with: NSSize(width: w, height: 100000),
                        options: [.usesLineFragmentOrigin, .usesFontLeading]).height)
}
func drawText(_ a: NSAttributedString, _ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat = 100000) {
    a.draw(with: NSRect(x: x, y: y, width: w, height: h),
           options: [.usesLineFragmentOrigin, .usesFontLeading])
}
func hrule(_ y: CGFloat, _ x0: CGFloat, _ x1: CGFloat, _ c: NSColor, _ lw: CGFloat = 0.8) {
    c.setStroke()
    let p = NSBezierPath()
    p.lineWidth = lw
    p.move(to: NSPoint(x: x0, y: y))
    p.line(to: NSPoint(x: x1, y: y))
    p.stroke()
}
func fillRect(_ r: NSRect, _ c: NSColor) { c.setFill(); NSBezierPath(rect: r).fill() }

func renderDocument() {
    var y: CGFloat = margin

    // confidential strip
    let stripH: CGFloat = 15
    fillRect(NSRect(x: margin, y: y, width: contentW, height: stripH), stripBG)
    drawText(attr("CONFIDENTIAL   ·   PROTECTED HEALTH INFORMATION (PHI)   ·   DO NOT DISTRIBUTE",
                  mono(6.6, .medium), faint, kern: 0.3), margin + 8, y + 3.6, contentW - 80, 12)
    drawText(attr("PAGE 1 OF 1", mono(6.6, .medium), faint, .right), margin, y + 3.6, contentW - 8, 12)
    y += stripH + 14

    // letterhead
    let logo = NSRect(x: margin, y: y, width: 34, height: 34)
    fillRect(logo, navy)
    NSColor.white.setFill()
    NSBezierPath(rect: NSRect(x: logo.midX - 2.6, y: logo.minY + 7, width: 5.2, height: 20)).fill()
    NSBezierPath(rect: NSRect(x: logo.minX + 7, y: logo.midY - 2.6, width: 20, height: 5.2)).fill()

    let lx = logo.maxX + 12
    drawText(attr("Summit Ridge Regional Medical Center", sys(15, .bold), navy), lx, y - 2, contentW - 44, 20)
    drawText(attr("EMERGENCY DEPARTMENT", sys(7.6, .semibold), faint, kern: 1.4), lx, y + 16, contentW - 44, 12)
    drawText(attr("1200 Cedar Hollow Parkway, Aurora, CO 80012", sys(8, .regular), faint), lx, y + 26, contentW - 44, 12)
    drawText(attr("Main (303) 555-0170     Fax (303) 555-0171     NPI 1043920175", sys(8, .regular), faint), lx, y + 35, contentW - 44, 12)
    y += 52
    hrule(y, margin, pageW - margin, navy, 1.4)
    y += 11

    // title row
    drawText(attr("DISCHARGE SUMMARY", sys(12, .bold), ink, kern: 0.8), margin, y, contentW - 150, 18)
    drawText(attr("Visit Date   06/01/2026", mono(8.6, .regular), faint, .right), margin, y + 2, contentW, 16)
    y += 21

    // demographics box
    struct DRow { let l: (String, String); let r: (String, String)?; let full: Bool }
    let rows: [DRow] = [
        DRow(l: ("Patient Name", "Whitfield, Jordan A."), r: nil, full: true),
        DRow(l: ("DOB", "07/22/1984"), r: ("Age / Sex", "41 / Female"), full: false),
        DRow(l: ("MRN", "SRMC-7741920"), r: ("SSN", "900-21-7755"), full: false),
        DRow(l: ("Encounter #", "ENC-20260601-3382"), r: ("Account #", "ACC-55810394"), full: false),
        DRow(l: ("Phone", "(720) 555-0148"), r: ("Preferred Language", "English"), full: false),
        DRow(l: ("Email", "jordan.whitfield@samplemail.test"), r: nil, full: true),
        DRow(l: ("Address", "4471 Lantern Ridge Ct, Aurora, CO 80016"), r: nil, full: true),
        DRow(l: ("Insurance", "Summit Health PPO, Member ID SHP-66201845, Group 4471"), r: nil, full: true),
        DRow(l: ("Emergency Contact", "Dana Whitfield (spouse), (720) 555-0193"), r: nil, full: true),
        DRow(l: ("PCP", "Priya Nandakumar, MD"), r: ("PCP NPI", "1841992307"), full: false),
        DRow(l: ("Employer", "Front Range Logistics"), r: nil, full: true),
    ]

    func cell(_ label: String, _ value: String, _ x: CGFloat, _ w: CGFloat, _ ry: CGFloat, _ labelW: CGFloat) {
        drawText(attr(label.uppercased(), sys(6.6, .semibold), faint, kern: 0.4), x, ry + 3.4, labelW, 11)
        drawText(attr(value, mono(8.4, .regular), ink), x + labelW, ry + 1.6, w - labelW, 12)
    }

    let boxTop = y
    let rowH: CGFloat = 14
    let halfW = contentW / 2
    var ry = y + 6
    for (i, row) in rows.enumerated() {
        let cx = margin + 9
        if row.full {
            cell(row.l.0, row.l.1, cx, contentW - 18, ry, 104)
        } else {
            cell(row.l.0, row.l.1, cx, halfW - 14, ry, 66)
            if let r = row.r { cell(r.0, r.1, margin + halfW + 5, halfW - 14, ry, 96) }
        }
        ry += rowH
        if i < rows.count - 1 { hrule(ry - 0.5, margin + 6, pageW - margin - 6, divC, 0.5) }
    }
    let boxH = (ry + 6) - boxTop
    boxC.setStroke()
    let bp = NSBezierPath(rect: NSRect(x: margin, y: boxTop, width: contentW, height: boxH))
    bp.lineWidth = 0.9
    bp.stroke()
    y = boxTop + boxH + 11

    // sections
    func section(_ title: String, _ body: NSAttributedString, after: CGFloat = 5.8) {
        fillRect(NSRect(x: margin, y: y + 1.5, width: 3, height: 8.5), navy)
        drawText(attr(title.uppercased(), sys(8.8, .bold), navy, kern: 0.6), margin + 9, y, contentW - 9, 12)
        y += 12.5
        let h = measureH(body, contentW)
        drawText(body, margin, y, contentW, h)
        y += h + after
    }
    func body(_ s: String) -> NSAttributedString { attr(s, sys(8.6, .regular), ink, lineMul: 1.16) }

    section("Chief Complaint",
            body("Frontal headache, dizziness, and nausea for three days, worse this morning with poor oral intake."))
    section("History of Present Illness",
            body("Ms. Whitfield is a 41-year-old woman with type 2 diabetes mellitus, essential hypertension, chronic migraine, hyperlipidemia, and GERD who presents after urgent care hydration. She reports three days of worsening frontal headache with photophobia, dizziness, and nausea. Home fingerstick glucose this morning was 212 mg/dL. She received one liter of normal saline and ondansetron 4 mg at urgent care yesterday with partial relief."))
    section("Allergies",
            body("Penicillin (pruritic rash in childhood). Sulfonamides (hives). No known latex allergy."))
    section("Home Medications",
            attr("•  Metformin 1000 mg PO twice daily\n•  Lisinopril 20 mg PO once daily\n•  Atorvastatin 40 mg PO nightly\n•  Sumatriptan 50 mg PO as needed for migraine\n•  Ondansetron 4 mg PO every 8 hours as needed for nausea",
                 mono(8.2, .regular), ink, lineMul: 1.34))
    section("Vitals",
            attr("BP 158/94 mmHg     HR 98 bpm     Temp 98.4 F     SpO2 98% on room air     POC glucose 212 mg/dL",
                 mono(8.2, .regular), ink, lineMul: 1.2))
    section("Assessment",
            body("Migraine flare with mild dehydration and hyperglycemia. Neurologic exam non-focal, with low concern for an acute intracranial process given a stable exam and improvement after hydration."))
    section("Plan",
            attr("1.  Oral hydration and ibuprofen 400 mg every 6 hours as needed.\n2.  Resume home medications; recheck fasting glucose with PCP.\n3.  PCP follow-up within 48 hours and neurology within two weeks.",
                 sys(8.6, .regular), ink, lineMul: 1.32))
    section("Return Precautions",
            body("Return immediately for chest pain, repeated vomiting, syncope, confusion, focal weakness, or the worst headache of life."))
    section("Disposition",
            body("Discharged home in stable condition. Work status: may return to work on 06/03/2026."))

    // footer
    let fY = pageH - margin - 48
    hrule(fY, margin, pageW - margin, ruleC, 0.8)
    drawText(attr("Document ID   SRMC-DS-20260601-3382       SummitChart EHR v8.2", mono(7, .regular), faint),
             margin, fY + 7, contentW * 0.62, 12)
    // deterministic barcode
    var seed: UInt64 = 9173
    var cx = margin
    let by = fY + 22
    for _ in 0..<48 {
        seed = seed &* 6364136223846793005 &+ 1442695040888963407
        let w = CGFloat(1 + Int((seed >> 33) % 3))
        let isBar = (seed >> 29) % 5 != 0
        if isBar { fillRect(NSRect(x: cx, y: by, width: w, height: 15), ink) }
        cx += w + 1.2
        if cx > margin + 170 { break }
    }
    drawText(attr("ELECTRONICALLY SIGNED", sys(6.6, .semibold), faint, .right, kern: 0.6), margin, fY + 7, contentW, 10)
    drawText(attr("Maya Shah, MD", sys(10.5, .semibold), ink, .right), margin, fY + 17, contentW, 14)
    drawText(attr("Emergency Medicine     06/01/2026 14:22 MT", sys(7.4, .regular), faint, .right), margin, fY + 31, contentW, 12)
}

// ---- render PDF ----
let pdfData = NSMutableData()
guard let consumer = CGDataConsumer(data: pdfData as CFMutableData) else { fatalError("consumer") }
var media = CGRect(x: 0, y: 0, width: pageW, height: pageH)
guard let pdf = CGContext(consumer: consumer, mediaBox: &media, nil) else { fatalError("pdf ctx") }
pdf.beginPDFPage(nil)
pdf.saveGState()
pdf.translateBy(x: 0, y: pageH)
pdf.scaleBy(x: 1, y: -1)
let gctx = NSGraphicsContext(cgContext: pdf, flipped: true)
NSGraphicsContext.saveGraphicsState()
NSGraphicsContext.current = gctx
renderDocument()
NSGraphicsContext.restoreGraphicsState()
pdf.restoreGState()
pdf.endPDFPage()
pdf.closePDF()
try pdfData.write(to: pdfURL, options: .atomic)

// ---- rasterize matching PNG for the in-app asset ----
guard let provider = CGDataProvider(url: pdfURL as CFURL),
      let doc = CGPDFDocument(provider),
      let page = doc.page(at: 1) else { fatalError("pdf read back") }
let box = page.getBoxRect(.mediaBox)
let dpi: CGFloat = 230
let scale = dpi / 72.0
let pxW = Int((box.width * scale).rounded())
let pxH = Int((box.height * scale).rounded())
let cs = CGColorSpaceCreateDeviceRGB()
guard let bmp = CGContext(data: nil, width: pxW, height: pxH, bitsPerComponent: 8, bytesPerRow: 0,
                          space: cs, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { fatalError("bmp") }
bmp.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
bmp.fill(CGRect(x: 0, y: 0, width: pxW, height: pxH))
bmp.interpolationQuality = .high
bmp.scaleBy(x: scale, y: scale)
bmp.drawPDFPage(page)
guard let cg = bmp.makeImage() else { fatalError("makeImage") }
let rep = NSBitmapImageRep(cgImage: cg)
guard let png = rep.representation(using: .png, properties: [:]) else { fatalError("png") }
try png.write(to: pngURL, options: .atomic)

print("PDF:", pdfURL.path)
print("PNG:", pngURL.path, "(\(pxW)x\(pxH))")
