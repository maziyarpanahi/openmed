/* =============================================================================
   OpenMed website — interactions.
   Theme toggle · animated PHI demo · tabbed code playground · FAQ · copy.
   ============================================================================= */

document.addEventListener("DOMContentLoaded", () => {
    initYear();
    initTheme();
    initPHIDemo();
    initCodePlayground();
    initModelFilter();
    initFAQ();
    initCopyInstall();
    initScrollSpy();
    initGitHubStars();
});

/* ---------- GitHub stars (live, with cache + graceful fallback) --------- */
function initGitHubStars() {
    const el = document.querySelector("[data-gh-count]");
    if (!el) return;

    const REPO = "maziyarpanahi/openmed";
    const KEY = "openmed-gh-stars";
    const TTL = 60 * 60 * 1000; // 1 hour

    function fmt(n) {
        if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, "") + "k";
        return String(n);
    }

    try {
        const raw = localStorage.getItem(KEY);
        if (raw) {
            const { count, at } = JSON.parse(raw);
            if (count && Date.now() - at < TTL) {
                el.textContent = fmt(count);
                return;
            }
        }
    } catch (e) {}

    fetch(`https://api.github.com/repos/${REPO}`)
        .then(r => r.ok ? r.json() : null)
        .then(d => {
            if (!d || typeof d.stargazers_count !== "number") return;
            el.textContent = fmt(d.stargazers_count);
            try { localStorage.setItem(KEY, JSON.stringify({ count: d.stargazers_count, at: Date.now() })); } catch (e) {}
        })
        .catch(() => { el.textContent = "★"; });
}

/* ---------- Year --------------------------------------------------------- */
function initYear() {
    const el = document.getElementById("year");
    if (el) el.textContent = new Date().getFullYear();
}

/* ---------- Theme -------------------------------------------------------- */
function initTheme() {
    const root = document.documentElement;
    const btn = document.getElementById("themeToggle");
    const stored = (() => { try { return localStorage.getItem("openmed-theme"); } catch (e) { return null; } })();
    const initial = stored || "light";
    applyTheme(initial);

    if (btn) {
        btn.addEventListener("click", () => {
            const next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
            applyTheme(next);
            try { localStorage.setItem("openmed-theme", next); } catch (e) {}
        });
    }

    function applyTheme(t) {
        root.setAttribute("data-theme", t);
        if (btn) btn.innerHTML = t === "dark" ? sunSVG() : moonSVG();
    }
}

function sunSVG() {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>`;
}
function moonSVG() {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>`;
}

/* ---------- PHI Demo (cycles through 6 languages OpenMed supports) ------ */
const PHI_SAMPLES = [
    {
        lang: "en",
        parts: [
            { t: "Patient " },
            { t: "John Doe", k: "NAME" },
            { t: ",\nDOB " },
            { t: "05/15/1990", k: "DATE" },
            { t: ",\nSSN " },
            { t: "123-45-6789", k: "ID" },
            { t: ",\nphone " },
            { t: "+1 (415) 555-0132", k: "PHONE" },
            { t: ",\ndiagnosed with " },
            { t: "acute pancreatitis", k: "DISEASE" },
            { t: "." },
        ],
    },
    {
        lang: "es",
        parts: [
            { t: "Paciente " },
            { t: "Maria Garcia Lopez", k: "NAME" },
            { t: ",\nFN " },
            { t: "15/03/1985", k: "DATE" },
            { t: ",\nDNI " },
            { t: "12345678Z", k: "ID" },
            { t: ",\ntel " },
            { t: "+34 612 345 678", k: "PHONE" },
            { t: ",\ndiagnosticado con " },
            { t: "leucemia mieloide crónica", k: "DISEASE" },
            { t: "." },
        ],
    },
    {
        lang: "fr",
        parts: [
            { t: "Patient " },
            { t: "Pierre Martin", k: "NAME" },
            { t: ",\nné le " },
            { t: "12/04/1982", k: "DATE" },
            { t: ",\nNIR " },
            { t: "1820475123456", k: "ID" },
            { t: ",\ntél " },
            { t: "+33 6 12 34 56 78", k: "PHONE" },
            { t: ",\ndiagnostiqué avec " },
            { t: "sclérose en plaques", k: "DISEASE" },
            { t: "." },
        ],
    },
    {
        lang: "de",
        parts: [
            { t: "Patientin " },
            { t: "Anna Schmidt", k: "NAME" },
            { t: ",\ngeb. " },
            { t: "22.08.1978", k: "DATE" },
            { t: ",\nSteuer-ID " },
            { t: "12345678901", k: "ID" },
            { t: ",\nTel " },
            { t: "+49 30 12345678", k: "PHONE" },
            { t: ",\ndiagnostiziert mit " },
            { t: "Typ-2-Diabetes", k: "DISEASE" },
            { t: "." },
        ],
    },
    {
        lang: "it",
        parts: [
            { t: "Paziente " },
            { t: "Luca Bianchi", k: "NAME" },
            { t: ",\nnato il " },
            { t: "03/11/1990", k: "DATE" },
            { t: ",\nCF " },
            { t: "BNCLCU90S03H501Z", k: "ID" },
            { t: ",\ntel " },
            { t: "+39 320 123 4567", k: "PHONE" },
            { t: ",\ndiagnosticato con " },
            { t: "morbo di Parkinson", k: "DISEASE" },
            { t: "." },
        ],
    },
    {
        lang: "nl",
        parts: [
            { t: "Patiënt " },
            { t: "Jan de Vries", k: "NAME" },
            { t: ",\ngeboren " },
            { t: "18-07-1985", k: "DATE" },
            { t: ",\nBSN " },
            { t: "123456782", k: "ID" },
            { t: ",\ntel " },
            { t: "+31 6 12345678", k: "PHONE" },
            { t: ",\ngediagnosticeerd met " },
            { t: "multiple sclerose", k: "DISEASE" },
            { t: "." },
        ],
    },
];

function initPHIDemo() {
    const card = document.querySelector("[data-phi]");
    if (!card) return;

    const bodyEl = card.querySelector("[data-phi-body]");
    const countEl = card.querySelector(".phi-count");
    const langEl = card.querySelector("[data-phi-lang]");
    const fnEl = card.querySelector("[data-phi-fn]");
    if (!bodyEl) return;

    function render(sample) {
        bodyEl.innerHTML = sample.parts.map(p => {
            if (!p.k) return `<span>${escapeHTML(p.t)}</span>`;
            return `<span class="phi-token" data-k="${p.k}">${escapeHTML(p.t)}<span class="sup">${p.k}</span></span>`;
        }).join("");
        if (langEl) langEl.textContent = `"${sample.lang}"`;
        if (fnEl) fnEl.textContent = `pii_${sample.lang}.detect() · on-device`;
    }

    let sampleIdx = 0;
    let step = 0;
    render(PHI_SAMPLES[sampleIdx]);
    const total = 5;

    function tick() {
        const reveal = Math.min(step, total);
        const tokens = bodyEl.querySelectorAll(".phi-token[data-k]");
        tokens.forEach((t, i) => t.classList.toggle("revealed", i < reveal));
        if (countEl) countEl.textContent = `${reveal}/${total} entities`;

        if (step >= total) {
            // fully revealed: pause one tick, then advance to next language
            step = 0;
            sampleIdx = (sampleIdx + 1) % PHI_SAMPLES.length;
            setTimeout(() => render(PHI_SAMPLES[sampleIdx]), 100);
        } else {
            step += 1;
        }
    }
    setInterval(tick, 1400);
}

function escapeHTML(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

/* ---------- Code playground --------------------------------------------- */
const CODE_SNIPPETS = {
    Python: `from openmed import OpenMedConfig, analyze_text, deidentify

config = OpenMedConfig.from_profile("prod")

text = "Patient John Doe diagnosed with chronic myeloid leukemia."
result = analyze_text(text, model_name="disease_detection_superclinical", config=config)

for entity in result.entities:
    print(entity.label, entity.text, entity.confidence)

deid = deidentify(text, method="mask", config=config)
print(deid.deidentified_text)`,
    "PII API": `from openmed import extract_pii, deidentify

text = """Paciente: Maria Garcia Lopez
Fecha de nacimiento: 15/03/1985
DNI: 12345678Z"""

result = extract_pii(text, lang="es", use_smart_merging=True)

for entity in result.entities:
    print(entity.label, entity.text, entity.confidence)

deid = deidentify(text, lang="es", method="mask")
print(deid.deidentified_text)`,
    "Apple MLX": `# uv pip install "openmed[mlx]"
from openmed.mlx import MLXPipeline

pipe = MLXPipeline.from_pretrained(
    "OpenMed/OpenMed-NER-DiseaseDetect-BioMed-335M",
    dtype="float16",
)

entities = pipe("Patient presents with acute pancreatitis and hypertension.")
for e in entities:
    print(f"{e.label:12} {e.text:40} conf={e.confidence:.2f}")`,
    Swift: `import OpenMedKit

let pipeline = try await OpenMedPipeline(
    model: .diseaseDetect335M,
    runtime: .coreML
)

let text = "Patient presents with acute pancreatitis."
let result = try await pipeline.analyze(text)

for entity in result.entities {
    print(entity.label, entity.text, entity.confidence)
}`,
};

const PY_KEYWORDS = /\b(from|import|for|in|print|if|else|return|def|class|with|as|try|except|True|False|None)\b/g;
const SWIFT_KEYWORDS = /\b(import|let|var|try|await|struct|class|func|if|else|return|guard)\b/g;

function highlight(src, lang) {
    const kwRe = lang === "Swift" ? SWIFT_KEYWORDS : PY_KEYWORDS;
    return src
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|"""[\s\S]*?""")/g, '<span class="str">$1</span>')
        .replace(/(#[^\n]*)/g, '<span class="cm">$1</span>')
        .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="num">$1</span>')
        .replace(kwRe, '<span class="kw">$1</span>');
}

function initCodePlayground() {
    const panel = document.querySelector("[data-playground]");
    if (!panel) return;

    const tabsWrap = panel.querySelector(".code-tabs");
    const body = panel.querySelector(".code-body");
    const copyBtn = panel.querySelector(".code-copy");

    const names = Object.keys(CODE_SNIPPETS);
    tabsWrap.innerHTML = names.map((n, i) =>
        `<button class="code-tab${i === 0 ? " active" : ""}" data-tab="${n}">${n}</button>`
    ).join("");

    let active = names[0];
    render();

    tabsWrap.addEventListener("click", (e) => {
        const btn = e.target.closest(".code-tab");
        if (!btn) return;
        active = btn.dataset.tab;
        tabsWrap.querySelectorAll(".code-tab").forEach(b => b.classList.toggle("active", b === btn));
        render();
    });

    if (copyBtn) {
        copyBtn.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(CODE_SNIPPETS[active]);
                copyBtn.textContent = "copied ✓";
                copyBtn.classList.add("copied");
                setTimeout(() => {
                    copyBtn.textContent = "copy";
                    copyBtn.classList.remove("copied");
                }, 1400);
            } catch (err) {
                copyBtn.textContent = "copy failed";
                setTimeout(() => { copyBtn.textContent = "copy"; }, 1400);
            }
        });
    }

    function render() {
        body.innerHTML = highlight(CODE_SNIPPETS[active], active);
    }
}

/* ---------- Models filter ------------------------------------------------ */
function initModelFilter() {
    const chips = document.querySelectorAll("[data-model-filter] .chip");
    const cells = document.querySelectorAll("[data-model-grid] .model-cell");
    if (!chips.length) return;

    chips.forEach(chip => {
        chip.addEventListener("click", () => {
            chips.forEach(c => c.classList.toggle("active", c === chip));
            const f = chip.dataset.filter;
            cells.forEach(cell => {
                const tags = (cell.dataset.tags || "").toLowerCase();
                const match = f === "All" || tags.includes(f.toLowerCase());
                cell.style.display = match ? "" : "none";
            });
        });
    });
}

/* ---------- FAQ ---------------------------------------------------------- */
function initFAQ() {
    const items = document.querySelectorAll(".faq-item");
    items.forEach((item, i) => {
        if (i === 0) item.classList.add("open");
        item.addEventListener("click", () => {
            const wasOpen = item.classList.contains("open");
            items.forEach(x => x.classList.remove("open"));
            if (!wasOpen) item.classList.add("open");
        });
    });
}

/* ---------- Copy install command ---------------------------------------- */
function initCopyInstall() {
    const btn = document.querySelector("[data-copy-install]");
    if (!btn) return;
    btn.addEventListener("click", async (e) => {
        e.preventDefault();
        const cmd = btn.dataset.copyInstall || btn.textContent.trim();
        try {
            await navigator.clipboard.writeText(cmd);
            const prev = btn.querySelector(".cmd").textContent;
            btn.querySelector(".cmd").textContent = "copied ✓";
            setTimeout(() => { btn.querySelector(".cmd").textContent = prev; }, 1400);
        } catch (err) {}
    });
}

/* ---------- Scroll-spy nav underline ------------------------------------ */
function initScrollSpy() {
    const links = document.querySelectorAll(".nav-links a[href^='#']");
    if (!links.length) return;

    const map = new Map();
    links.forEach(a => {
        const id = a.getAttribute("href").slice(1);
        const section = document.getElementById(id);
        if (section) map.set(section, a);
    });

    const obs = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                links.forEach(l => l.classList.remove("active"));
                map.get(entry.target)?.classList.add("active");
            }
        });
    }, { rootMargin: "-40% 0px -55% 0px" });

    map.forEach((_, section) => obs.observe(section));
}
