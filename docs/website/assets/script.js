/* OpenMed.life progressive enhancement.
   No analytics, runtime API fetches, or clinical-text network requests. */

document.documentElement.classList.add("js");

document.addEventListener("DOMContentLoaded", () => {
    initTheme();
    initMobileMenu();
    initDemoMotion();
    initTabs();
    initCopyControls();
    initModelFilters();
    initFAQ();
    initScrollSpy();
    initYear();
});

/* Theme ------------------------------------------------------------------ */

function initTheme() {
    const root = document.documentElement;
    const button = document.getElementById("themeToggle");
    const themeColors = [...document.querySelectorAll("[data-theme-color]")];
    const systemDark = window.matchMedia("(prefers-color-scheme: dark)");
    const preferences = ["light", "dark"];
    let preference = readThemePreference()
        || (systemDark.matches ? "dark" : "light");

    if (!preferences.includes(preference)) {
        preference = systemDark.matches ? "dark" : "light";
    }

    function applyTheme() {
        root.setAttribute("data-theme", preference);
        root.dataset.themePreference = preference;

        const currentIndex = preferences.indexOf(preference);
        const next = preferences[(currentIndex + 1) % preferences.length];
        const label = button?.querySelector("[data-theme-label]");
        const icon = button?.querySelector("[data-theme-icon]");

        if (label) label.textContent = titleCase(preference);
        if (icon) icon.dataset.themeState = preference;
        if (button) {
            button.setAttribute(
                "aria-label",
                `Color theme: ${preference}. Activate for ${next}.`,
            );
        }
        const selectedColor = preference === "dark" ? "#0B0E13" : "#F4F7F8";
        themeColors.forEach(meta => {
            meta.content = selectedColor;
        });
    }

    button?.addEventListener("click", () => {
        const currentIndex = preferences.indexOf(preference);
        preference = preferences[(currentIndex + 1) % preferences.length];
        try {
            localStorage.setItem("openmed-theme", preference);
        } catch {
            // Storage can be unavailable in private or restricted contexts.
        }
        applyTheme();
    });

    applyTheme();
}

function readThemePreference() {
    try {
        const value = localStorage.getItem("openmed-theme");
        return value === "light" || value === "dark" ? value : null;
    } catch {
        return null;
    }
}

function titleCase(value) {
    return value.charAt(0).toUpperCase() + value.slice(1);
}

function addMediaListener(media, handler) {
    if (media.addEventListener) {
        media.addEventListener("change", handler);
    } else {
        media.addListener(handler);
    }
}

/* Mobile navigation ------------------------------------------------------ */

function initMobileMenu() {
    const header = document.querySelector("[data-header]");
    const toggle = document.getElementById("navToggle");
    const nav = document.getElementById("primaryNav");
    const mobile = window.matchMedia("(max-width: 900px)");
    if (!header || !toggle || !nav) return;

    function setOpen(open, returnFocus = false) {
        header.classList.toggle("menu-open", open);
        toggle.setAttribute("aria-expanded", String(open));
        toggle.setAttribute("aria-label", open ? "Close navigation" : "Open navigation");

        if (open) {
            window.requestAnimationFrame(() => nav.querySelector("a")?.focus());
        } else if (returnFocus) {
            toggle.focus();
        }
    }

    toggle.addEventListener("click", () => {
        setOpen(!header.classList.contains("menu-open"));
    });

    nav.addEventListener("click", event => {
        if (event.target.closest("a")) setOpen(false);
    });

    document.addEventListener("keydown", event => {
        if (event.key === "Escape" && header.classList.contains("menu-open")) {
            setOpen(false, true);
        }
    });

    document.addEventListener("pointerdown", event => {
        if (
            mobile.matches
            && header.classList.contains("menu-open")
            && !header.contains(event.target)
        ) {
            setOpen(false);
        }
    });

    addMediaListener(mobile, event => {
        if (!event.matches) setOpen(false);
    });
}

/* Hero word and synthetic PHI demo -------------------------------------- */

const ROTATING_WORDS = [
    "hardware.",
    "laptop.",
    "iPhone.",
    "GPU server.",
    "air-gapped box.",
];

const PHI_SAMPLES = [
    {
        lang: "en",
        parts: [
            { text: "Patient " },
            { text: "Casey Example", label: "NAME" },
            { text: ", MRN " },
            { text: "00123", label: "ID" },
            { text: ", DOB " },
            { text: "03/14/1962", label: "DATE" },
            { text: ", contact " },
            { text: "(312) 555-0142", label: "PHONE" },
            { text: ", admitted to " },
            { text: "Example Hospital", label: "HOSPITAL" },
            { text: "." },
        ],
    },
    {
        lang: "fr",
        parts: [
            { text: "Mme " },
            { text: "Camille Exemple", label: "NAME" },
            { text: ", née le " },
            { text: "12/07/1958", label: "DATE" },
            { text: ", NIR " },
            { text: "2 58 07 75 116 001 23", label: "NIR" },
            { text: ", suivie à l’" },
            { text: "Hôpital Exemple", label: "HOSPITAL" },
            { text: "." },
        ],
    },
    {
        lang: "de",
        parts: [
            { text: "Patient " },
            { text: "Jonas Beispiel", label: "NAME" },
            { text: ", geb. " },
            { text: "21.11.1970", label: "DATE" },
            { text: ", Steuer-ID " },
            { text: "57 144 261 809", label: "STEUER_ID" },
            { text: ", behandelt in der " },
            { text: "Beispielklinik", label: "HOSPITAL" },
            { text: "." },
        ],
    },
    {
        lang: "tr",
        parts: [
            { text: "Hasta " },
            { text: "Ayşe Örnek", label: "NAME" },
            { text: ", TCKN " },
            { text: "10000000146", label: "TCKN" },
            { text: ", tel " },
            { text: "+90 532 555 0182", label: "PHONE" },
            { text: ", " },
            { text: "Örnek Hastanesi", label: "HOSPITAL" },
            { text: "." },
        ],
    },
];

function initDemoMotion() {
    const word = document.querySelector("[data-rotating-word]");
    const demo = document.querySelector("[data-phi-demo]");
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (!word && !demo) return;

    let wordIndex = 0;
    let sampleIndex = 0;
    let phase = 0;
    let wordInterval = null;
    let phiInterval = null;
    let swapTimeout = null;

    function stop() {
        window.clearInterval(wordInterval);
        window.clearInterval(phiInterval);
        window.clearTimeout(swapTimeout);
        wordInterval = null;
        phiInterval = null;
        swapTimeout = null;
    }

    function renderStatic() {
        if (word) {
            word.textContent = ROTATING_WORDS[0];
            word.classList.remove("is-swapping");
        }
        if (demo) renderPHI(demo, PHI_SAMPLES[0], Number.POSITIVE_INFINITY);
    }

    function start() {
        stop();
        if (reduceMotion.matches) {
            renderStatic();
            return;
        }

        if (word) {
            wordInterval = window.setInterval(() => {
                word.classList.add("is-swapping");
                swapTimeout = window.setTimeout(() => {
                    wordIndex = (wordIndex + 1) % ROTATING_WORDS.length;
                    word.textContent = ROTATING_WORDS[wordIndex];
                    word.classList.remove("is-swapping");
                }, 190);
            }, 2500);
        }

        if (demo) {
            renderPHI(demo, PHI_SAMPLES[sampleIndex], phase);
            phiInterval = window.setInterval(() => {
                const entityCount = PHI_SAMPLES[sampleIndex].parts.filter(
                    part => part.label,
                ).length;
                phase += 1;
                if (phase > entityCount + 2) {
                    phase = 0;
                    sampleIndex = (sampleIndex + 1) % PHI_SAMPLES.length;
                }
                renderPHI(demo, PHI_SAMPLES[sampleIndex], phase);
            }, 850);
        }
    }

    addMediaListener(reduceMotion, start);
    start();
}

function renderPHI(demo, sample, phase) {
    const body = demo.querySelector("[data-phi-body]");
    const count = demo.querySelector("[data-phi-count]");
    const lang = demo.querySelector("[data-phi-lang]");
    if (!body) return;

    const entityCount = sample.parts.filter(part => part.label).length;
    const masked = !Number.isFinite(phase) || phase > entityCount;
    const fragment = document.createDocumentFragment();
    let entityIndex = 0;

    sample.parts.forEach(part => {
        if (!part.label) {
            fragment.append(document.createTextNode(part.text));
            return;
        }

        entityIndex += 1;
        if (masked || entityIndex <= phase) {
            const mark = document.createElement("mark");
            mark.dataset.kind = part.label;
            mark.textContent = masked ? `[${part.label}]` : part.text;
            fragment.append(mark);
        } else {
            fragment.append(document.createTextNode(part.text));
        }
    });

    body.replaceChildren(fragment);
    if (count) {
        const found = masked ? entityCount : Math.min(phase, entityCount);
        count.textContent = masked
            ? `${entityCount}/${entityCount} · masked`
            : `${found}/${entityCount} entities`;
    }
    if (lang) lang.textContent = `"${sample.lang}"`;
}

/* Tabs ------------------------------------------------------------------- */

function initTabs() {
    document.querySelectorAll("[data-code-example]").forEach(group => {
        const tabs = [...group.querySelectorAll('[role="tab"]')];
        const panels = [...group.querySelectorAll('[role="tabpanel"]')];
        if (!tabs.length || !panels.length) return;

        function activate(tab, moveFocus = false) {
            tabs.forEach(candidate => {
                const selected = candidate === tab;
                candidate.setAttribute("aria-selected", String(selected));
                candidate.tabIndex = selected ? 0 : -1;
            });
            panels.forEach(panel => {
                panel.hidden = panel.id !== tab.getAttribute("aria-controls");
            });
            if (moveFocus) tab.focus();
        }

        tabs.forEach((tab, index) => {
            tab.addEventListener("click", () => activate(tab));
            tab.addEventListener("keydown", event => {
                let nextIndex = null;
                if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
                if (event.key === "ArrowLeft") {
                    nextIndex = (index - 1 + tabs.length) % tabs.length;
                }
                if (event.key === "Home") nextIndex = 0;
                if (event.key === "End") nextIndex = tabs.length - 1;
                if (nextIndex === null) return;
                event.preventDefault();
                activate(tabs[nextIndex], true);
            });
        });

        activate(tabs.find(tab => tab.getAttribute("aria-selected") === "true") || tabs[0]);
    });
}

/* Copy ------------------------------------------------------------------- */

function initCopyControls() {
    document.querySelectorAll("[data-copy-text]").forEach(button => {
        button.addEventListener("click", async () => {
            const ok = await copyText(button.dataset.copyText || "");
            flashCopyResult(button, ok);
        });
    });

    document.querySelectorAll("[data-copy-active]").forEach(button => {
        button.addEventListener("click", async () => {
            const group = button.closest("[data-code-example]");
            const activeTab = group?.querySelector('[role="tab"][aria-selected="true"]');
            const panelId = activeTab?.getAttribute("aria-controls");
            const code = panelId
                ? document.getElementById(panelId)?.querySelector("code")?.textContent
                : "";
            const ok = await copyText(code || "");
            flashCopyResult(button, ok);
        });
    });
}

async function copyText(value) {
    if (!value) return false;
    try {
        await navigator.clipboard.writeText(value);
        return true;
    } catch {
        const textarea = document.createElement("textarea");
        textarea.value = value;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.opacity = "0";
        document.body.append(textarea);
        textarea.select();
        let copied = false;
        try {
            copied = document.execCommand("copy");
        } catch {
            copied = false;
        }
        textarea.remove();
        return copied;
    }
}

function flashCopyResult(button, ok) {
    const status = document.getElementById("copyStatus");
    const originalLabel = button.getAttribute("aria-label");
    const originalText = button.textContent;
    const glyph = button.querySelector(".copy-glyph");
    const message = ok ? "Copied to clipboard." : "Copy failed.";

    if (glyph) {
        glyph.classList.toggle("is-success", ok);
        glyph.classList.toggle("is-error", !ok);
    } else {
        button.textContent = ok ? "copied" : "copy failed";
    }
    button.setAttribute("aria-label", message);
    if (status) status.textContent = message;

    window.setTimeout(() => {
        if (glyph) {
            glyph.classList.remove("is-success", "is-error");
        } else {
            button.textContent = originalText;
        }
        if (originalLabel) {
            button.setAttribute("aria-label", originalLabel);
        } else {
            button.removeAttribute("aria-label");
        }
    }, 1600);
}

/* Model filters ---------------------------------------------------------- */

function initModelFilters() {
    const controls = document.querySelector(".model-filters");
    const cards = [...document.querySelectorAll(".model-grid [data-category]")];
    const status = document.querySelector("[data-filter-status]");
    if (!controls || !cards.length) return;

    const buttons = [...controls.querySelectorAll("button[data-filter]")];
    buttons.forEach(button => {
        button.dataset.label = button.textContent.trim();
    });

    function activate(button, moveFocus = false) {
        const filter = button.dataset.filter;
        let visible = 0;
        buttons.forEach(candidate => {
            const active = candidate === button;
            candidate.setAttribute("aria-pressed", String(active));
            candidate.textContent = candidate.dataset.label;
        });
        cards.forEach(card => {
            const show = filter === "all" || card.dataset.category === filter;
            card.hidden = !show;
            if (show) visible += 1;
        });
        if (status) {
            status.textContent = `Showing ${visible} model example${visible === 1 ? "" : "s"}.`;
        }
        if (moveFocus) button.focus();
    }

    buttons.forEach((button, index) => {
        button.addEventListener("click", () => activate(button));
        button.addEventListener("keydown", event => {
            let nextIndex = null;
            if (event.key === "ArrowRight") nextIndex = (index + 1) % buttons.length;
            if (event.key === "ArrowLeft") {
                nextIndex = (index - 1 + buttons.length) % buttons.length;
            }
            if (event.key === "Home") nextIndex = 0;
            if (event.key === "End") nextIndex = buttons.length - 1;
            if (nextIndex === null) return;
            event.preventDefault();
            activate(buttons[nextIndex], true);
        });
    });
}

/* FAQ -------------------------------------------------------------------- */

function initFAQ() {
    const list = document.querySelector("[data-faq-list]");
    if (!list) return;

    const buttons = [...list.querySelectorAll("button[aria-controls]")];

    function setExpanded(button, expanded) {
        const panel = document.getElementById(button.getAttribute("aria-controls"));
        button.setAttribute("aria-expanded", String(expanded));
        if (panel) panel.hidden = !expanded;
    }

    buttons.forEach((button, index) => {
        setExpanded(button, index === 0);
        button.addEventListener("click", () => {
            const shouldOpen = button.getAttribute("aria-expanded") !== "true";
            buttons.forEach(candidate => setExpanded(candidate, false));
            if (shouldOpen) setExpanded(button, true);
        });
        button.addEventListener("keydown", event => {
            let nextIndex = null;
            if (event.key === "ArrowDown") nextIndex = (index + 1) % buttons.length;
            if (event.key === "ArrowUp") {
                nextIndex = (index - 1 + buttons.length) % buttons.length;
            }
            if (event.key === "Home") nextIndex = 0;
            if (event.key === "End") nextIndex = buttons.length - 1;
            if (nextIndex === null) return;
            event.preventDefault();
            buttons[nextIndex].focus();
        });
    });
}

/* Scroll state ----------------------------------------------------------- */

function initScrollSpy() {
    if (!("IntersectionObserver" in window)) return;

    const links = [...document.querySelectorAll('.primary-nav a[href^="#"]')];
    const targets = new Map(
        links
            .map(link => [link.getAttribute("href").slice(1), link])
            .filter(([id]) => document.getElementById(id)),
    );
    if (!targets.size) return;

    const visible = new Map();
    const observer = new IntersectionObserver(
        entries => {
            entries.forEach(entry => {
                visible.set(entry.target.id, entry.isIntersecting ? entry.intersectionRatio : 0);
            });
            const active = [...visible.entries()]
                .filter(([, ratio]) => ratio > 0)
                .sort((a, b) => b[1] - a[1])[0]?.[0];
            links.forEach(link => {
                if (link.getAttribute("href") === `#${active}`) {
                    link.setAttribute("aria-current", "location");
                } else {
                    link.removeAttribute("aria-current");
                }
            });
        },
        {
            rootMargin: "-20% 0px -60% 0px",
            threshold: [0, 0.1, 0.5, 1],
        },
    );

    targets.forEach((_, id) => observer.observe(document.getElementById(id)));
}

/* Static fallback details ------------------------------------------------ */

function initYear() {
    const year = document.getElementById("year");
    if (year) year.textContent = String(new Date().getFullYear());
}
