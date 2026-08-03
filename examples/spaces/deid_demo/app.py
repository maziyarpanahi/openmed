#!/usr/bin/env python3
"""Multilingual Gradio demo for synthetic OpenMed de-identification.

The module is import-safe: it does not build the UI, load a model, access the
network, or start a server until the corresponding function is called.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openmed import deidentify

try:
    from .samples import SAMPLES
except ImportError:  # Running ``python app.py`` from the Space directory.
    from samples import SAMPLES


GRADIO_INSTALL_HINT = (
    "Gradio is required for this demo. Install the pinned requirements with: "
    "pip install -r requirements.txt"
)

MODEL_IDS = {
    "zh": "OpenMed/OpenMed-PII-Chinese-BigMed-Large-278M-v1",
    "hi": "OpenMed/OpenMed-PII-Hindi-SuperClinical-Small-44M-v1",
    "en": "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1",
}
LANGUAGE_OVERRIDES = ("auto", "zh", "hi", "en")

DISCLAIMERS = {
    "en": (
        "Synthetic data only. Never paste real patient data or protected "
        "health information (PHI)."
    ),
    "zh": "仅限合成数据。切勿粘贴真实患者数据或受保护的健康信息（PHI）。",
    "hi": (
        "केवल कृत्रिम डेटा। कभी भी वास्तविक रोगी डेटा या संरक्षित स्वास्थ्य "
        "जानकारी (PHI) न चिपकाएँ।"
    ),
}
DISCLAIMER_MARKDOWN = "\n\n".join(
    f"> **{language.upper()}** — {disclaimer}"
    for language, disclaimer in DISCLAIMERS.items()
)

UI_STRINGS = {
    "en": {
        "title": "OpenMed multilingual de-identification",
        "intro": "Try a fabricated note in Chinese, Hindi, English, or Hinglish.",
        "input": "Synthetic input",
        "override": "Input language",
        "auto": "Auto-detect",
        "zh": "Chinese",
        "hi": "Hindi",
        "en": "English / Hinglish",
        "submit": "De-identify",
        "output": "Masked output",
        "route": "Detected route",
        "empty": "Enter synthetic text to continue.",
        "error": "The note could not be de-identified. Please try again.",
    },
    "zh": {
        "title": "OpenMed 多语言去标识化",
        "intro": "试用中文、印地语、英语或 Hinglish 的完全虚构病历。",
        "input": "合成文本输入",
        "override": "输入语言",
        "auto": "自动检测",
        "zh": "中文",
        "hi": "印地语",
        "en": "英语 / Hinglish",
        "submit": "去标识化",
        "output": "遮蔽后的输出",
        "route": "检测到的路由",
        "empty": "请输入合成文本后继续。",
        "error": "无法完成去标识化，请重试。",
    },
    "hi": {
        "title": "OpenMed बहुभाषी डी-आइडेंटिफिकेशन",
        "intro": "चीनी, हिन्दी, अंग्रेज़ी या हिंग्लिश में काल्पनिक नोट आज़माएँ।",
        "input": "कृत्रिम इनपुट",
        "override": "इनपुट भाषा",
        "auto": "स्वतः पहचानें",
        "zh": "चीनी",
        "hi": "हिन्दी",
        "en": "अंग्रेज़ी / हिंग्लिश",
        "submit": "पहचान हटाएँ",
        "output": "मास्क किया गया आउटपुट",
        "route": "चुना गया रूट",
        "empty": "आगे बढ़ने के लिए कृत्रिम टेक्स्ट दर्ज करें।",
        "error": "नोट से पहचान नहीं हटाई जा सकी। कृपया फिर प्रयास करें।",
    },
}


@dataclass(frozen=True)
class ModelRoute:
    """Resolved input language and OpenMed model configuration."""

    language: str
    model_id: str
    openmed_language: str


@dataclass(frozen=True)
class DemoResult:
    """Safe UI output for one de-identification request."""

    deidentified_text: str
    route: ModelRoute


Deidentifier = Callable[..., Any]


def _is_han(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
    )


def _is_devanagari(character: str) -> bool:
    codepoint = ord(character)
    return 0x0900 <= codepoint <= 0x097F or 0xA8E0 <= codepoint <= 0xA8FF


def detect_script(text: str) -> str:
    """Detect the routing language from Unicode script ranges.

    Han takes precedence for mixed-script text, followed by Devanagari.
    Latin, Hinglish, digits, punctuation, and empty text use the English path.
    """

    if any(_is_han(character) for character in text):
        return "zh"
    if any(_is_devanagari(character) for character in text):
        return "hi"
    return "en"


def resolve_model_route(text: str, language_override: str = "auto") -> ModelRoute:
    """Resolve a model route, honoring an explicit language override."""

    if language_override not in LANGUAGE_OVERRIDES:
        raise ValueError(
            f"Unsupported language override {language_override!r}; "
            f"choose one of {LANGUAGE_OVERRIDES}."
        )
    language = detect_script(text) if language_override == "auto" else language_override
    return ModelRoute(
        language=language,
        model_id=MODEL_IDS[language],
        openmed_language=language,
    )


def _sample_recognizer(language: str) -> dict[str, Any]:
    """Build deterministic rules for the fabricated sample identifiers."""

    return {
        "case_sensitive": False,
        "deny": {
            "terms": [
                {"term": term, "label": label}
                for term, label in SAMPLES[language].identifiers
            ]
        },
    }


def run_deidentification(
    text: str,
    language_override: str = "auto",
    *,
    deidentifier: Deidentifier = deidentify,
) -> DemoResult:
    """De-identify text without logging, analytics, caching, or persistence."""

    if not text.strip():
        raise ValueError("Synthetic input must not be empty.")
    route = resolve_model_route(text, language_override)
    result = deidentifier(
        text,
        method="mask",
        model_name=route.model_id,
        lang=route.openmed_language,
        policy="strict_no_leak",
        use_safety_sweep=True,
        custom_recognizer=_sample_recognizer(route.language),
        keep_mapping=False,
        audit=False,
        cache_results=False,
    )
    return DemoResult(deidentified_text=result.deidentified_text, route=route)


def localized_strings(locale: str) -> dict[str, str]:
    """Return UI copy for a supported display locale."""

    if locale not in UI_STRINGS:
        raise ValueError(f"Unsupported UI locale: {locale!r}")
    return UI_STRINGS[locale]


def language_choices(locale: str) -> list[tuple[str, str]]:
    """Return localized language-override labels and stable values."""

    strings = localized_strings(locale)
    return [(strings[value], value) for value in LANGUAGE_OVERRIDES]


def route_status(route: ModelRoute, locale: str) -> str:
    """Format safe route metadata without including user input."""

    strings = localized_strings(locale)
    return f"**{strings['route']}:** {strings[route.language]} · `{route.model_id}`"


def build_demo() -> Any:
    """Build the localized Gradio UI without loading models or launching."""

    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - exercised via import shim
        raise SystemExit(GRADIO_INSTALL_HINT) from exc

    initial = localized_strings("en")

    def _on_locale_change(locale: str) -> tuple[Any, ...]:
        strings = localized_strings(locale)
        return (
            gr.Markdown(value=f"# {strings['title']}\n\n{strings['intro']}"),
            gr.Textbox(label=strings["input"], value=SAMPLES[locale].text, lines=8),
            gr.Radio(
                label=strings["override"],
                choices=language_choices(locale),
                value="auto",
            ),
            gr.Button(value=strings["submit"], variant="primary"),
            gr.Textbox(label=strings["output"], lines=8, interactive=False),
            gr.Markdown(value=f"**{strings['route']}:** —"),
        )

    def _on_submit(text: str, override: str, locale: str) -> tuple[str, str]:
        strings = localized_strings(locale)
        if not text.strip():
            return strings["empty"], f"**{strings['route']}:** —"
        try:
            result = run_deidentification(text, override)
        except Exception:  # Keep raw input and backend details out of logs/UI.
            return strings["error"], f"**{strings['route']}:** —"
        return result.deidentified_text, route_status(result.route, locale)

    with gr.Blocks(
        title="OpenMed multilingual de-identification",
        analytics_enabled=False,
    ) as demo:
        intro = gr.Markdown(f"# {initial['title']}\n\n{initial['intro']}")
        gr.Markdown(DISCLAIMER_MARKDOWN)
        ui_locale = gr.Radio(
            choices=[("English", "en"), ("简体中文", "zh"), ("हिन्दी", "hi")],
            value="en",
            label="UI language / 界面语言 / इंटरफ़ेस भाषा",
        )
        text_in = gr.Textbox(
            label=initial["input"],
            value=SAMPLES["en"].text,
            lines=8,
        )
        override = gr.Radio(
            label=initial["override"],
            choices=language_choices("en"),
            value="auto",
        )
        submit = gr.Button(initial["submit"], variant="primary")
        text_out = gr.Textbox(
            label=initial["output"],
            lines=8,
            interactive=False,
        )
        status = gr.Markdown(f"**{initial['route']}:** —")

        ui_locale.change(
            _on_locale_change,
            inputs=ui_locale,
            outputs=[intro, text_in, override, submit, text_out, status],
        )
        submit.click(
            _on_submit,
            inputs=[text_in, override, ui_locale],
            outputs=[text_out, status],
        )

    return demo


def main() -> None:
    """Launch the app locally or in a Gradio Space."""

    build_demo().launch(show_error=False)


if __name__ == "__main__":
    main()
