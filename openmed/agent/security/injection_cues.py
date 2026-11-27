"""Draft lexical cue packs; fluent contributor review is recorded in fixtures.

All expressions operate on the injection guard's existing normalized text.
They identify narrow imperative constructions, not language or clinical truth.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class _CuePack:
    language: str
    instruction_override: tuple[str, ...]
    tool_name_spoofing: tuple[str, ...]
    data_exfiltration: tuple[str, ...]


def _expand_cue_accents(expression: str) -> str:
    # The existing offset mapper normalizes each codepoint independently.
    # Preserve its spans while matching canonically decomposed Latin accents
    # as well as the controlled composed/accentless character alternatives.
    def expand(match: re.Match[str]) -> str:
        base, accented = match.groups()
        decomposed = unicodedata.normalize("NFD", accented)
        if decomposed.startswith(base) and all(
            unicodedata.category(char) == "Mn" for char in decomposed[1:]
        ):
            return "(?:" + match.group() + "|" + re.escape(decomposed) + ")"
        return match.group()

    return re.sub(r"\[([a-z])([^\x00-\x7f])\]", expand, expression)


# Include Devanagari combining marks in word boundaries, but permit danda
# punctuation immediately after a complete cue.
_HI_START = r"(?<![\w\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962-\u0963])"
_HI_END = r"(?![\w\u0900-\u0903\u093a-\u094f\u0951-\u0957\u0962-\u0963])"

_CUE_PACKS = (
    _CuePack(
        "es",
        instruction_override=(
            r"\b(?:ignora|ignore|ignoren|omite|omita|olvida|olvide|descarta|descarte)\s+"
            r"(?:todas\s+)?(?:las\s+)?(?:instrucciones|reglas|indicaciones)\s+"
            r"(?:anteriores|previas|del\s+sistema)\b",
            r"\b(?:nuevas|verdaderas|reales)\s+instrucciones\s+"
            r"(?:del\s+sistema|del\s+desarrollador)\b",
        ),
        tool_name_spoofing=(
            r"\b(?:llama|llame|ejecuta|ejecute|invoca|invoque|usa|use)\s+"
            r"(?:(?:a|la|el)\s+){0,2}(?:herramienta|funci[oó]n)\s*[:=]?\s*"
            r"(?:openmed|mcp)[\w.-]*\b",
        ),
        data_exfiltration=(
            r"\b(?:exporta|exporte|env[ií]a|env[ií]e|sube|suba|transmite|transmita)\s+"
            r"(?:todos\s+)?(?:los\s+)?(?:datos|registros|historias\s+cl[ií]nicas)\s+"
            r"(?:(?:de|del)\s+)?(?:pacientes?|cl[ií]nicos|privados|m[eé]dicos)\b",
            r"\b(?:muestra|muestre|revela|revele|imprime|imprima)\s+(?:el\s+)?"
            r"(?:prompt|mensaje)\s+(?:oculto|secreto)\b",
        ),
    ),
    _CuePack(
        "fr",
        instruction_override=(
            r"\b(?:ignore|ignorez|oublie|oubliez|contourne|contournez)\s+"
            r"(?:toutes\s+)?(?:les\s+)?(?:instructions|r[eè]gles|consignes)\s+"
            r"(?:pr[eé]c[eé]dentes|ant[eé]rieures|du\s+syst[eè]me)\b",
            r"\b(?:nouvelles|vraies|v[eé]ritables)\s+instructions\s+"
            r"(?:du\s+syst[eè]me|du\s+d[eé]veloppeur)\b",
        ),
        tool_name_spoofing=(
            r"\b(?:appelle|appelez|ex[eé]cute|ex[eé]cutez|invoque|invoquez|"
            r"utilise|utilisez)\s+(?:l['’]|la\s+|le\s+)?(?:outil|fonction)\s*[:=]?\s*"
            r"(?:openmed|mcp)[\w.-]*\b",
        ),
        data_exfiltration=(
            r"\b(?:exporte|exportez|envoie|envoyez|transmets|transmettez|"
            r"t[eé]l[eé]verse)\s+(?:tous\s+)?(?:les\s+)?(?:donn[eé]es|dossiers)\s+"
            r"(?:(?:de|des|du)\s+)?(?:patients?|cliniques|priv[eé]s)\b",
            r"\b(?:montre|montrez|r[eé]v[eè]le|r[eé]v[eè]lez|affiche|affichez)\s+"
            r"(?:le\s+)?(?:prompt|message)\s+(?:syst[eè]me\s+)?(?:cach[eé]|secret)\b",
        ),
    ),
    _CuePack(
        "de",
        instruction_override=(
            r"\b(?:ignoriere|ignorieren|vergiss|umgehe)\s+(?:alle\s+)?(?:die\s+)?"
            r"(?:vorherigen|fr[uü]heren|bisherigen)\s+(?:anweisungen|regeln)\b",
            r"\b(?:neue|echte|wahre)\s+(?:systemanweisungen|entwickleranweisungen)\b",
        ),
        tool_name_spoofing=(
            r"\b(?:rufe|f[uü]hre|nutze|verwende)\s+(?:das\s+|die\s+|den\s+)?"
            r"(?:tool|werkzeug|funktion)\s*[:=]?\s*(?:openmed|mcp)[\w.-]*\b",
        ),
        data_exfiltration=(
            r"\b(?:exportiere|exportieren|sende|senden|[uü]bertrage)\s+"
            r"(?:alle\s+)?(?:die\s+)?(?:patientendaten|patientenakten|"
            r"medizinischen\s+daten|privaten\s+daten)\b",
            r"\b(?:zeige|offenbare|drucke)\s+(?:den\s+)?(?:geheimen|versteckten)\s+"
            r"(?:systemprompt|prompt)\b",
        ),
    ),
    _CuePack(
        "pt",
        instruction_override=(
            r"\b(?:ignore|ignora|esque[cç]a|esquece|desconsidere|contorne)\s+"
            r"(?:todas\s+)?(?:as\s+)?(?:instru[cç][oõ]es|regras|orienta[cç][oõ]es)\s+"
            r"(?:anteriores|pr[eé]vias|do\s+sistema)\b",
            r"\b(?:novas|verdadeiras|reais)\s+instru[cç][oõ]es\s+"
            r"(?:do\s+sistema|do\s+desenvolvedor)\b",
        ),
        tool_name_spoofing=(
            r"\b(?:chame|chama|execute|executa|invoque|invoca|use|usa)\s+"
            r"(?:a\s+|o\s+)?(?:ferramenta|fun[cç][aã]o)\s*[:=]?\s*"
            r"(?:openmed|mcp)[\w.-]*\b",
        ),
        data_exfiltration=(
            r"\b(?:exporte|exporta|envie|envia|transmita|transmite)\s+"
            r"(?:todos\s+)?(?:os\s+)?(?:dados|registros|prontu[aá]rios)\s+"
            r"(?:(?:de|do|dos)\s+)?(?:pacientes?|cl[ií]nicos|privados)\b",
            r"\b(?:mostre|mostra|revele|revela|imprima)\s+(?:o\s+)?"
            r"(?:prompt|mensagem)\s+(?:oculto|oculta|secreto|secreta)\b",
        ),
    ),
    _CuePack(
        "zh",
        instruction_override=(
            r"(?:忽略|无视|無視|忘记|忘記|绕过|繞過)\s*(?:所有\s*)?"
            r"(?:之前|以前|先前|系统|系統)(?:的)?(?:所有)?\s*(?:指令|指示|规则|規則)",
            r"(?:新的?|真正的?)\s*(?:系统|系統|开发者|開發者)(?:的)?\s*(?:指令|指示)",
        ),
        tool_name_spoofing=(
            r"(?:调用|調用|执行|執行|使用)\s*(?:工具|函数|函數)\s*[:：=]?\s*"
            r"(?:openmed|mcp)[\w.-]*",
        ),
        data_exfiltration=(
            r"(?:导出|導出|发送|發送|上传|上傳|传输|傳輸)\s*(?:所有\s*)?"
            r"(?:患者|病人)(?:的)?\s*(?:数据|數據|记录|記錄|病历|病歷)",
            r"(?:显示|顯示|泄露|洩露|输出|輸出)\s*(?:隐藏|隱藏|秘密)(?:的)?\s*"
            r"(?:系统|系統)?(?:提示词|提示詞|指令)",
        ),
    ),
    _CuePack(
        "hi",
        instruction_override=(
            _HI_START + r"(?:पिछले|पिछली|पूर्व|पुराने|pichle|pichhle|purane|"
            r"system\s+ke|सिस्टम\s+के)\s+(?:सभी\s+|sabhi\s+)?"
            r"(?:निर्देश(?:ों)?|नियम(?:ों)?|nirdesh(?:on)?|instructions|rules)\s+"
            r"(?:(?:को|ko)\s+)?(?:अनदेखा\s+करो|नज़रअंदाज़\s+करो|"
            r"नजरअंदाज\s+करो|भूल\s+जाओ|ignore\s+(?:karo|करो)|"
            r"andekha\s+karo|nazarandaz\s+karo|bhool\s+jao)" + _HI_END,
            _HI_START + r"(?:नए|नये|असली)\s+(?:सिस्टम|प्रणाली|डेवलपर)\s+निर्देश" + _HI_END,
        ),
        tool_name_spoofing=(
            _HI_START + r"(?:टूल|फ़ंक्शन|फंक्शन|tool|function)\s*[:=]?\s*"
            r"(?:openmed|mcp)[\w.-]*\s+(?:चलाओ|चलाएँ|चलाएं|कॉल\s+करो|"
            r"chalao|call\s+(?:karo|करो))" + _HI_END,
            _HI_START
            + r"(?:openmed|mcp)[\w.-]*\s+(?:टूल|tool)\s+(?:चलाओ|chalao)"
            + _HI_END,
        ),
        data_exfiltration=(
            _HI_START + r"(?:मरीजों|मरीज़ों|रोगियों|patients?|rogiyon|marizon)\s+"
            r"(?:(?:के|ke)\s+)?(?:सभी\s+|sabhi\s+)?"
            r"(?:डेटा|रिकॉर्ड|रिकार्ड|data|records?)\s+(?:(?:को|ko)\s+)?"
            r"(?:भेजो|भेजें|निर्यात\s+करो|अपलोड\s+करो|bhejo|"
            r"export\s+(?:karo|करो)|upload\s+karo)" + _HI_END,
            _HI_START + r"(?:छिपे\s+हुए|गुप्त)\s+(?:सिस्टम\s+|प्रणाली\s+के\s+)?"
            r"(?:निर्देश|प्रॉम्प्ट|prompt)\s+(?:दिखाओ|बताओ|प्रकट\s+करो)" + _HI_END,
        ),
    ),
)
