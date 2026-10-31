# groq_bot.py — drop-in, token-optimized prompts for richer EN/RU answers & formulas (free-tier friendly)
import os, re, html, json, threading
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict

# ----- PDF reader (pypdf preferred) -----
try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    from PyPDF2 import PdfReader  # fallback

from rank_bm25 import BM25Okapi
from groq import Groq

# ================== Config ==================
DEFAULT_MODEL        = os.environ.get("GROQ_MODEL","qwen/qwen3.8-27b")
FALLBACK_MODELS      = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b")
RU_GENERIC_FALLBACK  = os.environ.get("RU_GENERIC_FALLBACK", "1") == "1"
STOP_MARK            = "<END>"

# KB paths (override via env if needed)
FAQ_EN_PATH = os.environ.get("FAQ_EN_PATH", "kb/parsed_faqs_en.json")
FAQ_RU_PATH = os.environ.get("FAQ_RU_PATH", "kb/parsed_faqs_ru.json")

# ================== Lang detect =============
_CYR = re.compile(r"[\u0400-\u04FF]")
def detect_lang(text: str) -> str:
    """RU if any Cyrillic, else EN."""
    return "ru" if _CYR.search(text or "") else "en"

_MATH_RE = re.compile(r"(\\begin\{equation\}|\\\[|\\\(|\$\$|∑|∫|∂|λ|ε|≈|→|←|∇|∞|√|ℏ|±)")
def _has_math(text: str) -> bool:
    return bool(_MATH_RE.search(text or ""))

def _budget(lang: str, context_text: str, user_cap: int) -> int:
    """
    More headroom for formulas, still safe for free tier:
    EN base≈520; RU +30%; math +20%; hard cap 640.
    """
    base = 520
    if lang == "ru":
        base = int(base * 1.30)
    if _has_math(context_text):
        base = int(base * 1.20)
    return min(user_cap, base, 640)

# ================== Retrieval ===============
@dataclass
class Chunk:
    text: str
    start_page: int
    end_page: int
    source: str  # 'pdf' | 'faq-en' | 'faq-ru'

class DreamRetriever:
    """
    Unified retriever over:
      - PDF pages (chunked by words)
      - EN/RU FAQ JSONs (each FAQ becomes a chunk)
    """
    def __init__(
        self,
        pdf_path: str,
        faq_en_path: Optional[str] = None,
        faq_ru_path: Optional[str] = None,
        max_words_per_chunk: int = 220,
    ):
        self.pdf_path = pdf_path
        self.faq_en_path = faq_en_path or FAQ_EN_PATH
        self.faq_ru_path = faq_ru_path or FAQ_RU_PATH
        self.max_words_per_chunk = max_words_per_chunk

        self.chunks: List[Chunk] = []
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized: List[List[str]] = []

    @staticmethod
    def _clean(s: str) -> str:
        s = s.replace("\r", "")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{2,}", "\n", s)
        return s.strip()

    # ---------- PDF ----------
    def _read_pdf(self) -> List[Tuple[int, str]]:
        pages: List[Tuple[int, str]] = []
        if not self.pdf_path or not os.path.exists(self.pdf_path):
            return pages
        with open(self.pdf_path, "rb") as f:
            pdf = PdfReader(f)
            for i, p in enumerate(getattr(pdf, "pages", [])):
                t = (p.extract_text() or "")
                pages.append((i + 1, self._clean(t)))
        return pages

    def _chunk_pages(self, pages: List[Tuple[int, str]]) -> List[Chunk]:
        buf: List[str] = []
        st_pg: Optional[int] = None
        chunks: List[Chunk] = []
        words_in_buf = 0

        def flush(end_pg: int) -> None:
            nonlocal buf, st_pg, words_in_buf
            if not buf:
                return
            text = DreamRetriever._clean(" ".join(buf))
            chunks.append(Chunk(text=text, start_page=int(st_pg or end_pg), end_page=end_pg, source="pdf"))
            buf, st_pg, words_in_buf = [], None, 0

        for pg, txt in pages:
            for para in [p for p in txt.split("\n") if p.strip()]:
                w = para.split()
                if st_pg is None:
                    st_pg = pg
                if words_in_buf and words_in_buf + len(w) > self.max_words_per_chunk:
                    flush(pg)
                    st_pg = pg
                buf.append(para)
                words_in_buf += len(w)

        if st_pg is not None:
            flush(pages[-1][0] if pages else 1)
        return chunks

    # ---------- FAQ JSONs ----------
    @staticmethod
    def _safe_load_json(path: str) -> Dict:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _faq_chunks_from_json(self, path: str, src_tag: str) -> List[Chunk]:
        payload = self._safe_load_json(path)
        faqs = payload.get("faqs", []) if isinstance(payload, dict) else []
        out: List[Chunk] = []
        for f in faqs:
            q = (f.get("question") or "").strip()
            a = (f.get("answer") or "").strip()
            if not (q or a):
                continue
            txt = self._clean(f"Q: {q}\nA: {a}")
            out.append(Chunk(text=txt, start_page=-1, end_page=-1, source=src_tag))
        return out

    # ---------- Build unified index ----------
    @staticmethod
    def _tokenize(s: str) -> List[str]:
        return re.findall(r"[a-zA-Zа-яА-Я0-9]+", s.lower())

    def build(self) -> None:
        self.chunks = []
        # PDF
        pages = self._read_pdf()
        self.chunks.extend(self._chunk_pages(pages))
        # FAQs
        self.chunks.extend(self._faq_chunks_from_json(self.faq_en_path, "faq-en"))
        self.chunks.extend(self._faq_chunks_from_json(self.faq_ru_path, "faq-ru"))
        # BM25
        self._tokenized = [self._tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def search(self, query: str, k: int = 6, min_rel: float = 0.28) -> List[Chunk]:
        """BM25 with intro-page penalty for PDF; relevance threshold; NumPy-safe."""
        if not self._bm25:
            self.build()

        toks = self._tokenize(query)
        if not toks:
            return []

        scores_arr = self._bm25.get_scores(toks)
        scores = list(map(float, getattr(scores_arr, "tolist", lambda: scores_arr)()))
        if not scores:
            return []

        max_s = max(scores)
        if max_s <= 0:
            return []

        # Penalize early (intro) pages slightly — only for PDF chunks
        def penalized(i: int) -> float:
            if self.chunks[i].source != "pdf":
                return scores[i]
            mid_pg = (self.chunks[i].start_page + self.chunks[i].end_page) / 2.0
            return scores[i] * (0.85 if mid_pg <= 3 else 1.0)

        ranked = sorted(range(len(scores)), key=penalized, reverse=True)

        picks: List[Chunk] = []
        for i in ranked:
            if scores[i] < (min_rel * max_s):
                break
            picks.append(self.chunks[i])
            if len(picks) >= k:
                break
        return picks

# ================== Answerer =================
class GroqAnswerer:
    def __init__(self, model: str = DEFAULT_MODEL, temperature: float = 0.2, max_tokens: int = 800):
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set GROQ_API_KEY in environment.")
        self.client = Groq(api_key=api_key)
        self.model = model.strip()
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    # ------ private: build system (Context embedded) ------
    # 2026 Framework Summary — always injected into the system prompt so the
    # model knows the CURRENT state of DREAM, regardless of which Context
    # chunks get retrieved. The Context below may contain material from the
    # 2023 manuscript; the 2026 Summary takes precedence on conflicts.
    SUMMARY_EN = (
        "DREAM = Dimensional Resonant Emergent Attractors in Manifold — a "
        "projection-first physics framework. 4D observables are finite-resolution "
        "pushforwards of a higher-dimensional source; 10D is a parsimonious modeling "
        "choice, not an empirically observed dimensionality.\n"
        "Axioms A1-A6: A1 Existence(MM); A2 Projection(Kernel); A3 Retention Law (S2); "
        "A4 Constants are projection-fixed; A5 Topology<->classes; A6 Finite resolution => regimes.\n"
        "S2 Retention Law: S(W) = exp[-(W/lambda)^D] — the Weibull survival function. "
        "lambda = lifetime scale; D = aging regime (D<1 sub-exponential/natural decay; D>1 "
        "super-exponential/extraction). Integral E[T] = lambda*Gamma(1+1/D) recovers the "
        "physical timescale. S2 is the LOCAL law; the GLOBAL curve is piecewise (T7.1).\n"
        "Theorems: T1 Topological Quantization (Confirmed); T2 Symmetry Projection (Partial); "
        "T3 Coherence Threshold (Confirmed); T4 Structure Focusing & CPI (Partial); "
        "T5 Dimensional Flow to Homogeneity (Partial); T6 Cross-domain Ranked-Spectrum Ratio "
        "Law (Partial); T7 Interference (Aggregate Superposition of Local S2) (Partially "
        "Supported; simple-average mixture refuted at Level-2); T7.1 Multi-Regime S2 "
        "(Piecewise Composition) — Confirmed, guardrailed, 25 real datasets, binomial "
        "p=3.4e-18 vs single-S2 null. T7.1 adds NO new postulate (refines T7).\n"
        "AICc Guardrail: 'S2 present' requires passing all 5 guardrails — fit R^2>0.90, "
        "parameter stability CV<30%, null-destruction ratio >2x, identifiability, AICc. "
        "Current pass rate ~84.6%; the 15.4% failures (trend-dominated economic aggregates, "
        "power-law seismic decay) are genuinely non-S2.\n"
        "Tests registry: 338 tests (Oct 2026). Verdicts: S2_LOSES 210, S2_WINS 64, S2_TIES 39, "
        "S2_DUST_WINS 9, plus domain-specific tags (INTEGRAL_IS_LIFETIME, WEAR_OUT_AGING_REGIMES, "
        "WATER_IS_INFORMATION, THREE_FAUCET_TYPES, MULTIDIM_WEIBULL_AFT, etc.).\n"
        "Lifecycle program (T19-T31): thirteen tests turned 'what is the kernel's water?' "
        "into a quantitative lifecycle science. Information is born, ages, and dies by a "
        "Weibull mortality law. Validated on solar X-rays, macroeconomics, weather, tides, "
        "15,873 Colombian transformers, 17,939 Canadian transformers, 10,000 CNC machines.\n"
        "2026 Falsification status — SURVIVES: stretched-exponential form across 5 domains "
        "(r^2>0.96); T7.1 piecewise refinement; S2 beats P(k)+filter on cosmology "
        "(Delta r^2 +0.09 SDSS, +0.14 DESI). FALSIFIED: D invariance across probes; "
        "'S2 is intrinsic law of MM'; snake-pit rescue; z_depth kernel interpretation. "
        "Honest reading: S2 is a useful, sometimes superior, sometimes equivalent summary; "
        "it is not a unique signature of cosmological structure.\n"
        "Predictions status: S2 Retention Law (Multidomain Survival); T7.1 (Confirmed); "
        "S3 Cosmology & Fractal Spectrum (Partially confirmed, strong support to S2); "
        "S5 Strong Lensing Invariants (Early); S4 Vacuum-Energy Modulation (Open); "
        "S1 ALP Non-harmonic Ladder (Relegated / Null).\n"
        "Minimal-Extension Principle (Non-Augmentation Under Explanatory Pressure): "
        "DREAM explanations must be constructed from the existing axioms. A new phenomenon "
        "does NOT license a new hidden entity, force, coupling, dimension, kernel form, "
        "or free parameter. If the existing framework cannot account for an observation, "
        "the discrepancy is RETAINED as a limitation or falsification pressure, not "
        "repaired by ad hoc extension. 'If 10D works, don't use 40. If S2 works, don't "
        "invent a second decay mechanism. If the framework fails, record the failure.'\n"
        "Pixel Tracker: live daily scout for Configuration-Space Pixel Tests — "
        "the strongest surviving claim. Tests include Spectral Pixel (saturating), "
        "Correlation Pixel (saturating).\n"
        "Cross-domain validations (original + 2026 program): Sweden FTT, China gaming ban, "
        "Montréal Protocol, Volcker Rule, plus solar X-rays, transformers, CNC machines, "
        "weather, tides."
    )

    SUMMARY_RU = (
        "DREAM = Dimensional Resonant Emergent Attractors in Manifold — проекционно-первичная "
        "физическая теория. 4D-наблюдаемые суть push-forward проекции конечного разрешения "
        "из источника большей размерности; 10D — экономное модельное допущение, не "
        "эмпирически измеренная размерность.\n"
        "Аксиомы A1-A6: A1 Существование (MM); A2 Проекция (Ядро); A3 Закон удержания (S2); "
        "A4 Константы фиксированы проекцией; A5 Топология<->классы; A6 Конечное разрешение => режимы.\n"
        "Закон удержания S2: S(W) = exp[-(W/lambda)^D] — функция выживания Вейбулла. "
        "lambda = масштаб времени жизни; D = режим старения (D<1 доэкспоненциальное/естественное "
        "затухание; D>1 суперэкспоненциальное извлечение). Интеграл E[T] = lambda*Gamma(1+1/D) "
        "восстанавливает физический масштаб времени. S2 — ЛОКАЛЬНЫЙ закон; глобальная кривая "
        "кусочная (T7.1).\n"
        "Теоремы: T1 Топологическое квантование (Подтверждена); T2 Проекция симметрии (Частично); "
        "T3 Порог когерентности (Закон удержания) (Подтверждена); T4 Фокусировка структуры & CPI "
        "(Частично); T5 Размерностный поток к однородности (Частично); T6 Кросс-доменный закон "
        "ранжированного спектрального отношения (Частично); T7 Интерференция (Агрегатная "
        "суперпозиция локальных S2) (Частично; простое усреднение опровергнуто на Level-2); "
        "T7.1 Мульти-режимная S2 (Кусочная композиция) — Подтверждена, с guardrail, 25 реальных "
        "датасетов, биномиальное p=3.4e-18 против single-S2 null. T7.1 НЕ вводит новых постулатов "
        "(уточняет T7).\n"
        "AICc Guardrail: 'S2 присутствует' требует прохождения всех 5 охранных проверок — "
        "качество фит R^2>0.90, стабильность параметров CV<30%, ratio разрушения null >2x, "
        "идентифицируемость, AICc. Текущая доля прохождения ~84.6%; 15.4% неудач "
        "(тренд-доминированные экономические агрегаты, степенной сейсмический распад) — "
        "истинно не-S2.\n"
        "Реестр тестов: 338 тестов (октябрь 2026). Вердикты: S2_LOSES 210, S2_WINS 64, "
        "S2_TIES 39, S2_DUST_WINS 9, плюс домен-специфичные теги (INTEGRAL_IS_LIFETIME, "
        "WEAR_OUT_AGING_REGIMES, WATER_IS_INFORMATION, THREE_FAUCET_TYPES, MULTIDIM_WEIBULL_AFT "
        "и др.).\n"
        "Программа жизненного цикла (T19-T31): тринадцать тестов превратили вопрос 'что такое "
        "вода ядра?' в количественную науку о жизненном цикле. Информация рождается, стареет "
        "и умирает по закону смертности Вейбулла. Валидировано на солнечных рентгеновских "
        "вспышках, макроэкономике, погоде, приливах, 15 873 колумбийских трансформаторах, "
        "17 939 канадских трансформаторах, 10 000 станках с ЧПУ.\n"
        "Статус фальсификации 2026 — ВЫЖИВАЕТ: stretched-exponential форма в 5 доменах "
        "(r^2>0.96); T7.1 кусочное уточнение; S2 превосходит P(k)+фильтр на космологии "
        "(Delta r^2 +0.09 SDSS, +0.14 DESI). ОПРОВЕРГНУТО: инвариантность D между пробами; "
        "'S2 — внутренний закон MM'; snake-pit rescue; z_depth интерпретация ядра. "
        "Честное чтение: S2 — полезное, иногда превосходящее, иногда эквивалентное резюме; "
        "не уникальная сигнатура космологической структуры.\n"
        "Статус предсказаний: S2 Закон удержания (Мультидоменное выживание); T7.1 "
        "(Подтверждена); S3 Космология & Фрактальный спектр (Частично подтверждено, сильная "
        "поддержка S2); S5 Инварианты сильного линзирования (Раннее); S4 Модуляция энергии "
        "вакуума (Открыто); S1 Негармоническая лестница ALP (Религировано / Null).\n"
        "Принцип минимального расширения (Не-дополнение под объяснительным давлением): "
        "объяснения DREAM должны строиться из существующих аксиом. Новое явление НЕ даёт "
        "права вводить новую скрытую сущность, силу, связь, размерность, форму ядра или "
        "свободный параметр. Если существующая теория не объясняет наблюдение, расхождение "
        "СОХРАНЯЕТСЯ как ограничение или фальсификационное давление, а не устраняется ad hoc "
        "расширением. 'Если работает 10D — не используй 40. Если работает S2 — не выдумывай "
        "второй механизм распада. Если теория не справляется — зафиксируй неудачу.'\n"
        "Пиксельный трекер: живой ежедневный разведчик Configuration-Space Pixel Tests — "
        "самое сильное выжившее утверждение. Тесты включают Spectral Pixel (насыщающий), "
        "Correlation Pixel (насыщающий).\n"
        "Кросс-доменные валидации (исходные + программа 2026): Швеция FTT, Китай игровой "
        "запрет, Монреальский протокол, правило Волкера, плюс солнечные рентгеновские "
        "вспышки, трансформаторы, станки с ЧПУ, погода, приливы."
    )

    def _system(self, lang: str, layman_header: str, context_text: str) -> str:
        """Short, directive, token-light system prompt (keeps all rules).

        Always injects the 2026 Framework Summary so the model knows the current
        state of DREAM even when Context retrieval returns old 2023-manuscript
        material. On conflicts, the 2026 Summary takes precedence.
        """
        summary = self.SUMMARY_RU if lang == "ru" else self.SUMMARY_EN
        if lang == "ru":
            rules_ru = (
                "Русский ответ. Сначала опирайся на '2026 Сводку' ниже — это текущее состояние "
                "DREAM. Раздел Context может содержать материал из рукописи 2023; при "
                "конфликте 2026 Сводка важнее. Без прелюдий, извинений, ссылок/страниц. "
                "Формулы — LaTeX, кратко поясни символы. Не переводить имена переменных. "
                "ФОРМАТ: две секции — `### Научно` и `### {lay}`. "
                "Technical: 4–6 пунктов, ≤180 слов суммарно, до 2 коротких формульных блоков. "
                "{lay}: 3 пункта, ≤90 слов, минимум символов. Всегда выдай ОБЕ секции. "
                f"Заверши сообщением {STOP_MARK}."
            ).format(lay=layman_header)
            base = rules_ru + "\n\n---\n### 2026 Сводка\n" + summary + "\n---"
        else:
            base = (
                "English reply. First rely on the '2026 Summary' below — this is the current "
                "state of DREAM. The Context section may contain material from the 2023 "
                "manuscript; on conflicts the 2026 Summary takes precedence. "
                "No preambles/apologies/citations/pages. "
                "Formulas in LaTeX; briefly define symbols. Do not translate variable names. "
                "FORMAT: two sections — `### Technical` and `### {lay}`. "
                "Technical: 4–6 bullets, ≤180 words total, up to 2 short equation blocks. "
                "{lay}: 3 bullets, ≤90 words, minimal symbols. Always output BOTH sections. "
                f"Finish with {STOP_MARK}."
            ).format(lay=layman_header)
            base = base + "\n\n---\n### 2026 Summary\n" + summary + "\n---"

        if context_text and context_text != "NO_CONTEXT":
            base += "\n\n---\n### Context\n" + context_text + "\n---"
        return base

    @staticmethod
    def _strip_ru_anchor(q: str) -> str:
        return re.sub(r'(\s*(?:в|во)\s*DREAM)(?=[\s\.\!\?…]*$)', '', (q or ''), flags=re.I).strip()

    @staticmethod
    def _compact(s: str, max_chars: int) -> str:
        s = s.replace("\r", "")
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r"\n{3,}", "\n\n", s).strip()
        return s[:max_chars]

    @staticmethod
    def _tokens(s: str) -> List[str]:
        return re.findall(r"[a-zA-Zа-яА-Я0-9]+", (s or "").lower())

    def _key_terms(self, question: str, lang: str) -> List[str]:
        EN_STOP = {"the","a","an","and","or","of","to","in","on","for","with","by","from","at","as",
                   "is","are","was","were","be","being","been","that","this","these","those","it","its",
                   "we","you","he","she","they","i","my","our","your","their","but","if","than","then",
                   "about","into","over","after","before","not","no","yes","do","does","did","can","could"}
        RU_STOP = {"и","или","а","но","к","в","во","на","над","под","о","об","от","до","по","при","для",
                   "это","тот","та","те","этот","эта","эти","что","как","так","же","не","да","нет","ни",
                   "есть","быть","был","была","были","будет","мы","вы","он","она","они","мой","наш","ваш","их"}
        stop = RU_STOP if lang == "ru" else EN_STOP
        toks = self._tokens(question)
        out, seen = [], set()
        for t in toks:
            if len(t) >= 3 and t not in stop and t not in seen:
                seen.add(t); out.append(t)
        return out[:8]

    def _generic_answer(self, question: str, layman_header: str) -> str:
        """RU-only generic fallback, token-light, still structured."""
        sys = (
            "Русский ответ. Без упоминаний DREAM/PDF. "
            "ФОРМАТ: `### Technical` (4–6 пунктов, ≤180 слов, до 2 коротких формул) и `### {lay}` (3 пункта, ≤90 слов). "
            f"Заверши {STOP_MARK}."
        ).format(lay=layman_header)
        usr = "Дай краткое общее объяснение по теме вопроса:\n" + question.strip() + "\n\nОтвет:"

        try:
            gen_tok = _budget("ru", "", self.max_tokens)
            resp = self.client.chat.completions.create(
                model=self.model,
                temperature=min(self.temperature, 0.25),
                max_tokens=min(gen_tok, 560),
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.0,
                stop=[STOP_MARK],
                messages=[{"role": "system", "content": sys},
                          {"role": "user", "content": usr}],
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception:
            return (
                "### Technical\n- Временная ошибка генерации.\n- Повторите позже.\n\n"
                f"### {layman_header}\n- Сейчас ответ недоступен.\n{STOP_MARK}"
            )

    # ------ public: make an answer ------
    def answer(
        self,
        question: str,
        context_chunks: List[Chunk],
        lang: Optional[str] = None,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:

        lang = (lang or detect_lang(question)).lower()
        lang = "ru" if lang == "ru" else "en"
        science_header = "Научно" if lang == "ru" else "Technical"
        layman_header = "Простыми словами" if lang == "ru" else "Simply Put"

        # Build compact context (top 3); shorter by default, longer when math present
        ctx_blocks: List[str] = []
        for c in (context_chunks or [])[:3]:
            limit = 1200 if _has_math(c.text) else 900
            ctx_blocks.append(self._compact(c.text or "", limit))
        context = "\n\n---\n\n".join(ctx_blocks) if ctx_blocks else "NO_CONTEXT"

        # Context gate
        if context != "NO_CONTEXT":
            ctx_text_lc = context.lower()
            q_terms = self._key_terms(question, lang)
            need_hits = 1 if len(q_terms) <= 3 else 2
            hits_list = [t for t in q_terms if t in ctx_text_lc]
        else:
            hits_list, need_hits = [], 1

        def _insufficient_reply() -> str:
            if lang == "ru":
                missing = ", ".join([t for t in (self._key_terms(question, lang)) if t not in (context.lower() if context != "NO_CONTEXT" else "")][:5]) or "нужные термины"
                return (
                    "### Technical\n- Недостаточно контекста для точного ответа.\n"
                    f"- В контексте нет ключевых терминов: {missing}.\n\n"
                    f"### {layman_header}\n- Уточните разделы/формулы или переформулируйте вопрос.\n{STOP_MARK}"
                )
            else:
                missing = ", ".join([t for t in (self._key_terms(question, lang)) if t not in (context.lower() if context != "NO_CONTEXT" else "")][:5]) or "specific terms"
                return (
                    "### Technical\n- Context is insufficient for a precise answer.\n"
                    f"- Missing key terms in context: {missing}.\n\n"
                    f"### {layman_header}\n- Point to sections/formulas or rephrase the question.\n{STOP_MARK}"
                )

        if context == "NO_CONTEXT" or len(hits_list) < need_hits:
            if lang == "ru" and RU_GENERIC_FALLBACK:
                q_plain = self._strip_ru_anchor(question)
                return self._generic_answer(q_plain, layman_header)
            return _insufficient_reply()

        # Prompts
        system = self._system(lang, layman_header, context)

        user = (
            (f"Вопрос:\n{question.strip()}\n\n"
             "Отвечай, опираясь ТОЛЬКО на раздел «Context» системного сообщения. Следуй формату.\nОтвет:")
            if lang == "ru"
            else
            (f"Question:\n{question.strip()}\n\n"
             'Answer using ONLY the "Context" section in the system message. Follow the format.\nAnswer:')
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        for role, content in (history or [])[-6:]:
            if role in ("user", "assistant"):
                messages.append({"role": role, "content": self._compact(content or "", 1600)})
        messages.append({"role": "user", "content": user})

        def _call(model_name: str, max_tok: int):
            return self.client.chat.completions.create(
                model=model_name,
                temperature=self.temperature,
                max_tokens=max_tok,
                top_p=0.9,
                frequency_penalty=0.3,
                presence_penalty=0.0,
                stop=[STOP_MARK],
                messages=messages,
            )

        try:
            max_tok = _budget(lang, context, self.max_tokens)
            resp = _call(self.model, max_tok)
        except Exception as e:
            err = str(e)
            err_lc = err.lower()
            # Retry fallback models on rate-limit (429), over-capacity (503),
            # service-unavailable, or any transient error containing those markers.
            should_fallback = (
                "rate limit" in err_lc or "429" in err
                or "over capacity" in err_lc or "503" in err
                or "service unavailable" in err_lc
            )
            if should_fallback:
                for fb in FALLBACK_MODELS:
                    try:
                        resp = _call(fb, max_tok)
                        break
                    except Exception:
                        continue
                else:
                    return "Sorry, all models are busy right now. Please retry in a few seconds."
            else:
                return f"Error: {html.escape(err)}"

        out = (resp.choices[0].message.content or "").strip()
        finish_reason = getattr(resp.choices[0], "finish_reason", None)

        # Ensure layman section exists
        if f"### {layman_header}".lower() not in out.lower():
            try:
                fix_tok = min(140, max(80, int(max_tok * 0.28)))
                fix = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    max_tokens=fix_tok,
                    frequency_penalty=0.35,
                    presence_penalty=0.0,
                    stop=[STOP_MARK],
                    messages=messages + [
                        {"role": "assistant", "content": out},
                        {"role": "user", "content": f"Add ONLY the missing second section:\n\n### {layman_header}\n- 3 bullets, ≤90 words total.\n{STOP_MARK}"},
                    ],
                )
                out += "\n" + (fix.choices[0].message.content or "").strip()
            except Exception:
                pass

        # Brief continuation if cut
        if finish_reason == "length":
            try:
                cont_tok = min(160, max(90, int(max_tok * 0.30)))
                cont = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.2,
                    max_tokens=cont_tok,
                    frequency_penalty=0.35,
                    presence_penalty=0.0,
                    stop=[STOP_MARK],
                    messages=messages + [
                        {"role": "assistant", "content": out},
                        {"role": "user", "content": f"Continue briefly and finish the '{layman_header}' section within 3 bullets."},
                    ],
                )
                out += "\n" + (cont.choices[0].message.content or "").strip()
            except Exception:
                pass

        return out

# ================== Facade & hot-reload ===================
_retriever: Optional[DreamRetriever] = None
_answerer:  Optional[GroqAnswerer]  = None

# Signature of KB inputs to detect staleness
_KB_SIG: Optional[Tuple] = None
_RELOAD_LOCK = threading.Lock()

def _file_mtime(path: str) -> float:
    try:
        return os.path.getmtime(path)
    except Exception:
        return 0.0

def _pick_default_pdf() -> str:
    envp = os.environ.get("DREAM_PDF_PATH")
    if envp and os.path.exists(envp):
        return envp
    for p in ("kb/dream_faqs.pdf", "kb/dream.pdf"):
        if os.path.exists(p):
            return p
    return "kb/dream.pdf"

def _kb_signature(custom_pdf: Optional[str] = None) -> Tuple:
    pdf = custom_pdf or _pick_default_pdf()
    return (
        pdf, _file_mtime(pdf),
        FAQ_EN_PATH, _file_mtime(FAQ_EN_PATH),
        FAQ_RU_PATH, _file_mtime(FAQ_RU_PATH),
    )

def reload_index(force: bool = False) -> bool:
    """
    Rebuild the unified retriever if inputs changed (or on demand).
    Returns True if a rebuild actually happened.
    """
    global _retriever, _KB_SIG
    with _RELOAD_LOCK:
        sig = _kb_signature()
        if force or _retriever is None or sig != _KB_SIG:
            pdf_path = sig[0]
            _retriever = DreamRetriever(pdf_path, FAQ_EN_PATH, FAQ_RU_PATH)
            _retriever.build()
            _KB_SIG = sig
            return True
        return False

def _refresh_if_stale() -> None:
    """Cheap call per request; rebuilds only if something changed."""
    reload_index(False)

def warm_index(pdf_path: Optional[str] = None) -> None:
    """Build the retriever index once (idempotent)."""
    global _retriever, _KB_SIG
    with _RELOAD_LOCK:
        if pdf_path:
            _retriever = DreamRetriever(pdf_path, FAQ_EN_PATH, FAQ_RU_PATH)
            _retriever.build()
            _KB_SIG = _kb_signature(pdf_path)
        else:
            reload_index(force=True)

# --- lightweight term extraction + anchored retrieval ---
_EN_STOP = {"the","a","an","and","or","of","to","in","on","for","with","by","from","at","as",
            "is","are","was","were","be","being","been","that","this","these","those","it","its",
            "we","you","he","she","they","i","my","our","your","their","but","if","than","then",
            "about","into","over","after","before","not","no","yes","do","does","did","can","could"}
_RU_STOP = {"и","или","а","но","к","в","во","на","над","под","о","об","от","до","по","при","для",
            "это","тот","та","те","этот","эта","эти","что","как","так","же","не","да","нет","ни",
            "есть","быть","был","была","были","будет","мы","вы","он","она","они","мой","наш","ваш","их"}

def _terms(text: str, lang: str) -> List[str]:
    toks = re.findall(r"[a-zA-Zа-яА-Я0-9]+", (text or "").lower())
    stop = _RU_STOP if lang == "ru" else _EN_STOP
    out, seen = [], set()
    for t in toks:
        if len(t) >= 3 and t not in stop and t not in seen:
            seen.add(t); out.append(t)
    return out

def _select_chunks(question: str, lang: str, top_k: int) -> List[Chunk]:
    global _retriever
    base = _retriever.search(question, k=top_k)
    if not base:
        anchored_q = question + (" в контексте DREAM" if lang == "ru" else " in the DREAM context")
        return _retriever.search(anchored_q, k=top_k)

    ctx_text = " ".join(c.text for c in base).lower()
    qts = _terms(question, lang)
    need = 1 if len(qts) <= 3 else 2
    hits = sum(1 for t in qts if t in ctx_text)

    if hits >= need:
        return base

    anchored_q = question + (" в фреймворке DREAM" if lang == "ru" else " in the DREAM framework")
    alt = _retriever.search(anchored_q, k=top_k)
    return alt or base

def _append_ru_anchor(text: str, lang: str) -> str:
    """Append 'в DREAM' to RU questions unless DREAM is already present."""
    if lang != "ru":
        return text
    t = (text or "").strip()
    if re.search(r"\bDREAM\b", t, re.IGNORECASE):
        return t
    m = re.search(r"([\.!\?…]+)$", t)
    if m:
        return t[:m.start()] + " в DREAM" + m.group(0)
    return t + " в DREAM"

def groq_answer(
    question: str,
    top_k: int = 6,
    lang: Optional[str] = None,
    history: Optional[List[Tuple[str, str]]] = None,
) -> str:
    """Convenience wrapper used by app.py (hot-reload + unified KB)."""
    global _answerer, _retriever
    _refresh_if_stale()
    if _retriever is None:
        warm_index()
    if _answerer is None:
        _answerer = GroqAnswerer()

    resolved_lang = (lang or detect_lang(question)).lower()
    resolved_lang = "ru" if resolved_lang == "ru" else "en"

    # RU anchoring helps retrieval
    q_aug = _append_ru_anchor(question, resolved_lang)

    # Retrieval over PDF + FAQs
    chunks = _select_chunks(q_aug, resolved_lang, top_k)

    # Ask the model
    return _answerer.answer(q_aug, chunks, lang=resolved_lang, history=history)