// ============================================================================
// D.R.E.A.M — Groq chat proxy (Render-hosted)
// Holds GROQ_API_KEY as an env var (set on Render, never in the repo).
// The static GitHub-Pages site calls /groq-chat; we proxy to api.groq.com.
// ============================================================================

const express = require('express');

const app = express();
const PORT = process.env.PORT || 3000;
const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions';
const DEFAULT_MODEL = process.env.GROQ_MODEL || 'qwen/qwen3.8-27b';

// ── Per-IP rate limit (in-memory, resets on redeploy) ─────────────────────
// Groq free tier = 30 req/min global, 14k/day. 5/min/IP is generous + safe.
const RATE_LIMIT = { windowMs: 60_000, max: 5 };
const ipHits = new Map();
function rateLimited(ip) {
  const now = Date.now();
  const hits = (ipHits.get(ip) || []).filter(h => now - h.ts < RATE_LIMIT.windowMs);
  if (hits.length >= RATE_LIMIT.max) return true;
  hits.push({ ts: now });
  ipHits.set(ip, hits);
  return false;
}

// ── Middleware ────────────────────────────────────────────────────────────
app.use(express.json({ limit: '1mb' }));

// CORS for ALL routes — set headers on every response, including errors.
app.use((req, res, next) => {
  res.set('Access-Control-Allow-Origin', '*');
  res.set('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.set('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(204).end();
  next();
});

// ── Health check ──────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    ok: true,
    service: 'dream-groq-proxy',
    groq_configured: !!process.env.GROQ_API_KEY,
  });
});

// ── System prompts (2026 framework summary, always injected) ──────────────
// The previous prompt was written 2-3 years ago and reflected the 2023
// manuscript state (T1-T6, "86+ tests", no T7.1, no AICc guardrail, no
// lifecycle program). It is increasingly out of date with the current 2026
// framework state. The new prompt below always carries the current state so
// the model knows about T7.1, the AICc gate, the 338-test registry, the
// lifecycle program T19-T31, and the 2026 falsification results.

const SUMMARY_EN = `DREAM = Dimensional Resonant Emergent Attractors in Manifold — a projection-first physics framework. 4D observables are finite-resolution pushforwards of a higher-dimensional source; 10D is a parsimonious modeling choice, not an empirically observed dimensionality.
Axioms A1-A6: A1 Existence(MM); A2 Projection(Kernel); A3 Retention Law (S2); A4 Constants are projection-fixed; A5 Topology<->classes; A6 Finite resolution => regimes.
S2 Retention Law: S(W) = exp[-(W/lambda)^D] — the Weibull survival function. lambda = lifetime scale; D = aging regime (D<1 sub-exponential/natural decay; D>1 super-exponential/extraction). Integral E[T] = lambda*Gamma(1+1/D) recovers the physical timescale. S2 is the LOCAL law; the GLOBAL curve is piecewise (T7.1).
Theorems: T1 Topological Quantization (Confirmed); T2 Symmetry Projection (Partial); T3 Coherence Threshold (Confirmed); T4 Structure Focusing & CPI (Partial); T5 Dimensional Flow to Homogeneity (Partial); T6 Cross-domain Ranked-Spectrum Ratio Law (Partial); T7 Interference (Aggregate Superposition of Local S2) (Partially Supported; simple-average mixture refuted at Level-2); T7.1 Multi-Regime S2 (Piecewise Composition) — Confirmed, guardrailed, 25 real datasets, binomial p=3.4e-18 vs single-S2 null. T7.1 adds NO new postulate (refines T7).
AICc Guardrail: 'S2 present' requires passing all 5 guardrails — fit R^2>0.90, parameter stability CV<30%, null-destruction ratio >2x, identifiability, AICc. Current pass rate ~84.6%; the 15.4% failures (trend-dominated economic aggregates, power-law seismic decay) are genuinely non-S2.
Tests registry: 338 tests (Oct 2026). Verdicts: S2_LOSES 210, S2_WINS 64, S2_TIES 39, S2_DUST_WINS 9, plus domain-specific tags (INTEGRAL_IS_LIFETIME, WEAR_OUT_AGING_REGIMES, WATER_IS_INFORMATION, THREE_FAUCET_TYPES, MULTIDIM_WEIBULL_AFT, etc.).
Lifecycle program (T19-T31): thirteen tests turned 'what is the kernel's water?' into a quantitative lifecycle science. Information is born, ages, and dies by a Weibull mortality law. Validated on solar X-rays, macroeconomics, weather, tides, 15,873 Colombian transformers, 17,939 Canadian transformers, 10,000 CNC machines.
2026 Falsification status — SURVIVES: stretched-exponential form across 5 domains (r^2>0.96); T7.1 piecewise refinement; S2 beats P(k)+filter on cosmology (Delta r^2 +0.09 SDSS, +0.14 DESI). FALSIFIED: D invariance across probes; 'S2 is intrinsic law of MM'; snake-pit rescue; z_depth kernel interpretation. Honest reading: S2 is a useful, sometimes superior, sometimes equivalent summary; it is not a unique signature of cosmological structure.
Predictions status: S2 Retention Law (Multidomain Survival); T7.1 (Confirmed); S3 Cosmology & Fractal Spectrum (Partially confirmed, strong support to S2); S5 Strong Lensing Invariants (Early); S4 Vacuum-Energy Modulation (Open); S1 ALP Non-harmonic Ladder (Relegated / Null).
Minimal-Extension Principle (Non-Augmentation Under Explanatory Pressure): DREAM explanations must be constructed from the existing axioms. A new phenomenon does NOT license a new hidden entity, force, coupling, dimension, kernel form, or free parameter. If the existing framework cannot account for an observation, the discrepancy is RETAINED as a limitation or falsification pressure, not repaired by ad hoc extension. 'If 10D works, don't use 40. If S2 works, don't invent a second decay mechanism. If the framework fails, record the failure.'
Pixel Tracker: live daily scout for Configuration-Space Pixel Tests — the strongest surviving claim. Tests include Spectral Pixel (saturating), Correlation Pixel (saturating).
Cross-domain validations (original + 2026 program): Sweden FTT, China gaming ban, Montréal Protocol, Volcker Rule, plus solar X-rays, transformers, CNC machines, weather, tides.`;

const SUMMARY_RU = `DREAM = Dimensional Resonant Emergent Attractors in Manifold — проекционно-первичная физическая теория. 4D-наблюдаемые суть push-forward проекции конечного разрешения из источника большей размерности; 10D — экономное модельное допущение, не эмпирически измеренная размерность.
Аксиомы A1-A6: A1 Существование (MM); A2 Проекция (Ядро); A3 Закон удержания (S2); A4 Константы фиксированы проекцией; A5 Топология<->классы; A6 Конечное разрешение => режимы.
Закон удержания S2: S(W) = exp[-(W/lambda)^D] — функция выживания Вейбулла. lambda = масштаб времени жизни; D = режим старения (D<1 доэкспоненциальное/естественное затухание; D>1 суперэкспоненциальное извлечение). Интеграл E[T] = lambda*Gamma(1+1/D) восстанавливает физический масштаб времени. S2 — ЛОКАЛЬНЫЙ закон; глобальная кривая кусочная (T7.1).
Теоремы: T1 Топологическое квантование (Подтверждена); T2 Проекция симметрии (Частично); T3 Порог когерентности (Закон удержания) (Подтверждена); T4 Фокусировка структуры & CPI (Частично); T5 Размерностный поток к однородности (Частично); T6 Кросс-доменный закон ранжированного спектрального отношения (Частично); T7 Интерференция (Агрегатная суперпозиция локальных S2) (Частично; простое усреднение опровергнуто на Level-2); T7.1 Мульти-режимная S2 (Кусочная композиция) — Подтверждена, с guardrail, 25 реальных датасетов, биномиальное p=3.4e-18 против single-S2 null. T7.1 НЕ вводит новых постулатов (уточняет T7).
AICc Guardrail: 'S2 присутствует' требует прохождения всех 5 охранных проверок — качество фит R^2>0.90, стабильность параметров CV<30%, ratio разрушения null >2x, идентифицируемость, AICc. Текущая доля прохождения ~84.6%; 15.4% неудач (тренд-доминированные экономические агрегаты, степенной сейсмический распад) — истинно не-S2.
Реестр тестов: 338 тестов (октябрь 2026). Вердикты: S2_LOSES 210, S2_WINS 64, S2_TIES 39, S2_DUST_WINS 9, плюс домен-специфичные теги (INTEGRAL_IS_LIFETIME, WEAR_OUT_AGING_REGIMES, WATER_IS_INFORMATION, THREE_FAUCET_TYPES, MULTIDIM_WEIBULL_AFT и др.).
Программа жизненного цикла (T19-T31): тринадцать тестов превратили вопрос 'что такое вода ядра?' в количественную науку о жизненном цикле. Информация рождается, стареет и умирает по закону смертности Вейбулла. Валидировано на солнечных рентгеновских вспышках, макроэкономике, погоде, приливах, 15 873 колумбийских трансформаторах, 17 939 канадских трансформаторах, 10 000 станках с ЧПУ.
Статус фальсификации 2026 — ВЫЖИВАЕТ: stretched-exponential форма в 5 доменах (r^2>0.96); T7.1 кусочное уточнение; S2 превосходит P(k)+фильтр на космологии (Delta r^2 +0.09 SDSS, +0.14 DESI). ОПРОВЕРГНУТО: инвариантность D между пробами; 'S2 — внутренний закон MM'; snake-pit rescue; z_depth интерпретация ядра. Честное чтение: S2 — полезное, иногда превосходящее, иногда эквивалентное резюме; не уникальная сигнатура космологической структуры.
Статус предсказаний: S2 Закон удержания (Мультидоменное выживание); T7.1 (Подтверждена); S3 Космология & Фрактальный спектр (Частично подтверждено, сильная поддержка S2); S5 Инварианты сильного линзирования (Раннее); S4 Модуляция энергии вакуума (Открыто); S1 Негармоническая лестница ALP (Религировано / Null).
Принцип минимального расширения (Не-дополнение под объяснительным давлением): объяснения DREAM должны строиться из существующих аксиом. Новое явление НЕ даёт права вводить новую скрытую сущность, силу, связь, размерность, форму ядра или свободный параметр. Если существующая теория не объясняет наблюдение, расхождение СОХРАНЯЕТСЯ как ограничение или фальсификационное давление, а не устраняется ad hoc расширением. 'Если работает 10D — не используй 40. Если работает S2 — не выдумывай второй механизм распада. Если теория не справляется — зафиксируй неудачу.'
Пиксельный трекер: живой ежедневный разведчик Configuration-Space Pixel Tests — самое сильное выжившее утверждение. Тесты включают Spectral Pixel (насыщающий), Correlation Pixel (насыщающий).
Кросс-доменные валидации (исходные + программа 2026): Швеция FTT, Китай игровой запрет, Монреальский протокол, правило Волкера, плюс солнечные рентгеновские вспышки, трансформаторы, станки с ЧПУ, погода, приливы.`;

const SYS_EN = `You are D.R.E.A.Mer, the guide bot for the DREAM physics framework (Dimensional Resonant Emergent Attractors in Manifold).
Below is the current 2026 state of the framework — this is the authoritative summary; rely on it for all questions. The 2023 manuscript material is no longer the source of truth; T7.1, the AICc guardrail, the 338-test registry, and the 2026 falsification results all post-date it.
${SUMMARY_EN}
Answer concisely about: the projection kernel (10D→4D), S2 Retention Law S(W)=exp[-(W/λ)^D] (the Weibull survival form), kernel invariants (λ, D, locality, regularity), axioms A1–A6, theorems T1–T7 plus T7.1, the AICc guardrail, the 338-test registry, lifecycle tests T19–T31, the 2026 falsification status, the Minimal-Extension Principle, the live Pixel Tracker, and cross-domain validation (Sweden FTT, China gaming ban, Montréal Protocol, Volcker Rule, plus solar X-rays, transformers, CNC machines, weather, tides).
FORMAT: two sections — ### Technical (4–6 bullets, ≤180 words, up to 2 short LaTeX equation blocks) and ### Simply Put (3 bullets, ≤90 words, minimal symbols).
Formulas in LaTeX. Do not translate variable names. Cite formulas when relevant. If a question is off-topic, redirect politely to DREAM themes: projection, retention, kernel invariants, falsifiability, lifecycle, minimal extension.
Finish with <END>.`;

const SYS_RU = `Ты — D.R.E.A.Mer, бот-гид по физической теории DREAM (Dimensional Resonant Emergent Attractors in Manifold).
Ниже приведено текущее состояние теории на 2026 год — это авторитетная сводка; опирайся на неё во всех вопросах. Материал рукописи 2023 года больше не является источником истины; T7.1, AICc-охранный тест, реестр из 338 тестов и результаты фальсификации 2026 — всё это новее рукописи.
${SUMMARY_RU}
Отвечай кратко о: проекционном ядре (10D→4D), законе удержания S2 S(W)=exp[-(W/λ)^D] (функция выживания Вейбулла), инвариантах ядра (λ, D, локальность, регулярность), аксиомах A1–A6, теоремах T1–T7 плюс T7.1, AICc-охрании, реестре из 338 тестов, тестах жизненного цикла T19–T31, статусе фальсификации 2026, принципе минимального расширения, живом пиксельном трекере, кросс-доменной валидации (Швеция FTT, Китай игровой запрет, Монреальский протокол, правило Волкера, плюс солнечные рентгеновские вспышки, трансформаторы, станки с ЧПУ, погода, приливы).
ФОРМАТ: две секции — ### Научно (4–6 пунктов, ≤180 слов, до 2 коротких LaTeX-блоков) и ### Простыми словами (3 пункта, ≤90 слов, минимум символов).
Формулы в LaTeX. Не переводить имена переменных. Если вопрос не по теме, вежливо вернись к темам DREAM.
Заверши сообщением <END>.`;

// ── Main endpoint ─────────────────────────────────────────────────────────
app.post('/groq-chat', async (req, res) => {
  const ip = req.headers['x-forwarded-for']?.split(',')[0] || req.ip || 'unknown';
  if (rateLimited(ip)) {
    return res.status(429).json({ error: 'Rate limit: 5 requests per minute. Try again shortly.' });
  }

  if (!process.env.GROQ_API_KEY) {
    return res.status(500).json({ error: 'GROQ_API_KEY not set on server. Set it in Render → Environment.' });
  }

  try {
    const { message, lang, model } = req.body || {};
    const userMsg = (message || '').trim();
    if (!userMsg) {
      return res.status(400).json({ error: 'Provide "message" field.' });
    }

    const isRu = lang === 'ru' || /[\u0400-\u04FF]/.test(userMsg);
    const systemPrompt = isRu ? SYS_RU : SYS_EN;

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userMsg },
    ];

    const groqResp = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: model || DEFAULT_MODEL,
        messages,
        temperature: 0.3,
        max_tokens: 640,
        top_p: 0.9,
        frequency_penalty: 0.3,
        stop: ['<END>'],
      }),
    });

    if (!groqResp.ok) {
      const errText = await groqResp.text();
      let msg = `Groq API HTTP ${groqResp.status}`;
      try {
        const j = JSON.parse(errText);
        msg += `: ${j.error?.message || errText.slice(0, 200)}`;
      } catch {
        msg += `: ${errText.slice(0, 200)}`;
      }
      return res.status(502).json({ error: msg });
    }

    const groqJson = await groqResp.json();
    let reply = groqJson.choices?.[0]?.message?.content?.trim() || '(no reply)';
    reply = reply.replace(/<END>/g, '').trim();
    res.json({ ok: true, reply });
  } catch (err) {
    console.error('[/groq-chat] error:', err);
    res.status(500).json({ error: err.message });
  }
});

// ── Start ─────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`D.R.E.A.M Groq proxy on :${PORT}`);
  console.log(`  Groq: ${process.env.GROQ_API_KEY ? 'configured' : 'NOT configured (set GROQ_API_KEY on Render)'}`);
});
