# Trend Hunter — understanding changes in a Chinese game community

This project identifies emerging player concerns in 《星环守望》 using a local Qwen model. It interprets messages in conversation context, discovers categories from their meaning, and reports changes with traceable message evidence. The model handles language interpretation; Python handles dates, evidence counts and trend decisions. No model training or external inference API is required.

## Run

Requirements: Python 3.8+, Qwen3-4B-Instruct-2507 in Q4_K_M GGUF format, and the Windows CPU build of llama.cpp. Python uses only the standard library.

Place the GGUF beside `run.py` or inside `models/`. The filename `Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf` is supported. Place `llama-server.exe` and its accompanying DLLs beside `run.py`, or extract the complete runtime into `tools/`. Runtime downloads: https://github.com/ggml-org/llama.cpp/releases

```bat
python run.py --input voices.jsonl --as-of 2026-07-24 --out trends.json
```

The command starts the local server, waits for the model, performs analysis and writes the output. It stops the server it started when processing finishes. No second terminal is needed. Inference runs on the CPU with a 4,096-token context and one server slot. The model weights and runtime are separate from the source-code archive.

`versions.csv` is detected beside the input automatically. To specify a model elsewhere, append `--model-path "C:\Models\model.gguf"`. For the August report:

```bat
python run.py --input voices.jsonl --as-of 2026-08-30 --out august.json
```

## Data breakdown

The supplied dataset contains **48,125 messages**, dated **June 1–August 30, 2026**, across seven platforms. **20,213 messages (42.0%)** have a parent reference, making reply context central to interpretation.

| Platform | Messages |
|---|---:|
| QQ | 16,362 |
| Bilibili | 9,207 |
| Discord | 6,686 |
| Weibo | 5,945 |
| TapTap | 3,342 |
| Steam | 3,317 |
| Bahamut | 3,266 |

Each JSONL row contains `id`, `parent_id`, `platform`, `channel`, `user_name`, `title`, `content` and `created_at`. IDs preserve provenance; parent IDs connect replies to their original discussion. Platform and author names help distinguish repeated participation from broader discussion. Titles and content supply the language evidence.

Messages mix Simplified and Traditional Chinese, English, slang, nicknames, emojis, image placeholders and short reactions. A reply such as “确实” depends on its parent. A joke can contain a substantive complaint, while a quotation may repeat a claim without endorsing it. There are no numerical share, like or rating fields: amplification is interpreted from available text and conversation links. Image placeholders are treated as unavailable image content.

The event calendar records the June 1 data start, July 1 anniversary event, July 21 Selena release and limited banner, and July 31 maintenance patch. Events provide context without establishing causation. August 30 contains only three early-morning messages; the coverage heuristic uses August 29 as the last complete analysis day.

## Processing steps

1. **Apply the date cutoff.** Keep messages created on or before `--as-of`, interpreted in Hong Kong time. Filter before building context or interpreting language. Parent context and calendar events must also be available by that date.
2. **Select messages for review.** Remove simple filler and sample across platforms, recent days, the preceding baseline and older history. The default reviews 96 messages for a practical local run. Use `--max-messages 512` for broader review or `--max-messages 0` for all eligible messages. Counts and shares describe reviewed evidence; `first_seen` is the earliest supporting observation in that evidence.
3. **Interpret Chinese in context.** Supply the current message, up to two ancestors and explicit nickname mappings observed by then. Qwen returns literal English, intended meaning, feeling and substantive concerns. Each concern records its target, Chinese and English issue, stance, humour, support/opposition and evidence type. Exact supporting quotes are checked against the supplied text.
4. **Discover topics semantically.** Group concerns by the same underlying issue and target before applying support thresholds. Different wording can express the same grievance. Character enthusiasm, competitive imbalance and purchase pressure remain separate propositions. Larger reviews use bounded grouping and reconciliation passes.
5. **Measure change.** Compare the latest three days with the previous fourteen. Compare reviewed-message shares on platforms present in both windows. Track supporting messages, distinct platform/name pairs, active days, opposition, quotations and humorous support. Daily peak counts limit repeated contributions from the same platform/name on the same day.
6. **Write inspectable results.** Return up to eight trends, with supporting message IDs and relevant parent context. Cache validated model responses for repeat runs. Save translations and topic membership in the accompanying diagnostics file.

Humour and criticism are separate dimensions: a sarcastic complaint can support a concern. Disagreement is recorded as opposition. An unendorsed quotation contributes attention but is excluded from support. A short agreement inherits the parent's subject, not an automatically assumed sentiment.

## How status is decided

| Status | Decision |
|---|---|
| `new` | A concern first observed within fourteen days, with current support |
| `growing` | Reviewed share increases by at least 1.4×, with sufficient observations in both windows, at least three recent supporting messages, multiple authors and multiple days |
| `one-off` | A recent short episode, lasting at most four days around its peak, subsides |
| `fading` | A longer episode subsides or reviewed share drops below the baseline threshold |
| `not real` | Repetition, timing and concentrated participation trigger a separate model assessment that supports suspected artificial amplification |

Ordinary topic alerts require at least three supporting messages. Current growth is checked before decline from an earlier peak. Stable topics need not be reported. Repetition alone does not establish artificial activity: organic campaigns and uncertain cases are retained in diagnostics rather than automatically labelled fake. Account creation dates are not inferred from usernames.

Output uses the challenge fields: `title`, `title_en`, `category`, `status`, `first_seen`, `voice_ids` and `why`, under `as_of` and `trends`.

## Development choices and validation

An initial lexical-first approach could discard rare paraphrases before semantic interpretation. The revised process interprets before minimum-support filtering. Exact filename assumptions were replaced with GGUF discovery and explicit-path support. Support, opposition and quotation now contribute separately, while current growth takes priority over an old peak.

Run the regression suite:

```bat
python -m unittest -v test_project
```

The 29 deterministic tests cover cutoff handling, parent context, quote validation, caching, semantic-pipeline wiring, evidence counts, renewed growth, uncertain coordination and server lifecycle. Model responses in these tests are simulated. Full-dataset loading checks also pass. The local Windows run has successfully loaded Qwen and started generating responses.

## End-of-August reading for the studio

Manual inspection of late-August evidence points to three distinct concerns: Selena's competitive dominance, pressure to pay for progression, and distrust of draw probabilities. Message `01a04e26-2d80-7a94-a43d-a9a33f92f8cb` criticises Selena's PVP balance. Message `01a04e25-79d0-70dd-bfd5-6096b65b2dad` objects to core materials being gated behind a premium pass. Review competitive accessibility and progression requirements separately: both concern fairness, but suggest different product responses.

Message `01a04e30-4388-75bf-a103-c96bb9938cd9` alleges worse draw outcomes and an undisclosed probability change. This is evidence of distrust, not confirmation that probabilities changed. Clear explanations of draw mechanics and changes would address the communication concern. The event calendar confirms Selena's July release but does not confirm the alleged later adjustments.

Technical complaints also warrant triage. Message `01a04e21-cc68-7a52-aefb-2bfa27ebc4bb` reports repeated crashes; `01a04e3d-1c98-7d9e-b4f3-17dca639b2b1` reports saving freezes and latency. Collect reproduction details for these reports. These observations come from raw-message inspection; the generated date-specific reports provide the separate temporal classifications.

## Files and warm-up

| File | Purpose |
|---|---|
| `run.py` | Command-line entry point and server lifecycle |
| `data.py` | Loading, cutoff, context and sampling |
| `model.py` | Local inference, validation, caching and semantic grouping |
| `analysis.py` | Signals, status decisions and JSON output |
| `test_project.py` | Regression tests |
| `warmup.md` | All five Chinese warm-up explanations: literal meaning, intended meaning and feeling |

GPT-6 Astra assistance was used for implementation, review, tests and documentation. The inference pipeline uses the local Qwen model without fine-tuning.

