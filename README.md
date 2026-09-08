# SanskritUSR-NLG: Natural Language Generation from Sanskrit USR

This repository contains scripts to generate
natural language paragraphs (English and Hindi) from Sanskrit USR (Universal Semantic
Representation) inputs, with an intermediate JSON conversion step.

---

## Repository Structure

```
SKA/
├── InputDataSanskrit/
│   ├── JSON/                  ← Converted JSON files (intermediate format)
│   └── USR/                   ← Raw Sanskrit USR input files (.txt or .usr)
├── SanskritGeneratedOutputs/
│   └── GeminiFlash2-5/        ← Generated paragraph outputs
├── Scripts/
│   ├── json_formatter.py                      ← Converts USR → JSON
│   └── sanskrit_json_paragraph_nlg_inference.py  ← NLG from JSON input (Gemini API)
└── README.md
```

---

## Pipeline Overview

There are **two workflows** depending on your input format:

```
┌─────────────────────────────────────────────────────────┐
│  Input: USR file (.txt)                          │
│                                                         │
│  Option A (USR → JSON → NLG):                           │
│    USR  ──[json_formatter.py]──►  JSON                  │
│                                     │                   │
│                    [sanskrit_json_nlg.py]                │
│                                     │                   │
│                                     ▼                   │
│                           Generated Paragraph           │
│                                                         │
│  (Use Option A - convert USR → JSON first, then run the JSON NLG script) │
└─────────────────────────────────────────────────────────┘
```



## API Key Setup

Both NLG scripts use the **Google Gemini API**. You must set your API key before
running either script.

### Step 1 — Get a Gemini API Key

Visit [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
and generate a key.
### Step 2 — Where to put the API key (recommended)

Prefer keeping the key outside source files. The scripts in `Scripts/` read the
key from the `GEMINI_API_KEY` environment variable. Set it in your shell before
running the scripts:

```bash
# temporary for current session
export GEMINI_API_KEY="your-real-gemini-key-here"

# make it persistent (zsh)
echo 'export GEMINI_API_KEY="your-real-gemini-key-here"' >> ~/.zshrc
source ~/.zshrc
```

Optional environment variables the scripts accept:

- `GEMINI_MODEL`: model name to use (default: `gemini-2.5-flash`) — example:

```bash
export GEMINI_MODEL="gemini-2.5-pro"
```

- `BATCH_CHAR_LIMIT`: set the character limit used when batching large prompts
(default: `30000`):

```bash
export BATCH_CHAR_LIMIT=60000
```

Alternative ways to provide the key (choose one you prefer):

- Use a `.env` file and a loader (e.g., `python-dotenv`) in your scripts.
- Use a secrets manager or system-level environment variable provisioning.

Security note: never commit API keys to Git. Add files like `.env` to
`.gitignore` and use restricted keys where possible.

If you still prefer an inline assignment (not recommended), replace the
placeholder at the top of the script with your key, but be careful not to
commit it.

```python
# Not recommended for VCS: only for quick local testing
API_KEY = "your-real-gemini-key-here"
```
---

## Input Format

### USR Format (`.txt` or `.usr`)

Each sentence is enclosed in `<sent_id=...>` or `<segment_id=...>` tags.
Lines beginning with `#` contain the original Sanskrit sentence.
Lines beginning with `%` contain the sentence type (e.g., `%affirmative`).
All other non-tag lines are token rows with exactly 9 tab/space-separated columns.

```
<sent_id=SKA_001>
# रामः वनं गच्छति
% affirmative
rAma_1   1   male/per   sg   0:main   -   -   -   -
vana_1   2   place      sg   1:k2     -   -   -   -
gam-wA   3   -          sg   0:main   -   -   -   -
</sent_id>
```

### JSON Format (`.json`)

Produced by `json_formatter.py` from the USR files above. Each JSON file is an
array of sentence graph objects:

```json
[
  {
    "text": "रामः वनं गच्छति",
    "usr_id": "SKA_001",
    "SENT_TYPE": "affirmative",
    "nodes": [
      { "index": 1, "concept": "राम_1", "properties": { "attr_gen": "male" } },
      { "index": 2, "concept": "वन_1" },
      { "index": 3, "concept": "गम्_1" }
    ],
    "edges_dep": [
      ["SKA_001.1/राम_1", "k2", "SKA_001.2/वन_1"]
    ],
    "edges_cxn": [],
    "edges_discourse": []
  }
]
```

---

## Usage

### Option A — USR Input via JSON (Two-Step Pipeline)

#### Step 1: Convert USR to JSON

```bash
python Scripts/json_formatter.py --input-folder InputDataSanskrit/USR --output-folder InputDataSanskrit/JSON --log-folder InputDataSanskrit/usr_error_logs
```

Configure input/output paths inside the script (bottom of file):

```python
input_folder  = "InputDataSanskrit/USR"
output_folder = "InputDataSanskrit/JSON"
log_folder    = "InputDataSanskrit/usr_error_logs"
```

This will:
- Parse all `.txt` files in `input_folder`
- Write one `.json` file per input file to `output_folder`
- Log any malformed or skipped USR blocks to `log_folder`

#### Step 2: Generate Paragraphs from JSON

Run the JSON→paragraph script (zero-shot only):

```bash
# ensure API key is set first
export GEMINI_API_KEY="your-real-gemini-key-here"

python3 Scripts/sanskrit_json_paragraph_nlg_inference.py InputDataSanskrit/JSON \
  -o SanskritGeneratedOutputs/GeminiFlash2-5 \
  -l english
```

CLI summary:

| Argument | Description | Required / Default |
|---|---|---|
| `json_folder` | Folder containing `.json` input files | required |
| `-o / --output_folder` | Folder to save output `.txt` files | default: `./output` |
| `-l / --language` | Target language: `english` or `hindi` | default: `english` |

The script processes all `.json` files in the provided folder and writes
one `.txt` paragraph output per input file.

---

Supported Gemini models (set `GEMINI_MODEL` env to override):

| Model | Notes |
|---|---|
| `gemini-2.5-flash` | Fast, recommended default |
| `gemini-2.5-pro` | Higher quality, slower |
| `gemini-2.0-flash` | Stable previous generation |
| `gemini-1.5-flash` | Lightweight option |
| `gemini-1.5-pro` | High quality previous generation |

---

## Output Format

Each output file is a plain `.txt` file containing a single generated paragraph,
word-wrapped at 80 characters. Output filenames follow this convention:

| Script | Output filename pattern |
|---|---|
| `sanskrit_json_paragraph_nlg_inference.py` | `<input_stem>_gemini_english_json.txt` |

**Example output (`english`):**
```
Rama goes to the forest. The sages of the hermitage welcome him with offerings.
Sita and Lakshmana follow him faithfully on the path through the dense woodland.
```

**Example output (`hindi`):**
```
राम वन को जाते हैं। आश्रम के ऋषि उन्हें भेंट देकर स्वागत करते हैं।
सीता और लक्ष्मण घने वन के मार्ग पर उनका अनुसरण करते हैं।
```

---

## Error Logs

When running `json_formatter.py`, malformed or skipped USR blocks are logged per
input file under the configured `log_folder`. Each log file is named
`error-<input_stem>.txt` and contains lines of the form:

```
SKA_023 - Incorrect number of columns: 8 (expected 9)
SKA_047 - Missing sent_type
SKA_112 - Duplicate usr_id: SKA_112
```

If no errors are found for a file, no log file is created and the console
prints: `No errors for <filename>`.

---

## Rate Limiting and Batching

To avoid hitting Gemini service limits and to keep prompts within safe input
sizes the scripts use two protections:

- **Request rate limit:** at most 15 API calls per minute (configurable in
  code if you need to change it). When reached the script waits before
  continuing.
- **Batch character limit:** by default `BATCH_CHAR_LIMIT=30000` characters.

Why 30,000 characters? Briefly:

- Gemini models enforce an input size limit measured in tokens. Prompts and
  JSON payloads are counted toward this limit. 30,000 characters is a
  conservative, character-level proxy that keeps most prompts safely below
  typical token limits while leaving room for the script's instruction text
  and model metadata.
- It is intentionally conservative to avoid unexpected `413`/`429` errors
  caused by oversize requests. If you have measured average prompt size and
  want to increase the limit, set `BATCH_CHAR_LIMIT` (environment) higher.

Example override:

```bash
export BATCH_CHAR_LIMIT=60000
```

Note: a more precise approach is to compute token counts (e.g., using a
tokenizer matching the model) and batch by tokens instead of characters. Ask
me to add token-aware batching if you'd like that improvement.

---

## Dependencies

Install Python dependencies with:

```bash
pip install google-generativeai wxconv
```

| Package | Used by | Purpose |
|---|---|---|
| `google-generativeai` | `sanskrit_json_paragraph_nlg_inference.py` | Gemini API access |
| `wxconv` | `json_formatter.py` | WX → UTF-8 transliteration for Hindi concepts |

Python 3.8 or higher is recommended.

---

## Setup: Virtual environment and requirements

Create and activate a Python virtual environment, then install pinned dependencies from `requirements.txt`:

```bash
# create venv (uses python3 on macOS)
python3 -m venv .venv

# activate the venv (macOS / Linux)
source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

To deactivate the virtual environment run `deactivate`.

---


