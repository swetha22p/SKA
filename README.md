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
│   ├── json_formatter.py      ← Converts USR → JSON
│   ├── sanskrit_json_nlg.py   ← NLG from JSON input (Gemini API)
│   └── sanskrit_usr_nlg.py    ← NLG directly from USR input (Gemini API)
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
│  Option B (USR → NLG directly):                         │
│    USR  ──[sanskrit_usr_nlg.py]──► Generated Paragraph  │
└─────────────────────────────────────────────────────────┘
```



## API Key Setup

Both NLG scripts use the **Google Gemini API**. You must set your API key before
running either script.

### Step 1 — Get a Gemini API Key

Visit [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
and generate a key.

### Step 2 — Set the API Key in the Script

Open the relevant script (`sanskrit_json_nlg.py` or `sanskrit_usr_nlg.py`) and
replace the placeholder at the top of the file:

```python
# In sanskrit_json_nlg.py or sanskrit_usr_nlg.py
API_KEY = "YOUR_GEMINI_API_KEY_HERE"
```

> **Security note:** Do not commit your real API key to version control. Consider
> using an environment variable instead:
> ```python
> import os
> API_KEY = os.environ.get("GEMINI_API_KEY", "")
> ```
> Then set it in your shell: `export GEMINI_API_KEY="your-key-here"`

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
python Scripts/json_formatter.py
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

```bash
python Scripts/sanskrit_json_nlg.py InputDataSanskrit/JSON \
    -o SanskritGeneratedOutputs/GeminiFlash2-5 \
    -l english \
    -m zero_shot
```

| Argument | Description | Default |
|---|---|---|
| `json_folder` | Folder containing `.json` input files | *(required)* |
| `-o / --output_folder` | Folder to save output `.txt` files | `./output` |
| `-l / --language` | Target language: `english` or `hindi` | `english` |
| `-m / --mode` | Inference mode: `zero_shot` or `few_shot` | `zero_shot` |

> **Note:** The script currently filters for files whose names start with
> `test3_json`. Edit the `prefix_filter` variable inside `main()` to match
> your actual filenames.

---

### Option B — USR Input Directly (Single-Step Pipeline)

```bash
python Scripts/sanskrit_usr_nlg.py InputDataSanskrit/USR \
    -o SanskritGeneratedOutputs/GeminiFlash2-5 \
    -l hindi \
    -m gemini-2.5-flash
```

| Argument | Description | Default |
|---|---|---|
| `usr_folder` | Folder containing `.txt` or `.usr` input files | *(required)* |
| `-o / --output_folder` | Folder to save output `.txt` files | `./output_usr` |
| `-l / --language` | Target language: `english` or `hindi` | `hindi` |
| `-m / --model` | Gemini model to use (see below) | `gemini-2.5-flash` |

**Available Gemini models:**

| Model | Notes |
|---|---|
| `gemini-2.5-flash` | Fast, recommended default |
| `gemini-2.5-pro` | Higher quality, slower |
| `gemini-2.0-flash` | Stable previous generation |
| `gemini-1.5-flash` | Lightweight option |
| `gemini-1.5-pro` | High quality previous generation |

> **Note:** The script currently filters for files whose names start with
> `06_chapter`. Edit the `prefix_filter` variable inside `main()` to match
> your actual filenames.

---

## Output Format

Each output file is a plain `.txt` file containing a single generated paragraph,
word-wrapped at 80 characters. Output filenames follow this convention:

| Script | Output filename pattern |
|---|---|
| `sanskrit_json_nlg.py` | `<input_stem>_gemini_english_json.txt` |
| `sanskrit_usr_nlg.py` | `<input_stem>_gemini_hindi_usr.txt` |

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

## Rate Limiting

Both NLG scripts enforce a maximum of **15 requests per minute** to stay within
Gemini API free-tier limits. If the limit is reached, the script automatically
pauses and prints a countdown. For large files that exceed the 30,000-character
batch limit, the input is split into smaller batches with a 60-second cooldown
between them.

---

## Dependencies

Install Python dependencies with:

```bash
pip install google-generativeai wxconv
```

| Package | Used by | Purpose |
|---|---|---|
| `google-generativeai` | `sanskrit_json_nlg.py`, `sanskrit_usr_nlg.py` | Gemini API access |
| `wxconv` | `json_formatter.py` | WX → UTF-8 transliteration for Hindi concepts |

Python 3.8 or higher is recommended.

---


