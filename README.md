#Sanskrit USR-NLG: Natural Language Generation from USR

This repository contains scripts for generating natural language paragraphs from Sanskrit **USR (Universal Semantic Representation)** inputs using two approaches:

1. **USR → Paragraph NLG**
2. **USR → JSON → Paragraph NLG**

The project allows comparison of paragraph generation using the original USR representation versus the converted JSON representation.

---

# Repository Structure

```text
SKA/
│
├── InputDataSanskrit/
│   │
│   ├── USR  #Raw Sanskrit USR input files
│   │
│   ├── JSON #JSON files converted from USR
│   │
│   └── usr_error_logs #USR formatting and conversion error logs
│
├── SanskritGeneratedOutputs/
│   │
│   ├── Gemini-2.5-Flash_USR  #Paragraphs generated directly from USR input
│   │
│   └── Gemini-2.5-Flash_JSON #Paragraphs generated from JSON input
│
├── Scripts/
│   │
│   ├── json_formatter.py #Converts Sanskrit USR → JSON
│   │
│   ├── sanskrit_usr_paragraph_nlg_inference.py  #Generates paragraphs directly from USR
│   │
│   └── sanskrit_json_paragraph_nlg_inference.py #Generates paragraphs from JSON
│
├── requirements.txt
│
└── README.md
```

---

# Pipeline Overview

There are two workflows in this project.

## Workflow 1: Direct USR → Paragraph Generation

```text
Sanskrit USR Input
        │
        ▼
sanskrit_usr_paragraph_nlg_inference.py
        │
        ▼
Gemini 2.5 Flash
        │
        ▼
Generated English Paragraph
```

Output location:

```text
SanskritGeneratedOutputs/Gemini-2.5-Flash_USR/
```

---

## Workflow 2: USR → JSON → Paragraph Generation

```text
Sanskrit USR Input
        │
        ▼
json_formatter.py
        │
        ▼
JSON Representation
        │
        ▼
sanskrit_json_paragraph_nlg_inference.py
        │
        ▼
Gemini 2.5 Flash
        │
        ▼
Generated English Paragraph
```

Output location:

```text
SanskritGeneratedOutputs/Gemini-2.5-Flash_JSON/
```

---

# Complete Experimental Pipeline

```text
                         Sanskrit USR
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        Direct USR Input              JSON Formatter
                 │                         │
                 ▼                         ▼
          Gemini 2.5 Flash            JSON Input
                 │                         │
                 │                         ▼
                 │                  Gemini 2.5 Flash
                 │                         │
                 ▼                         ▼
     Gemini-2.5-Flash_USR       Gemini-2.5-Flash_JSON
```

This structure makes it easy to compare:

```text
USR representation  vs  JSON representation
```

using the same Gemini model.

---

# API Key Setup

The NLG scripts use the Google Gemini API.

You must set your Gemini API key before running the scripts.

## Step 1: Get a Gemini API Key

Get an API key from Google AI Studio.

## Step 2: Set the API Key

### Temporary for the current terminal session

```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

### Persistent setup for macOS using zsh

```bash
echo 'export GEMINI_API_KEY="your-gemini-api-key"' >> ~/.zshrc
```

Then reload the configuration:

```bash
source ~/.zshrc
```

Check whether the API key is available:

```bash
echo $GEMINI_API_KEY
```

---

# Optional Environment Variables

## Gemini Model

You can optionally specify the Gemini model:

```bash
export GEMINI_MODEL="gemini-2.5-flash"
```

## Batch Character Limit

You can specify the batch character limit:

```bash
export BATCH_CHAR_LIMIT=25000
```

Recommended default:

```text
25000 characters
```

This leaves additional room for prompt instructions and formatting.

---

# Input Format

## Sanskrit USR Format

The input files contain Sanskrit USR structures.

Example:

```text
<sent_id=SKA_001>

# रामः वनं गच्छति

% affirmative

rAma_1   1   male/per   sg   0:main   -   -   -   -

vana_1   2   place      sg   1:k2     -   -   -   -

gam-wA   3   -          sg   0:main   -   -   -   -

</sent_id>
```

---

# JSON Format

The JSON files are generated from the USR files using:

```text
json_formatter.py
```

Example JSON structure:

```json
[
  {
    "text": "रामः वनं गच्छति",
    "usr_id": "SKA_001",
    "SENT_TYPE": "affirmative",

    "nodes": [
      {
        "index": 1,
        "concept": "राम_1",
        "properties": {
          "attr_gen": "male"
        }
      },
      {
        "index": 2,
        "concept": "वन_1"
      },
      {
        "index": 3,
        "concept": "गम्_1"
      }
    ],

    "edges_dep": [
      [
        "SKA_001.1/राम_1",
        "k2",
        "SKA_001.2/वन_1"
      ]
    ],

    "edges_cxn": [],
    "edges_discourse": []
  }
]
```

---

# Usage

# Workflow 1: USR → Paragraph

This workflow generates paragraphs directly from Sanskrit USR files.

## Command

```bash
python3 Scripts/sanskrit_usr_paragraph_nlg_inference.py \
  InputDataSanskrit/USR \
  -o SanskritGeneratedOutputs/Gemini-2.5-Flash_USR \
  -m gemini-2.5-flash
```

### Input

```text
InputDataSanskrit/USR/
```

### Output

```text
SanskritGeneratedOutputs/Gemini-2.5-Flash_USR/
```

---

# Workflow 2: USR → JSON → Paragraph

This workflow contains two steps.

---

## Step 1: Convert USR to JSON

Run:

```bash
python3 Scripts/json_formatter.py \
  -i InputDataSanskrit/USR \
  -o InputDataSanskrit/JSON \
  -l InputDataSanskrit/usr_error_logs
```

### Input

```text
InputDataSanskrit/USR/
```

### JSON Output

```text
InputDataSanskrit/JSON/
```

### Error Logs

```text
InputDataSanskrit/usr_error_logs/
```

---

## Step 2: Generate Paragraphs from JSON

Run:

```bash
python3 Scripts/sanskrit_json_paragraph_nlg_inference.py \
  InputDataSanskrit/JSON \
  -o SanskritGeneratedOutputs/Gemini-2.5-Flash_JSON
```

### Input

```text
InputDataSanskrit/JSON/
```

### Output

```text
SanskritGeneratedOutputs/Gemini-2.5-Flash_JSON/
```

---



| Output Folder           | Input Representation | Model            |
| ----------------------- | -------------------- | ---------------- |
| `Gemini-2.5-Flash_USR`  | Direct Sanskrit USR  | Gemini 2.5 Flash |
| `Gemini-2.5-Flash_JSON` | Converted JSON       | Gemini 2.5 Flash |

---

# Output Format

Each generated output is saved as a `.txt` file.

Example:

```text
Rama goes to the forest. The sages of the hermitage welcome him with
offerings. Sita and Lakshmana follow him faithfully on the path through
the dense woodland.
```

---

# Error Logs

During USR → JSON conversion, malformed or invalid USR structures are written to:

```text
InputDataSanskrit/usr_error_logs/
```

Example errors:

```text
SKA_023 - Incorrect number of columns

SKA_047 - Missing sentence type

SKA_112 - Duplicate usr_id
```

If no errors are found, no error log file is created for that input file.

---

# Rate Limiting and Batching

The Gemini scripts use batching and rate limiting to avoid excessive API requests.

## Request Rate Limit

The scripts limit the number of API requests per minute.

Example configuration:

```text
Maximum requests per minute: 15
```

When the limit is reached, the script waits before continuing.

---

## Batch Character Limit

Large inputs are divided into batches.

Recommended default:

```text
BATCH_CHAR_LIMIT = 25000
```

You can override it using:

```bash
export BATCH_CHAR_LIMIT=25000
```

The batch size should account for:

* JSON or USR input content
* Prompt instructions
* Structure identifiers
* Formatting characters

For this reason, a limit of approximately **25,000 characters** is safer than simply counting raw input content up to the maximum.

---

# Dependencies

Install the required Python packages:

```bash
pip install google-generativeai wxconv
```

The main dependencies are:

| Package               | Purpose                       |
| --------------------- | ----------------------------- |
| `google-generativeai` | Gemini API access             |
| `wxconv`              | WX transliteration/conversion |

> Note: The `google-generativeai` package may display a deprecation warning. The scripts can later be migrated to the newer `google.genai` package.

---

# Virtual Environment Setup

## Create a virtual environment

```bash
python3 -m venv venv
```

## Activate it on macOS/Linux

```bash
source venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Run the scripts

After activation:

```bash
python3 Scripts/json_formatter.py \
  -i InputDataSanskrit/USR \
  -o InputDataSanskrit/JSON \
  -l InputDataSanskrit/usr_error_logs
```

Then:

```bash
python3 Scripts/sanskrit_json_paragraph_nlg_inference.py \
  InputDataSanskrit/JSON \
  -o SanskritGeneratedOutputs/Gemini-2.5-Flash_JSON
```

Or run the direct USR pipeline:

```bash
python3 Scripts/sanskrit_usr_paragraph_nlg_inference.py \
  InputDataSanskrit/USR \
  -o SanskritGeneratedOutputs/Gemini-2.5-Flash_USR \
  -m gemini-2.5-flash
```

---

# Summary

The project supports two experimental approaches:

```text
Approach 1:

USR
 ↓
Gemini 2.5 Flash
 ↓
Generated Paragraph


Approach 2:

USR
 ↓
JSON Formatter
 ↓
JSON
 ↓
Gemini 2.5 Flash
 ↓
Generated Paragraph
```

The outputs can then be compared to evaluate whether **direct USR input** or **JSON-based semantic representation** produces better natural language generation results.