# Handwritten PDF OCR → Excel

A local Python pipeline for converting scanned or handwritten PDF pages into machine-readable text and, when the required fields are known, structured Excel data.

The current implementation is designed as an **initial internship prototype**. The actual document formats, fields, languages, handwriting styles, and accuracy requirements have not yet been finalized, so the project intentionally avoids claiming production-level OCR accuracy.

## Current pipeline

### VLM workflow

```text
PDF
 │
 ▼
PDF page rendered to image
 │
 ▼
Light image cleanup
 │
 ▼
Vision-language model (VLM)
image → plain text transcription
 │
 ├─────────────── text mode ───────────────► .txt
 │
 ▼
Text-only LLM extraction
 │
 ▼
Evidence validation
 │
 ▼
Excel workbook
 ├── Data
 ├── Evidence
 └── Raw OCR
```

The important design choice is that the VLM is asked to **transcribe what is visible**, rather than directly inventing or filling database fields. Structured extraction happens separately and is restricted to the OCR transcript.

## Features

- PDF input, either a single file or a directory of PDFs.
- PDF pages rendered locally with PyMuPDF.
- Image preprocessing with OpenCV.
- VLM transcription through a local Ollama server.
- Optional text-only LLM extraction into user-specified columns.
- Evidence attached to extracted values.
- Evidence validation to reject values that cannot be supported by the OCR transcript.
- Excel output using `openpyxl`.
- Raw OCR/transcription preserved for inspection.
- Debug page images and transcripts can be saved.
- Optional TrOCR-based workflows are also present in the script.

## Current models

The current default VLM is:

```text
qwen3-vl:8b
```

The current default text extraction model is:

```text
qwen2.5:7b
```

These are local Ollama models. No cloud OCR API is required by the current implementation.

> Model choice may change once real internship documents are available.

## Requirements

- Windows/Linux/macOS
- Python 3.10+ recommended
- Ollama
- Sufficient RAM/VRAM for the selected local model
- The Python packages listed in `requirements.txt`

For the current Windows development environment, the project has also been tested with an NVIDIA GPU and CUDA-enabled PyTorch for the optional TrOCR path.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the core dependencies:

```powershell
pip install -r requirements.txt
```

Install and start Ollama separately.

Pull the current models:

```powershell
ollama pull qwen3-vl:8b
ollama pull qwen2.5:7b
```

If Ollama is not already running:

```powershell
ollama serve
```

Check installed models:

```powershell
ollama list
```

Check currently running models:

```powershell
ollama ps
```

## Main script

The current project script is:

```text
hw_pdf_to_new.py
```

If the project is cleaned up before delivery, a more neutral final name such as:

```text
handwritten_pdf_to_new.py
```

would be preferable.

## Basic usage

### 1. VLM transcription only

Use this when the goal is simply to determine what text is present in the document.

```powershell
python hw_pdf_to_xlsx_backup.py scans\scan.pdf --engine vlm --mode text --out-dir out_text --debug dbg_text
```

Output:

```text
out_text\scan.txt
```

Debug files include the processed page image and the VLM transcript for each page.

### 2. Process multiple PDFs as text

```powershell
python hw_pdf_to_new.py scans --engine vlm --columns "Field,Value" --out-dir out_fixed --debug dbg_fixed --vlm-model qwen2.5vl:7b
```

Every PDF in `scans` is processed.

### 3. Structured Excel extraction

When the required fields are known:

```powershell
python hw_pdf_to_xlsx_backup.py scans --engine vlm --columns "Name,Address,Year" --out-dir out_excel --debug dbg_excel
```

The workbook contains:

- **Data** — requested fields.
- **Evidence** — evidence associated with extracted values.
- **Raw OCR** — the original VLM transcription/extraction information retained for inspection.

If a requested value cannot be supported by the transcript, the validation layer removes the unsupported value instead of accepting an inferred value.

### 4. Example with a single PDF

```powershell
python hw_pdf_to_xlsx_backup.py scans\scan5.pdf --engine vlm --columns "Name,Address,Year" --out-dir out_excel --debug dbg_excel
```

## Important: do not invent fields yet

The final document format has not been provided yet.

Do **not** hard-code fields such as:

```text
Name
Address
Country
State
ZIP
Date
```

unless the actual internship documents require them.

The correct workflow is:

1. Receive representative PDFs.
2. Inspect their layouts and handwriting.
3. Identify the actual required fields.
4. Test transcription quality.
5. Define the Excel schema.
6. Improve preprocessing/model choice where the real documents expose weaknesses.
7. Measure accuracy against manually verified ground truth.

## Anti-hallucination design

The extraction prompt explicitly instructs the LLM to:

- use only information present in the OCR transcript;
- never guess missing information;
- never infer a country/state/city/etc.;
- preserve values as written;
- provide evidence for extracted values.

Example:

```text
OCR:
Essex
```

The system should not turn that into:

```text
Country = USA
```

because `USA` was not present in the OCR text.

If the OCR says:

```text
State: Essex
```

then:

```text
State = Essex
Evidence = State: Essex
```

is supportable.

This does not guarantee perfect OCR. The validation layer only checks whether an extracted value is supported by the transcript; it cannot determine whether the original handwriting was transcribed correctly.

## Debugging

For difficult documents, use the `--debug` option:

```powershell
python hw_pdf_to_xlsx_backup.py scans\scan.pdf --engine vlm --mode text --out-dir out --debug dbg
```

Inspect:

```text
dbg\
  <document>_p1.png
  <document>_p1.txt
  <document>_p2.png
  <document>_p2.txt
  ...
```

This helps separate three different problems:

1. **Image problem** — the page is poorly rendered or cleaned.
2. **Transcription problem** — the VLM cannot read the handwriting.
3. **Extraction problem** — the transcript is readable but structured extraction fails.

Do not try to solve all three problems with one prompt.

## Command-line options

The current script supports:

```text
--engine {trocr,vlm}
--mode {cells,lines,text}
--columns COLUMNS
--out-dir OUT_DIR
--debug DEBUG
--dpi DPI
--trocr-model TROCR_MODEL
--vlm-model VLM_MODEL
--vlm-max-side VLM_MAX_SIDE
--ollama-model OLLAMA_MODEL
--ollama-url OLLAMA_URL
--no-llm
--num-ctx NUM_CTX
```

Defaults currently include:

```text
engine       = trocr
mode         = cells
dpi          = 220
vlm-model    = qwen3-vl:8b
ollama-model = qwen2.5:7b
vlm-max-side = 1800
num-ctx      = 8192
```

For the internship's current VLM workflow, explicitly use:

```text
--engine vlm
```

and choose either:

```text
--mode text
```

for transcription-only testing, or:

```text
--columns "..."
```

for structured extraction.

## Optional TrOCR path

The script also contains a TrOCR implementation using:

```text
microsoft/trocr-base-handwritten
```

This path requires additional packages such as `transformers` and `torch`.

TrOCR is retained as an experimental/alternative engine. The VLM path is currently the main development path because the eventual document format and layout are not yet known.

Do not reinstall PyTorch blindly on a CUDA-enabled machine. Use the appropriate PyTorch installation for the machine's CUDA environment.

## Project structure

A clean project should eventually look approximately like:

```text
ocr/
├── handwritten_pdf_to_excel.py
├── README.md
├── requirements.txt
├── requirements-trocr.txt
├── .gitignore
├── scans/
│   └── # test PDFs only
├── out/
│   └── # generated files
└── debug/
    └── # generated debugging artifacts
```

Large model files should not be stored inside this repository. Ollama manages its own models separately.

## What is currently known

The prototype has been tested locally with handwritten/scanned test PDFs and the VLM workflow.

Observed behavior varies substantially between documents. Some pages transcribe successfully while difficult pages may return incomplete or empty output. This is expected at the prototype stage because the real internship document set has not yet been provided.

Therefore:

> **Current status: working prototype, not yet validated against the real production documents.**

## What remains intentionally undefined

The following should be decided after representative internship PDFs are received:

- required fields;
- document languages;
- handwriting styles;
- expected OCR accuracy;
- acceptable confidence/validation rules;
- page/layout variations;
- table handling;
- handwritten vs printed text;
- multi-document batch requirements;
- final Excel schema;
- whether human review is required for uncertain values;
- performance requirements.

## Development policy

Until representative documents are available, avoid unnecessary optimization.

The next development cycle should be driven by real documents:

```text
Real PDFs
   ↓
Failure cases
   ↓
Measure the failure
   ↓
Change one part of the pipeline
   ↓
Retest
   ↓
Keep/revert the change
```

This prevents optimizing the system for artificial test scans that may not resemble the internship data.

## Privacy

Scanned forms may contain personal or confidential information.

- Do not upload confidential documents to public repositories.
- Do not commit real scans to Git.
- Do not commit generated Excel files containing personal data.
- Do not commit debug images containing personal data.
- Keep local model/data directories outside version control.
- Share documents only through the approved university/company channels.

## Current project status

**Prototype stage**

Completed:

- PDF rendering
- image preprocessing
- local VLM transcription
- text-only mode
- optional structured extraction
- evidence generation
- evidence validation
- Excel generation
- debug output

Waiting for:

- representative internship PDFs
- final field/schema requirements
- accuracy expectations
- next mentor-defined task

The system should be improved further only after these requirements are known.
