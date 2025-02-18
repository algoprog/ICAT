# ICAT: Information Coverage & Accuracy in Text
Evaluation framework for topic coverage and factuality in LLMs

ICAT is a comprehensive framework for evaluating topic coverage and factual accuracy in Large Language Model (LLM) outputs. The framework provides three evaluation methods with varying levels of automation:

- **ICAT-M**: Manual evaluation using ground-truth relevance judgments
- **ICAT-S**: Semi-automatic evaluation using LLM-based aspect-claim alignment
- **ICAT-A**: Fully automatic evaluation with LLM-generated aspects

## Features

- Atomic claim generation from LLM outputs
- Factual verification through retrieval-based grounding
- Topic/aspect coverage assessment
- Support for both corpus-based and web-based retrieval
- Correlation analysis with human judgments
- Configurable trade-off between factuality and coverage using β parameter

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/icat.git
cd icat
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Modify the `config.ini` file with your credentials (you can also set these later):
```ini
[Paths]
CACHE_PATH = ./cache
[Tokens]
HF_TOKEN = your_huggingface_token
OPENAI_API_KEY = your_openai_key
BRAVE_API_KEY = your_brave_search_key
[URLs]
OPENAI_BASE_URL = https://api.openai.com/v1
[Logging]
VLLM_LOGGING_LEVEL = WARNING
```

## Usage

### Basic Usage
```python
from icat import ICAT

scorer = ICAT(
corpus_path="path/to/corpus.jsonl", # Optional for corpus-based retrieval
queries_path="path/to/queries.jsonl",
qrels_path="path/to/qrels.jsonl", # Optional for ICAT-M
use_web_search=False # Set to True for web-based retrieval
)
```
### Evaluate responses using different methods
```python
results_m, metrics_m = scorer.icat_score_m(model_responses=responses)
results_s, metrics_s = scorer.icat_score_s(model_responses=responses)
results_a, metrics_a = scorer.icat_score_a(model_responses=responses)
```

## Input Formats

### Corpus Format (JSON Lines)
```json
{"id": "doc1", "contents": "Document text here..."}
{"id": "doc2", "contents": "Another document text..."}
...
```
### Queries Format (JSON Lines)
```json
{"query_id": 1, "query": "some query text...", "subtopics": ["subtopic 1", "subtopic 2", ...]}
{"query_id": 2, "query": "another query text...", "subtopics": ["subtopic 3", "subtopic 4", ...]}
...
```
### Qrels Format (JSON Lines)
```json
{"query_id": 1, "doc_id": "clueweb09-en0000-08-10767", "relevance": 0, "subtopic_id": 0}
{"query_id": 2, "doc_id": "clueweb09-en0000-08-10769", "relevance": 1, "subtopic_id": 2}
...
```
