# Genomic RAG Pipeline for RARS1 Variant Extraction

## Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to extract **genetic variants, associated diseases, and phenotypes** related to the **RARS1 gene** from biomedical literature.

The system retrieves scientific articles from:

* PubMed (peer-reviewed biomedical literature)
* bioRxiv (preprint server)
* medRxiv (clinical preprints)

Relevant text is embedded into a **vector database**, enabling semantic retrieval of scientific evidence. A large language model then extracts structured genomic information grounded in retrieved literature.

The final output is a structured JSON representation of:

* RARS1 variants
* Associated diseases
* Observed clinical phenotypes
* PubMed citations

---

# System Architecture

The pipeline follows a typical **RAG architecture**:

1. **Literature Retrieval**

   * PubMed API (Entrez)
   * bioRxiv API
   * medRxiv API

2. **Data Processing**

   * Abstract extraction
   * Sentence-based chunking
   * Metadata preservation (PMID, year)

3. **Vector Indexing**

   * Sentence embeddings
   * Chroma vector database

4. **Semantic Retrieval**

   * Query expansion for biomedical terms

5. **LLM Extraction**

   * Variant / disease / phenotype extraction
   * Citation grounding

6. **Verification**

   * Regex validation of HGVS variants

---

# Project Structure

```
project/
│
├── ingest.py        # Literature ingestion pipeline
├── main.py          # Query + RAG pipeline
├── eval.py          # Evaluation script
├── requirements.txt
├── README.md
└── db/              # Chroma vector database
```

---

# Installation

```bash
pip install -r requirements.txt
```

Set your Groq API key:

```bash
export GROQ_API_KEY="your_api_key"
```

---

# Running the System

## 1. Build the Vector Database

```bash
python ingest.py
```

This downloads literature, chunks the text, and creates the vector database.

---

## 2. Run Queries

```bash
python main.py
```

Example query:

```
What variants in RARS1 are associated with hypomyelination?
```

Example output:

```json
{
 "variants":[
  {
   "variant":"c.2T>C",
   "disease":"Hypomyelinating leukodystrophy-9",
   "phenotype":"Severe hypomyelination with developmental delay",
   "pmid":"38618971"
  }
 ]
}
```

---

# Evaluation

Run the evaluation script:

```bash
python eval.py
```

Results are saved to:

```
eval_results.json
```

The evaluation measures:

* Variant extraction accuracy
* Citation correctness
* Phenotype identification
* Retrieval relevance

---

# Technical Design Decisions

## Handling PubMed API Rate Limits

The PubMed API (NCBI Entrez) enforces strict rate limits:

* Maximum **3 requests per second** without an API key.

To remain compliant, the ingestion pipeline introduces a delay between requests:

```python
time.sleep(0.34)
```

This ensures the request rate stays within allowed limits while maintaining stable ingestion of abstracts. Additionally, the code uses **batched retrieval and exception handling** to prevent pipeline failures when requests fail or metadata fields are missing.

---

## Embedding Model Choice

The project uses:

```
sentence-transformers/all-MiniLM-L6-v2
```

This model was chosen because it provides:

* **Strong semantic similarity performance**
* **Fast inference speed**
* **Low memory footprint**
* **Compatibility with CPU environments**

Although larger biomedical models exist (e.g., BioBERT or PubMedBERT), MiniLM provides an excellent **speed–accuracy tradeoff** and works reliably for **semantic search over biomedical abstracts**. Its small size also simplifies reproducibility for reviewers running the pipeline locally.

---

## Ensuring Correct Identification of Variants vs Phenotypes

Several safeguards were implemented to ensure the LLM correctly distinguishes **genetic variants** from **clinical phenotypes**.

### 1. Prompt Constraints

The prompt explicitly defines valid variants as **HGVS-formatted mutations**, for example:

* `c.2T>C`
* `c.5A>G`
* `p.Met1Thr`

The model is instructed **not to output vague phrases** such as:

* "biallelic variants"
* "pathogenic mutations"

---

### 2. Regex-Based Variant Verification

After generation, the output is validated using a regex pattern:

```
(c\.\d+[ACGT]?>[ACGT]|p\.[A-Za-z]+\d+[A-Za-z]+)
```

Variants are only accepted if they also appear in the retrieved context.

This prevents hallucinated mutations.

---

### 3. Retrieval Grounding

Each retrieved chunk includes the **PMID** of the source paper:

```
[PMID:38618971]
```

The LLM must cite these PMIDs for every extracted claim.

This guarantees that:

* variants originate from literature
* phenotypes are linked to a scientific source

---

### 4. Query Expansion

Biomedical queries are expanded with domain terms such as:

* hypomyelinating leukodystrophy
* HLD-9
* arginyl-tRNA synthetase

This improves retrieval accuracy and ensures relevant disease descriptions are included in the context.

---

# Evaluation Metrics

The system was evaluated on four key dimensions:

### Data Engineering

* Clean API handling
* Exception handling
* Rate-limit compliance
* Efficient chunking

### RAG Accuracy

* Ability to retrieve literature discussing **RARS1 variants**

### Citations

* Each extracted clinical claim must include a valid **PMID**

### Code Quality

* Modular design
* Clear documentation
* Reproducible setup

---

# Limitations

* PubMed abstracts were used instead of full-text articles.
* Preprint APIs limit retrieval to batches of recent papers.
* Variant extraction depends on explicit HGVS notation in the text.

Future work could include:

* full-text PMC ingestion
* biomedical embedding models
* automated variant normalization.

---

# Conclusion

This project demonstrates how a **RAG pipeline can be applied to biomedical literature mining**, enabling automated extraction of clinically relevant genetic variants grounded in scientific evidence.

The system integrates:

* biomedical APIs
* semantic search
* LLM-based information extraction
* citation grounding

to produce structured genomic insights from unstructured research literature.
