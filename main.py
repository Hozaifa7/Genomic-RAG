from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv
import os
import re
import logging

from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()


# -----------------------------
# Logging
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# -----------------------------
# Vector Store
# -----------------------------

def load_vector_store():

    logger.info("Loading vector database")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma(
        persist_directory="./db",
        embedding_function=embedding_model
    )

    return vectordb


# -----------------------------
# Retrieval
# -----------------------------

def expand_query(query):

    biomedical_terms = [
        "RARS1",
        "hypomyelinating leukodystrophy",
        "HLD-9",
        "arginyl-tRNA synthetase",
        "gene variant",
        "HGVS",
        "mutation",
        "genetic variant"
    ]

    expanded = query + " " + " ".join(biomedical_terms)

    return expanded

def retrieve_context(vectordb, query, k=5):

    logger.info(f"Retrieving top {k} documents")

    expanded_query = expand_query(query)

    logger.info(f"Expanded query: {expanded_query}")

    results = vectordb.similarity_search(expanded_query, k=k)

    context = ""

    for r in results:

        pmid = r.metadata.get("pmid", "unknown")

        text = r.page_content

        context += f"[PMID:{pmid}] {text}\n\n"

    logger.info(f"Retrieved {len(results)} chunks")

    return context


# -----------------------------
# LLM Setup
# -----------------------------

# Load variables from .env
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def ask_llm(context, query):

    logger.info("Sending prompt to LLM")

    prompt = f"""
You are a genomic research assistant.

Extract:
1. RARS1 variants
2. Diseases
3. Phenotypes

Rules:
- Only use the provided context
- Every claim must include a PMID
- If not present say "Not reported"

Definition of a variant:
A variant must follow HGVS notation such as:
- c.5A>G
- c.2T>C
- p.Met1Thr

Do NOT output phrases like:
- "biallelic variants"
- "pathogenic variants"
- "RARS1 mutations"

Return JSON in this exact format:

{{
 "variants":[
  {{
   "variant":"HGVS variant name",
   "disease":"associated disease",
   "phenotype":"clinical phenotype",
   "pmid":"PubMed ID"
  }}
 ]
}}

Use the PMID shown in the context.

Context:
{context}

Question:
{query}
"""

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )

    return completion.choices[0].message.content


# -----------------------------
# Variant Verification
# -----------------------------

def verify_variants(llm_output, context):

    variant_pattern = r"(c\.\d+[ACGT]?>[ACGT]|p\.[A-Za-z]+\d+[A-Za-z]+)"

    found_variants = re.findall(variant_pattern, llm_output)

    verified = []

    for variant in found_variants:

        if variant.lower() in context.lower():
            verified.append(variant)

    return verified


# -----------------------------
# Pipeline
# -----------------------------

def run_query(query):

    vectordb = load_vector_store()

    context = retrieve_context(vectordb, query)

    print("\nRetrieved Context:\n")
    print(context[:1500])

    answer = ask_llm(context, query)

    verified_variants = verify_variants(answer, context)

    print("\nVerified variants:")
    print(verified_variants)

    return answer


# -----------------------------
# CLI Entry
# -----------------------------

if __name__ == "__main__":

    query = input("Enter your question: ")

    result = run_query(query)

    print("\nAnswer:\n")
    print(result)