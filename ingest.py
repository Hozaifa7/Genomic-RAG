import time
import requests
import logging

from Bio import Entrez
from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from transformers import logging as transformers_logging


# -----------------------------
# Configuration
# -----------------------------

Entrez.email = "YOUR_ENTREZ_EMAIL"

QUERY = "RARS1"

PUBMED_MAX_RESULTS = 50
BIORXIV_MAX_PAPERS = 500
MEDRXIV_MAX_PAPERS = 500

CHUNK_SIZE = 800
VECTOR_DB_DIR = "./db"

# Hide transformer warnings
transformers_logging.set_verbosity_error()


# -----------------------------
# Logging Setup
# -----------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# PubMed Retrieval
# =========================================================

def search_pubmed(query: str, max_results: int):
    """Search PubMed and return PMIDs."""
    
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=max_results,
        sort="pub date"
    )

    results = Entrez.read(handle)
    handle.close()

    return results["IdList"]


def fetch_pubmed_abstracts(pmids):
    """Retrieve PubMed abstracts."""

    documents = []

    for pmid in tqdm(pmids, desc="Fetching PubMed abstracts"):

        try:

            handle = Entrez.efetch(
                db="pubmed",
                id=pmid,
                rettype="abstract",
                retmode="xml"
            )

            records = Entrez.read(handle)
            handle.close()

            article = records["PubmedArticle"][0]["MedlineCitation"]["Article"]

            title = article.get("ArticleTitle", "")

            year = article["Journal"]["JournalIssue"]["PubDate"].get(
                "Year", "Unknown"
            )

            abstract_text = ""

            if "Abstract" in article:
                sections = article["Abstract"]["AbstractText"]
                abstract_text = " ".join(str(s) for s in sections)

            text = f"{title}. {abstract_text}"

            documents.append({
                "text": text,
                "pmid": pmid,
                "year": year
            })

            # Respect NCBI rate limits
            time.sleep(0.34)

        except Exception as e:

            logger.warning(f"Failed to fetch PMID {pmid}: {e}")

    return documents


def ingest_pubmed(query):
    """Full PubMed ingestion."""

    logger.info("Searching PubMed...")

    pmids = search_pubmed(query, PUBMED_MAX_RESULTS)

    logger.info(f"Found {len(pmids)} papers")

    docs = fetch_pubmed_abstracts(pmids)

    logger.info(f"Ingested {len(docs)} PubMed documents")

    return docs


# =========================================================
# Preprint Retrieval (bioRxiv / medRxiv)
# =========================================================

def fetch_preprints(query, server="biorxiv", max_papers=500):
    """
    Retrieve preprints from bioRxiv or medRxiv using cursor pagination.
    """

    base_url = "https://api.biorxiv.org/details"

    collected_docs = []
    cursor = 0
    batch_size = 100

    logger.info(f"Querying {server} API (max {max_papers})")

    while cursor < max_papers:

        url = f"{base_url}/{server}/2020-01-01/2026-01-01/{cursor}"

        logger.info(f"{server} cursor={cursor}")

        try:

            response = requests.get(url, timeout=20)

            if response.status_code != 200:
                logger.warning(f"{server} API request failed")
                break

            data = response.json()

        except Exception as e:

            logger.warning(f"{server} API error: {e}")
            break

        papers = data.get("collection", [])

        if len(papers) == 0:
            logger.info(f"No more {server} papers available")
            break

        for paper in papers:

            title = paper.get("title", "")
            abstract = paper.get("abstract", "")
            doi = paper.get("doi", "")
            date = paper.get("date", "")

            text = f"{title}. {abstract}"

            if query.lower() in text.lower():

                collected_docs.append({
                    "text": text,
                    "pmid": f"preprint:{doi}",
                    "year": date
                })

        cursor += batch_size

    logger.info(f"{server} matches: {len(collected_docs)}")

    return collected_docs


# =========================================================
# Source Aggregation
# =========================================================

def ingest_all_sources(query):
    """Collect literature from all sources."""

    pubmed_docs = ingest_pubmed(query)

    biorxiv_docs = fetch_preprints(query, "biorxiv", BIORXIV_MAX_PAPERS)

    medrxiv_docs = fetch_preprints(query, "medrxiv", MEDRXIV_MAX_PAPERS)

    all_docs = pubmed_docs + biorxiv_docs + medrxiv_docs

    logger.info(f"Total documents collected: {len(all_docs)}")

    return all_docs


# =========================================================
# Chunking
# =========================================================

def chunk_documents(documents, max_chars=CHUNK_SIZE):
    """
    Sentence-based chunking that preserves genomic variant tokens.
    """

    chunks = []

    for doc in documents:

        text = doc["text"]
        pmid = doc["pmid"]
        year = doc["year"]

        sentences = text.split(". ")

        current_chunk = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if len(current_chunk) + len(sentence) < max_chars:

                current_chunk += sentence + ". "

            else:

                chunks.append({
                    "id": f"{pmid}_{len(chunks)}",
                    "text": current_chunk.strip(),
                    "pmid": pmid,
                    "year": year
                })

                current_chunk = sentence + ". "

        if current_chunk:

            chunks.append({
                "id": f"{pmid}_{len(chunks)}",
                "text": current_chunk.strip(),
                "pmid": pmid,
                "year": year
            })

    logger.info(f"Created {len(chunks)} chunks")

    return chunks


# =========================================================
# LangChain Conversion
# =========================================================

def convert_to_langchain_docs(chunks):

    docs = []

    for chunk in chunks:

        docs.append(
            Document(
                page_content=chunk["text"],
                metadata={
                    "pmid": chunk["pmid"],
                    "year": chunk["year"]
                }
            )
        )

    return docs


# =========================================================
# Vector Store
# =========================================================

def create_vector_store(documents):

    logger.info("Loading embedding model")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    logger.info("Creating vector database")

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=VECTOR_DB_DIR
    )

    logger.info("Vector database created")

    return vectordb


# =========================================================
# Pipeline Runner
# =========================================================

def run_ingestion_pipeline():

    logger.info("Starting ingestion pipeline")

    docs = ingest_all_sources(QUERY)

    chunks = chunk_documents(docs)

    lc_docs = convert_to_langchain_docs(chunks)

    create_vector_store(lc_docs)

    logger.info("Pipeline complete. Ready for querying.")


if __name__ == "__main__":
    run_ingestion_pipeline()