import logging

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = "cpu"

# Cache the embedding model
EMBEDDER = None


def get_embedder():
    """
    Load the BGE embedding model only once and reuse it.
    """

    global EMBEDDER

    if EMBEDDER is None:
        logger.info("Loading embedding model: %s", MODEL_NAME)

        EMBEDDER = HuggingFaceEmbeddings(
            model_name=MODEL_NAME,
            model_kwargs={
                "device": DEVICE
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

    return EMBEDDER