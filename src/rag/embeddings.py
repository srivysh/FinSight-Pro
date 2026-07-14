import logging

from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-base-en-v1.5"
DEVICE = "cpu"


def get_embedder():
    """
    Load the BGE embedding model.
    """

    logger.info("Loading embedding model: %s", MODEL_NAME)

    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": DEVICE
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )