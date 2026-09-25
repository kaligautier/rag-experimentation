from app.config.settings import settings


def vector(x: float, y: float) -> list[float]:
    """Use the real storage dimension while keeping cosine scores transparent."""
    return [x, y] + [0.0] * (settings.rag.EMBEDDING_DIMENSIONS - 2)
