import logging
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

class SemanticService:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SemanticService, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._model is None:
            logger.info("Loading Semantic model (all-MiniLM-L6-v2)...")
            try:
                self._model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                logger.error(f"Failed to load Semantic model: {e}")
                raise e

    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        Computes semantic similarity between two texts.
        Returns a score between 0.0 and 1.0.
        """
        if not text1 or not text2:
            return 0.0

        self._load_model()
        
        embeddings1 = self._model.encode(text1, convert_to_tensor=True)
        embeddings2 = self._model.encode(text2, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(embeddings1, embeddings2)
        return float(cosine_scores[0][0])
