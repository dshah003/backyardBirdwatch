"""BioCLIP species classifier using open_clip.

Model: imageomics/bioclip — a CLIP model trained on 10M+ biological images
from iNaturalist and GBIF.  Identifies species via zero-shot image-text
similarity: text embeddings for every species name in the allowlist are
precomputed at load time; at inference the image embedding is compared
against all of them.

Install:
    pip install open-clip-torch pillow

Reference: https://huggingface.co/imageomics/bioclip
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "hf-hub:imageomics/bioclip"


class BioCLIPClassifier:
    """Zero-shot bird species classifier backed by BioCLIP.

    Requires a species_list_path — the allowlist defines which species names
    are embedded as text prompts.  Without it there is nothing to compare
    image embeddings against.
    """

    def __init__(
        self,
        species_list_path: Path | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self._model_name = model_name
        self._species_list_path = species_list_path
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._species: list[str] = []
        self._text_embeddings = None  # torch.Tensor shape (n_species, embed_dim)
        self._device = "cpu"

    def load(self) -> None:
        """Download (if needed) and load model + precompute text embeddings."""
        import open_clip
        import torch

        logger.info("Loading BioCLIP model: %s …", self._model_name)
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self._model_name
        )
        self._tokenizer = open_clip.get_tokenizer(self._model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = self._model.to(self._device).eval()

        self._load_species()
        self._precompute_text_embeddings()
        logger.info("BioCLIP ready (%d species)", len(self._species))

    def _load_species(self) -> None:
        if self._species_list_path and self._species_list_path.exists():
            self._species = [
                line.strip()
                for line in self._species_list_path.read_text().splitlines()
                if line.strip()
            ]
        else:
            logger.warning(
                "No species_list_path set — BioCLIP has nothing to classify against"
            )
            self._species = []

    def _precompute_text_embeddings(self) -> None:
        import torch

        if not self._species:
            return

        prompts = [f"a photo of {s}" for s in self._species]
        tokens = self._tokenizer(prompts).to(self._device)
        with torch.no_grad():
            text_features = self._model.encode_text(tokens)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        self._text_embeddings = text_features
        logger.info(
            "Precomputed text embeddings for %d species", len(self._species)
        )

    def classify(
        self,
        frame_bgr: np.ndarray,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Classify a BGR crop and return top-N (species, confidence) pairs."""
        if self._model is None or self._text_embeddings is None:
            logger.error("Model not loaded — call load() first")
            return []
        if frame_bgr.size == 0 or not self._species:
            return []

        try:
            import cv2
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            img_tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                img_features = self._model.encode_image(img_tensor)
                img_features /= img_features.norm(dim=-1, keepdim=True)
                # Cosine similarities → softmax probabilities
                sims = (img_features @ self._text_embeddings.T).squeeze(0)
                probs = sims.softmax(dim=-1).cpu().numpy()

            ranked = np.argsort(probs)[::-1][:top_n]
            return [(self._species[int(i)], float(probs[i])) for i in ranked]
        except Exception:
            logger.exception("BioCLIP classification failed")
            return []
