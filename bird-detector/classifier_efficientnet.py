"""Fine-tuned EfficientNet-B0 species classifier.

Loads a model checkpoint produced by scripts/train_efficientnet.py.
The checkpoint is a single .pt file containing both the model weights
and the label map, so no external species list is needed at inference time.

Install:
    pip install torch torchvision pillow
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EfficientNetClassifier:
    """EfficientNet-B0 fine-tuned on your own feeder crops.

    The model file path is set by EFFICIENTNET_MODEL_PATH in config / .env.
    Train the model first with scripts/train_efficientnet.py.
    """

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._model = None
        self._labels: list[str] = []
        self._transform = None
        self._device = "cpu"

    def load(self) -> None:
        """Load the fine-tuned checkpoint."""
        import torch
        import torchvision.transforms as T

        if not self._model_path.exists():
            raise FileNotFoundError(
                f"EfficientNet model not found: {self._model_path}\n"
                f"Train it first with:\n"
                f"  python scripts/train_efficientnet.py --data-dir training-data/"
            )

        logger.info("Loading EfficientNet checkpoint: %s", self._model_path)
        checkpoint = torch.load(self._model_path, map_location="cpu")
        self._labels = checkpoint["labels"]
        n_classes = len(self._labels)

        import torchvision.models as models
        backbone = models.efficientnet_b0(weights=None)
        in_features = backbone.classifier[1].in_features
        backbone.classifier[1] = torch.nn.Linear(in_features, n_classes)
        backbone.load_state_dict(checkpoint["model_state"])

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = backbone.to(self._device).eval()

        # Same normalisation used during training
        self._transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        logger.info(
            "EfficientNet ready: %d classes on %s", n_classes, self._device
        )

    def classify(
        self,
        frame_bgr: np.ndarray,
        top_n: int = 5,
    ) -> list[tuple[str, float]]:
        """Classify a BGR crop and return top-N (species, confidence) pairs."""
        if self._model is None:
            logger.error("Model not loaded — call load() first")
            return []
        if frame_bgr.size == 0:
            return []

        try:
            import cv2
            import torch
            from PIL import Image

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            tensor = self._transform(pil_img).unsqueeze(0).to(self._device)

            with torch.no_grad():
                logits = self._model(tensor)
                probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()

            ranked = np.argsort(probs)[::-1][:top_n]
            return [(self._labels[int(i)], float(probs[i])) for i in ranked]
        except Exception:
            logger.exception("EfficientNet classification failed")
            return []
