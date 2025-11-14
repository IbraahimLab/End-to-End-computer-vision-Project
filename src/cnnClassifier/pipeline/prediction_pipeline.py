import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.logger.logging import logger


class PredictionPipeline:

    LABEL_COLUMNS = [
        "Acne",
        "Blackheads",
        "Dark Spots",
        "Dry Skin",
        "Eye bags",
        "Normal Skin",
        "Oily Skin",
        "Pores",
        "Skin Redness",
        "Wrinkles",
    ]

    def __init__(self):
        """
        Loads model + params from config.yaml + params.yaml
        """

        config = ConfigurationManager()

        # training config to get model path
        training_config = config.get_training_config()
        params = config.params

        self.model_path = training_config.trained_model_path
        self.image_size = tuple(params.IMAGE_SIZE[:2])  # (224, 224)

        logger.info(f"Loading trained model from: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        logger.info("Model loaded successfully.")

    def _preprocess_image(self, image_path: str):
        """
        Loads and preprocesses an image exactly like training.
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.image_size)
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)
        return img

    def predict(self, image_path: str, threshold: float = 0.5):
        """
        Performs prediction on a single image.

        Args:
            image_path: path to input image
            threshold: cut-off for multi-label classification
        """
        img = self._preprocess_image(image_path)

        logger.info(f"Performing inference on: {image_path}")
        preds = self.model.predict(img)[0]  # shape (10,)

        # Convert sigmoid outputs → 0/1 labels
        binary_outputs = (preds >= threshold).astype(int)

        result = {
            label: {
                "probability": float(pred),
                "prediction": int(binary)
            }
            for label, pred, binary in zip(self.LABEL_COLUMNS, preds, binary_outputs)
        }

        return result
