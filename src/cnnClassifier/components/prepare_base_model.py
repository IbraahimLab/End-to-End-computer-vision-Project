from pathlib import Path
from tensorflow import keras
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier.entity.artifact_entity import PrepareBaseModelArtifact
from cnnClassifier.utils.common import create_directories
from cnnClassifier.logger.logging import logger


class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config
        self.model = None

    @staticmethod
    def _save_model(path: Path, model: keras.Model):
        create_directories([path.parent])
        model.save(path)
        logger.info(f"Model saved at: {path}")

    def get_base_model(self) -> keras.Model:
        """
        Load VGG16 with imagenet weights, without top classifier.
        """
        logger.info("Loading VGG16 base model with ImageNet weights")

        self.model = keras.applications.vgg16.VGG16(
            include_top=self.config.params_include_top,  # False
            weights=self.config.params_weights,          # "imagenet"
            input_shape=self.config.params_image_size,   # [224, 224, 3]
        )

        logger.info("Base VGG16 model loaded successfully")
        self._save_model(self.config.base_model_path, self.model)

        return self.model

    def prepare_full_model(self) -> keras.Model:
        """
        Freeze CNN layers and add custom ANN head for 10 classes.
        """
        if self.model is None:
            self.get_base_model()

        logger.info("Freezing all layers of the base CNN model")
        for layer in self.model.layers:
            layer.trainable = False

        logger.info("Adding custom ANN classifier head on top")
        x = keras.layers.Flatten(name="flatten")(self.model.output)
        x = keras.layers.Dense(256, activation="relu", name="fc1")(x)
        x = keras.layers.Dropout(0.5, name="dropout")(x)
        output = keras.layers.Dense(
            self.config.params_classes,
            activation="softmax",
            name="predictions",
        )(x)

        full_model = keras.models.Model(
            inputs=self.model.input,
            outputs=output,
            name="vgg16_transfer_learning",
        )

        logger.info("Compiling updated model with Adam + categorical_crossentropy")
        full_model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=self.config.params_learning_rate  # 0.0001
            ),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        self._save_model(self.config.updated_base_model_path, full_model)

        return full_model

    def initiate_prepare_base_model(self) -> PrepareBaseModelArtifact:
        """
        Orchestrates base model loading + head creation + saving.
        """
        logger.info("=== Stage 02: Prepare Base Model started ===")
        self.get_base_model()
        self.prepare_full_model()
        logger.info("=== Stage 02: Prepare Base Model completed ===")

        return PrepareBaseModelArtifact(
            base_model_path=self.config.base_model_path,
            updated_base_model_path=self.config.updated_base_model_path,
        )
