from pathlib import Path
import os

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from cnnClassifier.entity.config_entity import TrainingConfig

from cnnClassifier.entity.artifact_entity import (
    DataIngestionArtifact,
    PrepareBaseModelArtifact,
    ModelTrainingArtifact,
)
from cnnClassifier.logger.logging import logger


class ModelTraining:
    """
    Stage 03: Training
    """

    # label column names from your CSV header (exactly as you sent)
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

    DATASET_SUBDIR = "skin_problems_dataset_multilabel" # folder created after unzip
    TRAIN_CSV_NAME = "_classes.csv"
    VALID_CSV_NAME = "_classes.csv"
    TEST_CSV_NAME =  "_classes.csv"


     

    def __init__(
        self,
        config: TrainingConfig,
        params,
        data_ingestion_artifact: DataIngestionArtifact,
        prepare_base_model_artifact: PrepareBaseModelArtifact,
    ):
        self.config = config
        self.params = params
        self.data_ingestion_artifact = data_ingestion_artifact
        self.prepare_base_model_artifact = prepare_base_model_artifact

        # params
        self.image_size = tuple(self.params.IMAGE_SIZE[:2])  # (224, 224)
        self.batch_size = self.params.BATCH_SIZE
        self.epochs = self.params.EPOCHS

        # build dataset root
        self.dataset_root = Path(self.data_ingestion_artifact.unzip_dir) / self.DATASET_SUBDIR

    # ---------- helpers for data pipeline ----------

    def _load_split_df(self, split: str) -> pd.DataFrame:
        """
        split: 'train', 'valid', 'test'
        """
        if split == "train":
            csv_name = self.TRAIN_CSV_NAME
        elif split == "valid":
            csv_name = self.VALID_CSV_NAME
        else:
            csv_name = self.TEST_CSV_NAME

        csv_path = self.dataset_root / split / csv_name
        logger.info(f"Reading {split} CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        df.columns = df.columns.str.strip() 

        # ensure filename column + labels exist
        missing = [c for c in self.LABEL_COLUMNS + ["filename"] if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {split} CSV: {missing}")

        # build full image paths
        img_dir = self.dataset_root / split
        df["filepath"] = df["filename"].apply(lambda fn: os.path.join(img_dir, fn))

        return df

    def _df_to_tfdata(self, df: pd.DataFrame, shuffle: bool = True) -> tf.data.Dataset:
        filepaths = df["filepath"].tolist()
        labels = df[self.LABEL_COLUMNS].values.astype("float32")

        paths_tensor = tf.constant(filepaths)
        labels_tensor = tf.constant(labels, dtype=tf.float32)

        def _process(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, self.image_size)
            img = img / 255.0  # normalise
            return img, label

        ds = tf.data.Dataset.from_tensor_slices((paths_tensor, labels_tensor))
        ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            ds = ds.shuffle(buffer_size=len(filepaths))

        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def _create_datasets(self):
        train_df = self._load_split_df("train")
        valid_df = self._load_split_df("valid")
        test_df = self._load_split_df("test")

        train_ds = self._df_to_tfdata(train_df, shuffle=True)
        valid_ds = self._df_to_tfdata(valid_df, shuffle=False)
        test_ds = self._df_to_tfdata(test_df, shuffle=False)

        return train_ds, valid_ds, test_ds

    # ---------- model training ----------

    def _load_model(self) -> keras.Model:
        """
        Load the updated base model (VGG16 + ANN head)
        """
        model_path = self.prepare_base_model_artifact.updated_base_model_path
        logger.info(f"Loading updated base model from: {model_path}")
        model = keras.models.load_model(model_path)
        logger.info("Model loaded successfully.")
        return model

    def initiate_model_training(self) -> ModelTrainingArtifact:
        """
        Orchestrates dataset creation + training + saving final model.
        """
        logger.info("=== Stage 03: Model Training started ===")

        # create datasets
        train_ds, valid_ds, test_ds = self._create_datasets()

        # load model
        model = self._load_model()

        # for multi-label classification:
        # final layer must have sigmoid activation and binary_crossentropy loss
        logger.info("Re-compiling model for multi-label training.")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.params.LEARNING_RATE),
            loss="binary_crossentropy",
            metrics=["binary_accuracy"],
        )

        logger.info(
            f"Starting training for {self.epochs} epochs, "
            f"batch size {self.batch_size}, image size {self.image_size}"
        )

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=self.epochs,
        )

        logger.info("Evaluating on test set.")
        test_metrics = model.evaluate(test_ds)
        logger.info(f"Test metrics: {test_metrics}")

        # save final trained model
        trained_model_path = self.config.trained_model_path
        logger.info(f"Saving trained model to: {trained_model_path}")
        model.save(trained_model_path)

        logger.info("=== Stage 03: Model Training completed ===")

        return ModelTrainingArtifact(
            trained_model_path=trained_model_path,
        )
