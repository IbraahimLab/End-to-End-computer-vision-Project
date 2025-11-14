from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.components.prepare_base_model import PrepareBaseModel

from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig)

    
from cnnClassifier.entity.artifact_entity import (
    DataIngestionArtifact,
    PrepareBaseModelArtifact,
)
from cnnClassifier.logger.logging import logger


class TrainingPipeline:
    def __init__(self):
        pass

    def start_data_ingestion(self) -> DataIngestionArtifact:
        logger.info("Entered start_data_ingestion of TrainingPipeline")

        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        logger.info("Completed data ingestion in TrainingPipeline")
        return data_ingestion_artifact

    def start_prepare_base_model(self) -> PrepareBaseModelArtifact:
        logger.info("Entered start_prepare_base_model of TrainingPipeline")

        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()

        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model_artifact = prepare_base_model.initiate_prepare_base_model()

        logger.info("Completed prepare_base_model in TrainingPipeline")
        return prepare_base_model_artifact

    def main(self):
        try:
            logger.info("=== Training Pipeline started ===")

            # Stage 01
            _ = self.start_data_ingestion()

            # Stage 02
            _ = self.start_prepare_base_model()

            logger.info("=== Training Pipeline finished (Stages 1 & 2) ===")

        except Exception as e:
            logger.error(e)
            raise e
