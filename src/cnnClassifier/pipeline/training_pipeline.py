from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier.components.model_training import ModelTraining


from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,)


from cnnClassifier.entity.artifact_entity import (
    DataIngestionArtifact,
    PrepareBaseModelArtifact,
    ModelTrainingArtifact
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






    def start_model_training(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        prepare_base_model_artifact: PrepareBaseModelArtifact,
    ) -> ModelTrainingArtifact:
        logger.info("Entered start_model_training of TrainingPipeline")

        config = ConfigurationManager()
        training_config = config.get_training_config()

        model_trainer = ModelTraining(
            config=training_config,
            params=config.params,
            data_ingestion_artifact=data_ingestion_artifact,
            prepare_base_model_artifact=prepare_base_model_artifact,
        )

        model_training_artifact = model_trainer.initiate_model_training()

        logger.info("Completed model_training in TrainingPipeline")
        return model_training_artifact
    








    def main(self):
        try:
            logger.info("=== Training Pipeline started ===")

            # Stage 01
            data_ingestion_artifact= self.start_data_ingestion()

            # Stage 02
            prepare_base_model_artifact = self.start_prepare_base_model()



             # Stage 03
            _ = self.start_model_training(
                data_ingestion_artifact=data_ingestion_artifact,
                prepare_base_model_artifact=prepare_base_model_artifact,
            )

            logger.info("=== Training Pipeline finished (Stages 1 & 2) ===")

        except Exception as e:
            logger.error(e)
            raise e
