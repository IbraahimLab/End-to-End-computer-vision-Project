from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.logger.logging import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info("Stage 01: Data Ingestion started")

            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()

            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion.initiate_data_ingestion()

            logger.info("Stage 01: Data Ingestion completed")

        except Exception as e:
            logger.error(e)
            raise e
