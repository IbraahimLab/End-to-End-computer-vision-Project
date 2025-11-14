from cnnClassifier.pipeline.training_pipeline import DataIngestionTrainingPipeline
from cnnClassifier.logger.logging import logger

if __name__ == "__main__":
    try:
        obj = DataIngestionTrainingPipeline()
        obj.main()
    except Exception as e:
        logger.error(e)
        raise e
