from cnnClassifier.pipeline.training_pipeline import TrainingPipeline
from cnnClassifier.logger.logging import logger

if __name__ == "__main__":
    try:
        pipeline = TrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.error(e)
        raise e
