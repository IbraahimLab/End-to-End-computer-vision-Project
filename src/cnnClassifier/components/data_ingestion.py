import gdown
import zipfile
from pathlib import Path
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier.entity.artifact_entity import DataIngestionArtifact
from cnnClassifier.utils.common import create_directories
from cnnClassifier.logger.logging import logger

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.config = data_ingestion_config

    def download_from_gdrive(self) -> Path:
        """
        Download ZIP from Google Drive based on file ID.
        """
        try:
            url = f"https://drive.google.com/uc?id={self.config.gdrive_file_id}"
            output_path = str(self.config.local_data_file)

            logger.info(f"Downloading dataset from Google Drive: {url}")
            logger.info(f"Saving to: {output_path}")

            gdown.download(url, output_path, quiet=False)

            logger.info("Download completed successfully.")
            return self.config.local_data_file

        except Exception as e:
            raise e

    def extract_zip_file(self, zip_path: Path = None) -> Path:
        """
        Extract the downloaded ZIP.
        """
        try:
            if zip_path is None:
                zip_path = self.config.local_data_file

            unzip_dir = self.config.unzip_dir
            create_directories([unzip_dir])

            logger.info(f"Extracting: {zip_path}")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzip_dir)

            logger.info(f"Extraction completed: {unzip_dir}")
            return unzip_dir

        except Exception as e:
            raise e

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Run download + extraction.
        """
        logger.info("=== Starting Data Ingestion ===")

        zip_path = self.download_from_gdrive()
        unzip_path = self.extract_zip_file(zip_path)

        artifact = DataIngestionArtifact(
            zip_file_path=zip_path,
            unzip_dir=unzip_path
        )

        logger.info("Data Ingestion Artifact created successfully.")
        logger.info(f"{artifact}")

        return artifact
