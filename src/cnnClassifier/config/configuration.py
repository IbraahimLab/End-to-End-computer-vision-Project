from pathlib import Path
from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import * 
from cnnClassifier.entity.config_entity import DataIngestionConfig

CONFIG_FILE_PATH = Path("config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config["data_ingestion"]

        root_dir = Path(config["root_dir"])
        local_data_file = Path(config["local_data_file"])
        unzip_dir = Path(config["unzip_dir"])

        # Your Google Drive file id
        gdrive_file_id = "1XaNxpHP3XwDyKjEw-1wirLcgLqMRSsV-"

        create_directories([root_dir, unzip_dir, local_data_file.parent])

        return DataIngestionConfig(
            root_dir=root_dir,
            local_data_file=local_data_file,
            unzip_dir=unzip_dir,
            gdrive_file_id=gdrive_file_id,
        )
