from pathlib import Path
from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import * 
from cnnClassifier.entity.config_entity import DataIngestionConfig ,PrepareBaseModelConfig

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



    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        params = self.params

        root_dir = Path(config.root_dir)
        base_model_path = Path(config.base_model_path)
        updated_base_model_path = Path(config.updated_base_model_path)

        create_directories([root_dir, base_model_path.parent, updated_base_model_path.parent])

        return PrepareBaseModelConfig(
            root_dir=root_dir,
            base_model_path=base_model_path,
            updated_base_model_path=updated_base_model_path,
            params_image_size=params.IMAGE_SIZE,
            params_learning_rate=params.LEARNING_RATE,
            params_include_top=params.INCLUDE_TOP,
            params_weights=params.WEIGHTS,
            params_classes=params.CLASSES,
        )
