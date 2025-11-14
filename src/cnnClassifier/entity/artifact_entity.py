from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionArtifact:
    zip_file_path: Path
    unzip_dir: Path



@dataclass(frozen=True)
class PrepareBaseModelArtifact:
    base_model_path: Path
    updated_base_model_path: Path

    