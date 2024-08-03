import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'ProductionLineVerification'

list_of_files = [
    "bluewasher",
    "memory_logs",
    "research",
    "yellowasher",
    "resources",
    f"{project_name}/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/config/configuration.py",
]

for file_path in list_of_files:
    file_path_obj = Path(file_path)
    filedir, filename = os.path.split(file_path_obj)

    if filedir != "":
        filedir_obj = Path(filedir)
        filedir_obj.mkdir(parents=True, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if not file_path_obj.exists():
        with open(file_path_obj, 'w') as f:
            logging.info(f'Creating empty file {filename}')
    else:
        logging.info(f'{filename} already exists')
