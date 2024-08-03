import os 
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO , format = '[%(asctime)s]: %(message)s:')

project_name = 'ProductionLineVerification'

list_of_files = [
    "bluewasher",
    "memory_logs",
    "research",
    "yellowasher",
    "resources",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

]
for file_path in list_of_files:
    file_path = Path(file_path)
    filedir , filename =os.path.split(file_path)

    if filedir != "":
        os.makedirs(filedir , exist_ok=True)
        logging.info(f"Creating directory:{filedir} for the file {filename}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path , 'w') as f:
            pass
            logging.info(f'creating empty file {filename}')

    else:
        logging.info(f'{filename} already exsists')

