import os
import inspect

current_file_path = inspect.getfile(inspect.currentframe())

project_folder_path = os.path.abspath(os.path.join(current_file_path, '..', '..'))
src_folder_path = os.path.abspath(os.path.join(project_folder_path, 'src'))
data_folder_path = os.path.abspath(os.path.join(project_folder_path, 'data'))
