import os
import sys
sys.path.append('..')
import paths

partition_file_path = os.path.join(paths.data_folder_path, 'Eval', 'list_eval_partition.txt')
identity_file_path = os.path.join(paths.data_folder_path, 'Anno', 'identity_celeba.txt')
bbox_file_path = os.path.join(paths.data_folder_path, 'Anno', 'list_bbox_celeba.txt')
with open(partition_file_path) as partition_file, open(identity_file_path) as identity_file, open(bbox_file_path) as bbox_file:
    partition_file_rows = partition_file.read().splitlines()
    identity_file_rows = identity_file.read().splitlines()
    bbox_file_rows = bbox_file.read().splitlines()

master_file_path = os.path.join(paths.data_folder_path, 'master.csv')
with open(master_file_path, 'w') as master_file:
    header = ['image_file_name', 'dataset_type', 'identity', 'anchor_x', 'anchor_y', 'width', 'height']
    master_file.write(','.join(header) + '\n')
    for i in range(len(partition_file_rows)):
        row = []
        row += partition_file_rows[i].split(' ')
        row.append(identity_file_rows[i].split(' ')[1])
        bboxes_row_elements = bbox_file_rows[i].split(' ')
        for j in range(len(bboxes_row_elements)):
            if j > 0 and bboxes_row_elements[j] != '':
                row.append(bboxes_row_elements[j])

        master_file.write(','.join(row) + '\n')
