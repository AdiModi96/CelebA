import os
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
    header_row = ['image_file_name', 'dataset_type', 'identity', 'bbox_tl_x', 'bbox_tl_y', 'bbox_br_x', 'bbox_br_y']
    master_file.write(','.join(header_row) + '\n')
    for i in range(len(partition_file_rows)):
        row = []
        row += partition_file_rows[i].split(' ')
        row.append(identity_file_rows[i].split(' ')[1])
        bboxes_row_elements = bbox_file_rows[i].split(' ')
        bbox = []
        for j in range(len(bboxes_row_elements)):
            if j > 0 and bboxes_row_elements[j] != '':
                bbox.append(bboxes_row_elements[j])
        bbox[2] = str(int(bbox[0]) + int(bbox[2]))
        bbox[3] = str(int(bbox[1]) + int(bbox[3]))
        row += bbox

        master_file.write(','.join(row) + '\n')
