import os
import shutil


def find_the_label(filename):
    return filename.split('_')[1]


def iterate_over_folder(folder_list, destiny_folder):
    for folder in folder_list:
        folder_path = os.path.join(path, folder)
        for file in os.listdir(folder_path):
            label = find_the_label(file)
            origin_file_path = os.path.join(folder_path, file)
            destiny_folder_path = os.path.join(destiny_folder, label)
            os.makedirs(destiny_folder_path, exist_ok=True)
            shutil.move(origin_file_path, os.path.join(destiny_folder_path, file))


def generate_dataset(path, train_folder_path, test_folder_path):
    '''
    Take care that the linux ordering of os listdir might be different than the windows one. Linux output:
        Train folders: ['handgesturedataset_part1', 'handgesturedataset_part4', 'handgesturedataset_part2', 'handgesturedataset_part5']
        Test folders: ['handgesturedataset_part3']
    '''
    train_folders = os.listdir(path)[:-1]
    test_folders = os.listdir(path)[-1:]
    print('Train folders: {}\nTest folders: {}'.format(train_folders, test_folders))
    iterate_over_folder(train_folders, train_folder_path)
    iterate_over_folder(test_folders, test_folder_path)


if __name__ == '__main__':
    path = '../../data/classifier/original_data'
    train_folder_path = '../../data/classifier/train'
    test_folder_path = '../../data/classifier/test'
    os.makedirs(train_folder_path, exist_ok=True)
    os.makedirs(test_folder_path, exist_ok=True)
    generate_dataset(path, train_folder_path, test_folder_path)
