import os
import numpy
from random import shuffle

PATH = './data/ISIC2018_Task1_npy_all/image'
SAVE_PATH = './Datasets'


def create_5_floder(folder, save_foler):
    file_list = os.listdir(folder)
    shuffle(file_list)

    for i in range(5):
        if i != 0:
            pre_test_list = file_list[0:i*518]
        else:
            pre_test_list = []
        test_list = file_list[i*518:(i+1)*518]

        if i < 4:
            valid_list = file_list[(i+1)*518:(i+1)*518+260]
            train_list = file_list[(i+1)*518+260:] + pre_test_list
        else:
            valid_list = file_list[-4:] + file_list[:256]
            train_list = file_list[256:i*518]

        if not os.path.isdir(save_foler + '/folder'+str(i+1)):
            os.makedirs(save_foler + '/folder'+str(i+1))

        text_save(os.path.join(save_foler, 'folder'+str(i+1), 'folder'+str(i+1)+'_train.list'), train_list)
        text_save(os.path.join(save_foler, 'folder'+str(i+1), 'folder'+str(i+1)+'_validation.list'), valid_list)
        text_save(os.path.join(save_foler, 'folder'+str(i+1), 'folder'+str(i+1)+'_test.list'), test_list)


def text_save(filename, data):      # filename: path to write CSV, data: data list to be written.
    file = open(filename, 'w+')
    for i in range(len(data)):
        s = str(data[i]).replace('[', '').replace(']', '')
        s = s.replace("'", '').replace(',', '') + '\n'
        file.write(s)
    file.close()
    print("Save {} successfully".format(filename.split('/')[-1]))


if __name__ == "__main__":
    create_5_floder(PATH, SAVE_PATH)
