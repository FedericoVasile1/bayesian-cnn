import numpy as np
import os

def string_to_int_conversion():
    """
    Converts the elements in the arrays real_classes_train.npy and real_classes_test.npy from string
    to int based on the above table.
    The newer arrays are saved in the same folder as real_classes_{train|test}_int.npy
    "AC"      => 0
    "AD"      => 1
    "H"       => 2
    "blood"   => 3
    "fat"     => 4
    "glass"   => 5
    "stroma"  => 6
    """
    phases = ['train', 'test']
    for phase in phases:
        real_classes = np.load(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'real_classes_'+phase+'.npy'))
        real_classes = np.where(real_classes=='AC', 0, real_classes)
        real_classes = np.where(real_classes == 'AD', 1, real_classes)
        real_classes = np.where(real_classes == 'H', 2, real_classes)
        real_classes = np.where(real_classes == 'blood', 3, real_classes)
        real_classes = np.where(real_classes == 'fat', 4, real_classes)
        real_classes = np.where(real_classes == 'glass', 5, real_classes)
        real_classes = np.where(real_classes == 'stroma', 6, real_classes)

        real_classes = real_classes.astype('uint8')
        np.save(os.path.join(os.getcwd(), 'data', 'crc_3_noisy', 'real_classes_'+phase+'_int.npy'), real_classes)

if __name__ == '__main__':
    base_dir = os.getcwd()
    base_dir = base_dir.split('/')[-1]
    if base_dir != 'Project8':
        raise Exception('Wrong base dir, this file must be run from Project8/ directory.')

    string_to_int_conversion()