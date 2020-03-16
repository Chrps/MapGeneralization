import os
import numpy as np
from SGPN.data import data_prep_util
from SGPN.data import indoor3d_util

# Constants
PointCloudPath = 'generated_pointclouds'
NUM_POINT = 50
data_dtype = 'float32'
label_dtype = 'int32'

# Set paths
filelist = os.path.join(PointCloudPath, 'all_room_names.txt')
data_label_files = [os.path.join(PointCloudPath, 'annotations/', line.rstrip()) for line in open(filelist)]
output_dir = os.path.join(PointCloudPath, 'training_data')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_room_filelist = os.path.join(PointCloudPath, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')
train_list = open('generated_pointclouds/train_hdf5_file_list.txt', 'w')

sample_cnt = 0
for i in range(0, len(data_label_files)):
    data_label_filename = data_label_files[i]
    fname = os.path.basename(data_label_filename).strip('.npy')
    if not os.path.exists(data_label_filename):
        continue
    data, label, inslabel = indoor3d_util.room2blocks_wrapper_normalized(data_label_filename, NUM_POINT, block_size=70.0, stride=35.0,
                                                 random_sample=False, sample_num=None)
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    h5_filename = os.path.join(output_dir, '%s.h5' % fname)
    print('{0}: {1}, {2}, {3}'.format(h5_filename, data.shape, label.shape, inslabel.shape))
    data_prep_util.save_h5ins(h5_filename,
                              data,
                              label,
                              inslabel,
                              data_dtype, label_dtype)
    train_list.write('data/generated_pointclouds/training_data/' + fname + '.h5' + '\n')

fout_room.close()
train_list.close()
print("Total samples: {0}".format(sample_cnt))