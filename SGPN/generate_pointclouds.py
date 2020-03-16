import numpy as np
import math
import matplotlib.pyplot as plt
import h5py
import random

# x y z xn yn zn l sl

# data:
  # rectangle1:  x y z, x y z ...
  # circle1:  x y z, x y z ...
  # ...
# label
  # rectangle1: 0 0, 0 0
  # circle1: 1 0, 1 0

def divisible_random(a,b,n):
    if b-a < n:
      raise Exception('{} is too big'.format(n))
    result = random.randint(a, b)
    while result % n != 0:
      result = random.randint(a, b)
    return result


def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom


def populate_line(line, tmp_line, displacement, side):
    for idx, element in enumerate(tmp_line):
        x_noise = random.uniform(-1, 1)
        y_noise = random.uniform(-1, 1)
        if side:
            line[idx][0] = float(displacement) + x_noise
            line[idx][1] = float(element) + y_noise
        else:
            line[idx][0] = float(element) + x_noise
            line[idx][1] = float(displacement) + y_noise


def create_random_rectangle(total_points, bVisualize = False):
    w = random.randrange(10, 100)
    h = random.randrange(10, 100)

    bl_x = random.randrange(200, 400)
    bl_y = random.randrange(200, 400)

    tl_x = bl_x
    tl_y = bl_y + h

    tr_x = bl_x + w
    tr_y = bl_y + h

    br_x = bl_x + w
    br_y = bl_y

    nr_pr_line = int(total_points/4)

    b_line_tmp = np.linspace(bl_x, br_x, num=nr_pr_line)
    b_line = np.zeros((len(b_line_tmp), 6))
    populate_line(b_line, b_line_tmp, displacement=bl_y, side=False)

    t_line_tmp = np.linspace(tl_x, tr_x, num=nr_pr_line)
    t_line = np.zeros((len(t_line_tmp), 6))
    populate_line(t_line, t_line_tmp, displacement=tl_y, side=False)

    l_line_tmp = np.linspace(bl_y, tl_y, num=nr_pr_line)
    l_line = np.zeros((len(l_line_tmp), 6))
    populate_line(l_line, l_line_tmp, displacement=bl_x, side=True)

    r_line_tmp = np.linspace(br_y, tr_y, num=nr_pr_line)
    r_line = np.zeros((len(r_line_tmp), 6))
    populate_line(r_line, r_line_tmp, displacement=br_x, side=True)

    rectangle = np.concatenate((b_line, t_line), axis=0)
    rectangle = np.concatenate((rectangle, l_line), axis=0)
    rectangle = np.concatenate((rectangle, r_line), axis=0)

    vect_ones = np.ones(rectangle.shape[0])
    rectangle[:,2] = vect_ones

    # Normalize everything
    '''max_x = max(rectangle[:, 0])
    max_y = max(rectangle[:, 1])

    for coor_idx, coord in enumerate(rectangle):
        rectangle[coor_idx][0] = coord[0] / max_x
        rectangle[coor_idx][1] = coord[1] / max_y'''

    return rectangle

def generate_random_circle_pc(total_points, bVisualize = False):
    radius = random.randrange(10, 50)
    x_center = random.randrange(200, 400)
    y_center = random.randrange(200, 400)
    normalizing_constant = 150

    circle_points = np.zeros((total_points, 6))
    for j in range(0, total_points):
        # random angle
        alpha = 2 * math.pi * random.random()
        # calculating coordinates
        x_noise = random.uniform(-1, 1)
        y_noise = random.uniform(-1, 1)
        x = radius * math.cos(alpha) + x_center + x_noise
        y = radius * math.sin(alpha) + y_center + y_noise
        point = (x, y, 1, 0, 0, 0)
        circle_points[j] = point

    # Normalize everything
    '''max_x = max(circle_points[:, 0])
    max_y = max(circle_points[:, 1])

    for coor_idx, coord in enumerate(circle_points):
        circle_points[coor_idx][0] = coord[0] / max_x
        circle_points[coor_idx][1] = coord[1] / max_y'''

    return circle_points

def create_hdf5_file(data, inst_label, seglabel, name):
    folder_name = "data/inst_seg/"
    file_name = name + ".hdf5"
    f = h5py.File(folder_name + file_name, "w")
    f.create_dataset("data", data=data)
    f.create_dataset("inst_label", data=inst_label)
    f.create_dataset("seglabel", data=seglabel)
    f.close()

    return file_name

def create_npy_file(data, inst_label, seglabel, name):
    folder_name = "data/inst_seg/"
    file_name = name + ".hdf5"
    f = h5py.File(folder_name + file_name, "w")
    f.create_dataset("data", data=data)
    f.create_dataset("inst_label", data=inst_label)
    f.create_dataset("seglabel", data=seglabel)
    f.close()

    return file_name


def room_creator():
    num_of_rect = random.randint(2, 4)
    num_of_circles = random.randint(2, 4)
    rectangles = []
    circles = []
    for idx in range(0, num_of_rect):
        total_points_rec = divisible_random(100, 300, 4)
        rectangles.append(create_random_rectangle(total_points_rec, bVisualize=True))
        # file_shape.write("rectangle" + "\n")
    for idx in range(0, num_of_circles):
        total_points_circle = random.randint(100, 300)
        circles.append(generate_random_circle_pc(total_points_circle, bVisualize=True))
        # file_shape.write("circle" + "\n")

    # Create data arrays
    data = np.empty((0, 6), float)
    seg_label = np.empty((0, 1), int)
    inst_label = np.empty((0, 1), int)
    for idx, circle in enumerate(circles):
        data = np.append(data, circle, axis=0)
        seg_label = np.append(seg_label, np.full((circle.shape[0], 1), 0), axis=0)
        inst_label = np.append(inst_label, np.full((circle.shape[0], 1), idx), axis=0)
    for idx, rectangle in enumerate(rectangles):
        data = np.append(data, rectangle, axis=0)
        seg_label = np.append(seg_label, np.full((rectangle.shape[0], 1), 1), axis=0)
        inst_label = np.append(inst_label, np.full((rectangle.shape[0], 1), idx), axis=0)

    # Resampling
    #idx = np.random.randint(data.shape[0], size=nr_samples)
    #idx = random.sample(range(0, (data.shape[0]-1)), nr_samples)
    #idx = np.sort(idx)
    #data = data[idx, :]
    #seg_label = seg_label[idx, :]
    #inst_label = inst_label[idx, :]

    room_data = np.empty((data.shape[0], 8), float)
    room_data[:, 0:6] = data
    xyz_min = np.amin(data, axis=0)[0:3]
    room_data[:, 0:3] -= xyz_min
    room_data[:, 6] = np.squeeze(seg_label)
    room_data[:, 7] = np.squeeze(inst_label)

    return room_data


def main():
    nr_rooms = 2
    #nr_samples = 1000
    all_room_names = open('data/generated_pointclouds/all_room_names.txt', "w+")

    path = 'data/generated_pointclouds/annotations/'
    # Create rooms
    for room_idx in range(nr_rooms):
        room_data = room_creator()
        room_name = 'room_' + str(room_idx)
        full_path = path + room_name + '.npy'
        # Save .npy file for each room
        np.save(full_path, room_data)
        all_room_names.write(room_name + '.npy' + "\n")
    all_room_names.close()

    '''
    # Data split
    # Splits must be even numbers and add to 1
    train_split = int(nr_rooms*0.6)
    valid_split = int(nr_rooms*0.2)
    test_split = int(nr_rooms*0.2)

    train_data = rooms_data[:train_split]
    train_inst_label = rooms_inst_label[:train_split]
    train_seglabel = rooms_seglabel[:train_split]
    #create_hdf5_file(train_data, train_inst_label, train_seglabel, name="synth_train")

    valid_data = rooms_data[train_split:train_split+valid_split]
    valid_inst_label = rooms_inst_label[train_split:train_split+valid_split]
    valid_seglabel = rooms_seglabel[train_split:train_split+valid_split]
    #create_hdf5_file(valid_data, valid_inst_label, valid_seglabel, name="synth_valid")

    test_data = rooms_data[train_split+valid_split:]
    test_inst_label = rooms_inst_label[train_split+valid_split:]
    test_seglabel = rooms_seglabel[train_split+valid_split:]
    #create_hdf5_file(test_data, test_inst_label, test_seglabel, name="synth_test")
    '''

    '''file_all = open("data/inst_seg/all_files.txt", "w")
    file_shape = open("data/inst_seg/shape_filelist.txt", "w")
    num_of_rect = random.randint(0, 9)
    num_of_circles = random.randint(0, 9)
    total_points_circle = random.randint(100, 200)
    total_points_rec = divisible_random(100, 200, 4)
    all_rectangles = np.zeros((num_of_rect, total_points_rec, 3))
    all_circles = np.zeros((num_of_circles, total_points_circle, 3))

    for idx in range(0, num_of_rect):
        total_points_rec = divisible_random(100, 200, 4)
        all_rectangles[idx] = create_random_rectangle(total_points_rec, bVisualize=True)
        file_shape.write("rectangle" + "\n")

    for idx in range(0, num_of_circles):
        total_points_circle = random.randint(100, 200)
        all_circles[idx] = generate_random_circle_pc(total_points_circle, bVisualize=True)
        file_shape.write("circle" + "\n")


    label_rect = np.zeros((np.size(all_rectangles, 0), total_points_rec))
    label_rect.fill(0)
    label_circle = np.zeros((np.size(all_circles, 0), total_points_circle))
    label_circle.fill(1)

    all_labels = np.concatenate((label_rect, label_circle), axis=0)
    all_data = np.concatenate((all_rectangles, all_circles), axis=0)

    file_name = create_hdf5_file(all_data, all_labels, name="rect_and_circles")

    file_all.write("my_data/" + file_name + "\n")

    file_all.close()
    file_shape.close()'''

if __name__ == "__main__":
    main()