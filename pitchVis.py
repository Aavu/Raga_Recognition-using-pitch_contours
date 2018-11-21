import glob

import h5py
import numpy as np

shankarabharanam_pitch_files = glob.glob('dataset/shankarabharanam/*.pitch')
shankarabharanam_pitch_files.sort()
shankarabharanam_tonic_files = glob.glob('dataset/shankarabharanam/*.tonicFine')
shankarabharanam_tonic_files.sort()

shanmukapriya_pitch_files = glob.glob('dataset/shanmukapriya/*.pitch')
shanmukapriya_pitch_files.sort()
shanmukapriya_tonic_files = glob.glob('dataset/shanmukapriya/*.tonicFine')
shanmukapriya_tonic_files.sort()

todi_pitch_files = glob.glob('dataset/todi/*.pitch')
todi_pitch_files.sort()
todi_tonic_files = glob.glob('dataset/todi/*.tonicFine')
todi_tonic_files.sort()

maya_pitch_files = glob.glob('dataset/mayamalavagowla/*.pitch')
maya_pitch_files.sort()
maya_tonic_files = glob.glob('dataset/mayamalavagowla/*.tonicFine')
maya_tonic_files.sort()

tonic_shankarabharanam = []
for item in shankarabharanam_tonic_files:
    data = open(item).read()
    # print(len(data))
    tonic_shankarabharanam.append(float(data))
tonic_shankarabharanam = np.array(tonic_shankarabharanam)

# print(tonic_shankarabharanam)

tonic_shanmukapriya = []
for item in shanmukapriya_tonic_files:
    data = open(item).read()
    # print(len(data))
    tonic_shanmukapriya.append(float(data))
tonic_shanmukapriya = np.array(tonic_shanmukapriya)

# print(tonic_shanmukapriya)

tonic_todi = []
for item in todi_tonic_files:
    data = open(item).read()
    # print(len(data))
    tonic_todi.append(float(data))
tonic_todi = np.array(tonic_todi)

# print(tonic_todi)

tonic_maya = []
for item in maya_tonic_files:
    data = open(item).read()
    # print(len(data))
    tonic_maya.append(float(data))
tonic_maya = np.array(tonic_maya)

# print(tonic_maya)

shankarabharanam = []
for item in shankarabharanam_pitch_files:
    data = open(item).readlines()
    # print(len(data))
    pitch = []
    for x in range(len(data)):
        t, p = data[x].split("\t")
        pitch.append(float(p))
    shankarabharanam.append(np.array(pitch))

shankarabharanam = np.array(shankarabharanam)

# print(shankarabharanam.shape)

shanmukapriya = []
for item in shanmukapriya_pitch_files:
    data = open(item).readlines()
    # print(len(data))
    pitch = []
    for x in range(len(data)):
        t, p = data[x].split("\t")
        pitch.append(float(p))
    shanmukapriya.append(np.array(pitch))

shanmukapriya = np.array(shanmukapriya)

# print(shanmukapriya.shape)

todi = []
for item in todi_pitch_files:
    data = open(item).readlines()
    # print(len(data))
    pitch = []
    for x in range(len(data)):
        t, p = data[x].split("\t")
        pitch.append(float(p))
    todi.append(np.array(pitch))

todi = np.array(todi)

maya = []
for item in maya_pitch_files:
    data = open(item).readlines()
    # print(len(data))
    pitch = []
    for x in range(len(data)):
        t, p = data[x].split("\t")
        pitch.append(float(p))
    maya.append(np.array(pitch))

maya = np.array(maya)

eps = 1e-15
shanmukapriya_scaled = np.array([])
for i in range(len(tonic_shanmukapriya)):
    c = np.log2(((shanmukapriya[i] + eps) / tonic_shanmukapriya[i]))
    c = np.array(c)
    c[c < 0] = 0
    shanmukapriya_scaled = np.concatenate((shanmukapriya_scaled, c))

shanmukapriya_scaled = np.array(shanmukapriya_scaled)

shankarabharanam_scaled = np.array([])
for i in range(len(tonic_shankarabharanam)):
    c = np.log2(((shankarabharanam[i] + eps) / tonic_shankarabharanam[i]))
    c = np.array(c)
    c[c < 0] = 0
    shankarabharanam_scaled = np.concatenate((shankarabharanam_scaled, c))

todi_scaled = np.array([])
for i in range(len(tonic_todi)):
    c = np.log2(((todi[i] + eps) / tonic_todi[i]))
    c = np.array(c)
    c[c < 0] = 0
    todi_scaled = np.concatenate((todi_scaled, c))

maya_scaled = np.array([])
for i in range(len(tonic_maya)):
    c = np.log2(((maya[i] + eps) / tonic_maya[i]))
    c = np.array(c)
    c[c < 0] = 0
    maya_scaled = np.concatenate((maya_scaled, c))

length = 230 * 5
split_shankarabharanam = int(len(shankarabharanam_scaled) / length)
split_shanmukapriya = int(len(shanmukapriya_scaled) / length)
split_todi = int(len(todi_scaled) / length)
split_maya = int(len(maya_scaled) / length)

shankarabharanam_cropped = np.split(shankarabharanam_scaled[:split_shankarabharanam * length], split_shankarabharanam)
shanmukapriya_cropped = np.split(shanmukapriya_scaled[:split_shanmukapriya * length], split_shanmukapriya)
todi_cropped = np.split(todi_scaled[:split_todi * length], split_todi)
maya_cropped = np.split(maya_scaled[:split_maya * length], split_maya)
print(len(shankarabharanam_cropped), len(shanmukapriya_cropped), len(todi_cropped), len(maya_cropped))

with h5py.File("dataset.h5", 'w') as hdf:
    g1 = hdf.create_group('shankarabharanam')
    for i in range(len(shankarabharanam_cropped)):
        g1.create_dataset('shankarabharanam_' + str(i), data=shankarabharanam_cropped[i])

    g2 = hdf.create_group('shanmukapriya')
    for i in range(len(shanmukapriya_cropped)):
        g2.create_dataset('shanmukapriya_' + str(i), data=shanmukapriya_cropped[i])

    g3 = hdf.create_group('todi')
    for i in range(len(todi_cropped)):
        g3.create_dataset('todi_' + str(i), data=todi_cropped[i])

    g4 = hdf.create_group('maya')
    for i in range(len(maya_cropped)):
        g4.create_dataset('maya_' + str(i), data=maya_cropped[i])

# plt.plot(shankarabharanam_cropped[10])
# plt.tight_layout()
# plt.show()
