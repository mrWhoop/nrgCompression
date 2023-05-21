import os
import sys
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
#
# BVP
#
# miranda = np.fromfile('miranda_1024x1024x1024_float32-010.raw', dtype='float32')
#
# mirandaSV = miranda.reshape(1024, 1024, 1024)
#
# print(mirandaSV)
#
# np.save('miranda3d', mirandaSV)

# data = np.zeros((1024*1024*1024, 4))
#
#
# i = 0
#
# for x in range(len(miranda)):
#     for y in range(len(miranda[x])):
#         for z in range(len(miranda[x][y])):
#             data[i][0] = x
#             data[i][1] = y
#             data[i][2] = z
#             data[i][3] = miranda[x][y][z]
#             i += 1
#
# np.save('miranda', data)
#
# foot = np.fromfile('foot_256x256x256_1x1x1_uint8.raw', dtype='uint8')
#
# print(len(foot))
#
# foot = foot.reshape(256, 256, 256)
# np.save('foot3d', foot)
#
# print(foot)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = np.nonzero(foot)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Plot')
#
# ax.scatter(x, y, z, s=0.1, c=foot[x, y, z], cmap='jet', vmin=0, vmax=255)
#
# plt.show()


# engine = np.fromfile('engine_256x256x256_1x1x1_uint8.raw', dtype='uint8')
#
# print(len(engine))
#
# engine = engine.reshape(256,256,256)
# np.save('engine3d', engine)
#
# print(engine)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = np.nonzero(engine)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Plot')
#
# ax.scatter(x, y, z, s=0.1, c=engine[x, y, z], cmap='jet', vmin=0, vmax=255)
#
# plt.show()

skull = np.fromfile('skull_256x256x256_1x1x1_uint8.raw', dtype='uint8')

print(len(skull))

skull = skull.reshape(256,256,256)
np.save('skull3d', skull)

print(skull)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = np.nonzero(skull)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot')

ax.scatter(x, y, z, s=0.1, c=skull[x, y, z], cmap='jet', vmin=0, vmax=255)

plt.show()

# bucky = np.fromfile('bucky_32x32x32_1x1x1_uint8.raw', dtype='uint8')
# bucky = np.fromfile('engine_256x256x256_1x1x1_uint8.raw', dtype='uint8')
#
# print(len(bucky))
#
# # bucky = bucky.reshape(32, 32, 32)
# bucky = bucky.reshape(256, 256, 256)
# np.save('engine3d', bucky)
#
# print(bucky)
#
# bucky[bucky == 48] = 0
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = np.nonzero(bucky)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Plot')
#
# ax.scatter(x, y, z, c=bucky[x, y, z], cmap='jet')
#
# plt.show()

# box = np.fromfile('box_64x64x64_1x1x1_uint8.raw', dtype='uint8')
#
# print(len(box))
#
# box = box.reshape(64, 64, 64)
# np.save('box3d', box)
#
# print(box)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x, y, z = np.nonzero(box)
#
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Plot')
#
# ax.scatter(x, y, z, s=0.1, c=box[x, y, z], cmap='jet', vmin=0, vmax=256)
#
# plt.show()
