import json
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from mpl_toolkits.mplot3d import Axes3D
from math import atan2
from matplotlib import cm


def convert_view_to_lat_long(points):
    """
    :param a list of points [[x, y, z],]
    :return: an array of (经度,纬度)
    """
    lat_longs = np.empty(shape=(points.shape[0], 2))
    for i in range(len(points)):
        azimuth, elevation = viewpoint_to_azimuth_and_elevation(points[i])
        lat_longs[i] = np.array([azimuth, elevation])

    return lat_longs


def viewpoint_to_azimuth_and_elevation(point):

    x, y, z = point
    azimuth = atan2(y, x)
    elevation = atan2(z, (x ** 2 + y ** 2))
    return azimuth, elevation


with open("scene_gt.json", 'r') as f:
    json_dict = json.loads(f.read())

viewpoints = list()
orientations = list()

obj_interest = 1
for label in json_dict.values():
    for obj in label:
        # if obj['obj_id'] != obj_interest:
        #     continue
        R = np.array(obj['cam_R_m2c']).reshape(3, 3)
        T = np.array(obj['cam_t_m2c']).reshape(3, 1) / 1000.0
        cam_position = R.dot(T).reshape(3)
        cam_position /= np.linalg.norm(cam_position)
        viewpoints.append(cam_position)

        # orientation = R.T.dot(np.array([1, 0, 0])).reshape(3)
        # orientation /= np.linalg.norm(orientation)
        # orientations.append(orientation)

fig = plt.figure()
ax = Axes3D(fig=fig)

viewpoints = np.array(viewpoints)
# orientations = np.array(orientations)

ax.scatter(viewpoints[:, 0], viewpoints[:, 1], viewpoints[:, 2], color='r', s=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


# examples can be found at https://scikit-learn.org/stable/modules/density.html
kde = KernelDensity(bandwidth=0.04, metric='haversine',
                    kernel='gaussian', algorithm='ball_tree')


Xtrain = convert_view_to_lat_long(viewpoints)
print(f"Azimuth range:{min(Xtrain[:, 0]) / math.pi * 180}° - {max(Xtrain[:, 0]) / math.pi * 180}°")
print(f"Elevation range:{min(Xtrain[:, 1]) / math.pi * 180}° - {max(Xtrain[:, 1]) / math.pi * 180}°")
print(Xtrain.shape)
kde.fit(Xtrain)

n_theta = 200  # number of values for theta
n_phi = 200  # number of values for phi
r = 1  # radius of sphere

theta, phi = np.mgrid[-0.5 * np.pi:0.5 * np.pi:n_theta * 1j, 0.0:2.0 * np.pi:n_phi * 1j]
x = r * np.cos(theta) * np.cos(phi)
y = r * np.cos(theta) * np.sin(phi)
z = r * np.sin(theta)

inp = []

for j in phi[0, :]:
    for i in theta[:, 0]:
        # i, j is elevation and azimuth in radian
        val = kde.score_samples(np.array([[j, i], ]))[0]
        inp.append([j, i, val])
inp = np.array(inp)

# reshape the input array to the shape of the x,y,z arrays.
c = inp[:, 2].reshape((n_phi, n_theta)).T

# Set colours and render
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# normalize the value
c /= c.max()
# visit https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html to choose facecolors
surf = ax.plot_surface(
    x, y, z, rstride=1, cstride=1, facecolors=cm.jet(c), alpha=1,
    linewidth=0, antialiased=False, shade=False)  # cm.hot(c/c.max()) shape is (50, 200

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_title('viewpoints distribtion heatmap')

m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(c)
plt.colorbar(m, shrink=0.8)
plt.savefig("heapmap.png")
plt.show()
