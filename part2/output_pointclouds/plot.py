import numpy as np
import matplotlib.pyplot as plt

# Load .ply file
from plyfile import PlyData
ply_data = PlyData.read("merged_pointcloud.ply")  # Replace with your .ply file path

# Extract coordinates (assuming a point cloud structure)
x = ply_data['vertex']['x']
y = ply_data['vertex']['y']
z = ply_data['vertex']['z']

# Plot the points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1)  # s=1 sets the size of points
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
