import numpy as np
import open3d as o3d
import cv2
# TODO test, GPT generated

# def pointcloud2equirectangular(pointcloud, viewpoint, resolution=360):
#     # Convert the Open3D point cloud to a numpy array
#     np_points = np.asarray(pointcloud.points)

#     # Get the range information from the point cloud
#     x, y, z = np.transpose(np_points)
#     d = np.sqrt(np.power(x - viewpoint[0], 2) + np.power(y - viewpoint[1], 2) + np.power(z - viewpoint[2], 2))

#     # Convert the range information to spherical coordinates
#     theta = np.arctan2(y, x)
#     phi = np.arccos(z / d)

#     # Map the spherical coordinates to the equirectangular image
#     width = resolution
#     # height = int(width / 2)
#     height = 45
#     equirectangular = np.zeros((height, width))
#     x_map = (theta / (2 * np.pi)) * width
#     y_map = (phi / np.pi) * height
#     for i, j in zip(x_map, y_map):
#         if i >= 0 and i < width and j >= 0 and j < height:
#             equirectangular[int(np.round(j)), int(np.round(i))] = np.mean(d)
#     return equirectangular

def pointcloud2equirectangular(pointcloud, viewpoint, resolution=360):
    # Convert the Open3D point cloud to a numpy array
    np_points = np.asarray(pointcloud.points)

    # Get the range information from the point cloud
    x, y, z = np.transpose(np_points)
    d = np.sqrt(np.power(x - viewpoint[0], 2) + np.power(y - viewpoint[1], 2) + np.power(z - viewpoint[2], 2))

    # Convert the range information to spherical coordinates
    theta = np.arctan2(y, x)
    phi = np.arccos(z / d)

    # Map the spherical coordinates to the equirectangular image
    width = resolution
    # height = int(width / 2)
    height = 45
    equirectangular = np.zeros((height, width))
    x_map = (theta / (2 * np.pi)) * width
    y_map = (phi / np.pi) * height
    for i, j, depth in zip(x_map, y_map, d):
        if i >= 0 and i < width and j >= 0 and j < height:
            if equirectangular[int(np.round(j)), int(np.round(i))] == 0:
                equirectangular[int(np.round(j)), int(np.round(i))] = depth
    
    equirectangular = equirectangular / np.max(equirectangular)
    equirectangular = np.repeat(equirectangular[:, :, np.newaxis], 3, axis=2)
    equirectangular = cv2.applyColorMap(np.uint8(equirectangular * 255), cv2.COLORMAP_JET)
    return equirectangular


pointcloud = o3d.io.read_point_cloud("/home/yashturkar/Syncspace/final_results_reg/Mai_City/Puma/Puma_3_100k.ply")
o3d.visualization.draw_geometries([pointcloud])

np_points = np.asarray(pointcloud.points)

center = np.mean(np_points, axis=0)

print (center)
img = pointcloud2equirectangular(pointcloud, (100,-0.2,0.5))

cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()