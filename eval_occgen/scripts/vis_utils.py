import xml.etree.ElementTree as ET

import math
import numpy as np
import open3d as o3d
import pyrender
import trimesh


def parse_xml(xml_string):
    root = ET.fromstring(xml_string)
    camera_node = root.find('VCGCamera')

    focal_mm = float(camera_node.attrib['FocalMm'])
    pixel_size_mm = tuple(map(float, camera_node.attrib['PixelSizeMm'].split()))
    viewport_px = tuple(map(int, camera_node.attrib['ViewportPx'].split()))
    sensor_size = [pixel_size_mm[0] * viewport_px[0], pixel_size_mm[1] * viewport_px[1]]
    fov_y = 2 * math.atan((sensor_size[1] / 2) / focal_mm)
    camera = pyrender.PerspectiveCamera(yfov=fov_y)

    translation_vector = np.array(list(map(float, camera_node.attrib['TranslationVector'].split()[:3])))
    rotation_values = list(map(float, camera_node.attrib['RotationMatrix'].split()))
    rotation_matrix = np.array(rotation_values).reshape(4, 4)[:3, :3]
    rotation_matrix = rotation_matrix.T

    camera_pose = np.eye(4)
    camera_pose[:3, :3] = rotation_matrix
    camera_pose[:3, 3] = -translation_vector
    return camera, camera_pose


def get_mesh_trimesh(voxel_map, shape, ignore, color_map, voxel_size, ignores=None):
    voxels = voxel_map != ignore

    if ignores is not None:
        for val in ignores:
            voxels &= voxel_map != val

    occupied_voxels = np.argwhere(voxels)

    mesh_list = []

    for position in occupied_voxels:
        color = color_map[voxel_map[tuple(position)]][::-1]
        voxel = trimesh.creation.box(extents=(voxel_size, voxel_size, voxel_size))
        voxel.apply_translation(position)
        voxel.visual.vertex_colors = np.tile(color, (8, 1))
        mesh_list.append(voxel)

    combined_mesh = trimesh.util.concatenate(mesh_list)
    return combined_mesh, pyrender.Mesh.from_trimesh(combined_mesh, smooth=False)


def get_mesh_open3d(voxel_map, shape, ignore, color_map, voxel_size, ignores=None):
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    positions = np.stack([y, x, z], axis=-1).reshape(-1, 3)

    mask = voxel_map.reshape(-1) != ignore

    if ignores is not None:
        for val in ignores:
            mask &= voxel_map.reshape(-1) != val

    positions = positions[mask]
    colors = np.array([color_map[voxel_map.reshape(-1)[i]] for i in np.where(mask)[0]]) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions * voxel_size)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, ::-1])  # RGB to BGR

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    mesh = o3d.geometry.TriangleMesh()
    voxels = voxel_grid.get_voxels()
    for voxel in voxels:
        cube = o3d.geometry.TriangleMesh.create_box(width=voxel_size, height=voxel_size, depth=voxel_size)
        cube.paint_uniform_color(voxel.color)
        cube.translate(voxel.grid_index * voxel_size)
        mesh += cube

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)

    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
    return trimesh_mesh, pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=False)
