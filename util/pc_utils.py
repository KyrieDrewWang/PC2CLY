import numpy as np
from plyfile import PlyData, PlyElement
import open3d as o3d

def read_ply(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    return vertex

def read_ply_norms(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        xx = np.array(plydata['normals']['x'])
        yy = np.array(plydata['normals']['y'])
        zz = np.array(plydata['normals']['z'])
        vertex = np.stack([x, y, z, xx, yy, zz], axis=1)
    return vertex  
    

def read_ply_bbox(path):
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    pcd = o3d.io.read_point_cloud(path)
    aabb = pcd.get_axis_aligned_bounding_box()
    min_point = aabb.min_bound
    max_point = aabb.max_bound
    return vertex, min_point, max_point

def read_ply_norm_cal(path, radius=0.05, max_nn=30):
    """
    radius: the search radius
    max_nn: the maximum number of points to estimate the norm
    """
    with open(path, 'rb') as f:
        plydata = PlyData.read(f)
        x = np.array(plydata['vertex']['x'])
        y = np.array(plydata['vertex']['y'])
        z = np.array(plydata['vertex']['z'])
        vertex = np.stack([x, y, z], axis=1)
    # pcd = o3d.io.read_point_cloud(path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertex)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    # pc_with_norms = np.hstack((vertex, np.array(pcd.normals)))
    # import pdb;pdb.set_trace()
    vertex = np.concatenate([vertex, np.array(pcd.normals)], -1)
    return vertex

        

def write_ply_norm(pc, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(pc[i, 0], pc[i, 1], pc[i, 2]) for i in range(pc.shape[0])]
    norms = [(pc[i, 3], pc[i, 4], pc[i, 5]) for i in range(pc.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    normals = np.array(norms, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    nl = PlyElement.describe(normals, 'normals')
    with open(filename, mode='wb') as f:
        PlyData([el, nl], text=text).write(f)


def write_ply(points, filename, text=False):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    with open(filename, mode='wb') as f:
        PlyData([el], text=text).write(f)

if __name__ == "__main__":
    path = "/comp_robot/wangcheng/DeepCAD/data/pc_cad/0000/00000007.ply"
    path = "/comp_robot/wangcheng/DeepCAD/test.ply"
    pc = read_ply_norms(path)
    write_ply_norm(pc, "test.ply")
    
    print(type(pc))