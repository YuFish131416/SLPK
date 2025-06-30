import struct

import DracoPy as draco
import numpy as np
import open3d as o3d


class GeometryProcessor:
    """处理几何数据的类"""

    def transform(self, data, translation, simplify_ratio=0):
        """转换几何坐标系（保留三维信息）"""
        try:
            # 如果是文件路径，读取数据
            if isinstance(data, str):
                with open(data, 'rb') as f:
                    data = f.read()

            # 1. 解码几何数据
            vertices, faces = self.parse_geometry(data)

            if vertices is None or len(vertices) == 0:
                return None

            # 2. 应用坐标系转换（三维平移）
            vertices = vertices + translation

            # 3. 重新编码为Draco格式
            if faces is not None and len(faces) > 0:
                # 三角形网格
                encoded = draco.encode(vertices, faces=faces)
            else:
                # 点云
                encoded = draco.encode(vertices, faces=None)

            return encoded

        except Exception as e:
            print(f"几何转换失败: {str(e)}")
            return None

    def parse_geometry(self, data):
        """解析几何数据"""
        # 尝试解析为Draco格式
        if data.startswith(b'DRAC'):
            mesh = draco.decode(data)
            vertices = np.array(mesh.points)
            faces = np.array(mesh.faces).reshape(-1, 3) if mesh.faces else None
            return vertices, faces

        # 尝试解析为I3S标准二进制格式
        try:
            # I3S二进制格式参考：
            # 头部：4字节特征数量 + 4字节顶点数量
            feature_count = struct.unpack('<I', data[0:4])[0]
            vertex_count = struct.unpack('<I', data[4:8])[0]

            # 检查数据长度
            expected_length = 8 + vertex_count * 12
            if len(data) < expected_length:
                return None, None

            # 提取顶点
            vertices = []
            offset = 8
            for _ in range(vertex_count):
                x, y, z = struct.unpack('<fff', data[offset:offset + 12])
                vertices.append([x, y, z])
                offset += 12

            return np.array(vertices), None
        except:
            return None, None

    def simplify_mesh(self, vertices, faces, ratio):
        """三维感知的网格简化（保留空间特征）"""
        try:
            # 创建Open3D网格对象
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)

            if faces is not None:
                mesh.triangles = o3d.utility.Vector3iVector(faces)

            # 计算目标三角形数量
            original_count = len(mesh.triangles) if mesh.has_triangles() else len(vertices)
            target_count = max(1, int(original_count * (1 - ratio)))

            # 使用Quadric Error Metrics简化
            simplified_mesh = mesh.simplify_quadric_decimation(target_count)

            # 获取简化后的顶点和面
            simplified_vertices = np.asarray(simplified_mesh.vertices)
            simplified_faces = np.asarray(simplified_mesh.triangles) if simplified_mesh.has_triangles() else None

            return simplified_vertices, simplified_faces

        except Exception as e:
            print(f"三维网格简化失败: {str(e)}")
            return vertices, faces
