import gzip
import json
import struct
import tempfile
import os
import argparse
import zipfile
import numpy as np
from PIL import Image
from sklearn.cluster import DBSCAN
import shutil
import draco
import open3d as o3d
from scipy.spatial.transform import Rotation


def optimize_slpk_nodes(slpk_path, output_path,
                        min_points=100,
                        merge_distance=5.0,
                        simplify_ratio=0.3,
                        texture_max_size=1024,
                        lod_scale=1.5):
    """
    优化I3S SLPK节点结构 (符合1.8规范)

    参数:
    slpk_path: 输入SLPK文件路径
    output_path: 输出优化后SLPK路径
    min_points: 最小点数阈值（低于此值合并）
    merge_distance: 节点合并距离阈值(米)
    simplify_ratio: 几何简化比例(0-1)
    texture_max_size: 纹理最大尺寸
    lod_scale: LOD过渡缩放因子
    """
    # 创建临时工作目录
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"解压SLPK文件到临时目录: {temp_dir}")
        # 解压SLPK文件
        extract_slpk(slpk_path, temp_dir)

        # 读取3dSceneLayer.json
        scene_layer_path = os.path.join(temp_dir, "3dSceneLayer.json.gz")
        if not os.path.exists(scene_layer_path):
            # 尝试未压缩版本
            scene_layer_path = os.path.join(temp_dir, "3dSceneLayer.json")
            if not os.path.exists(scene_layer_path):
                raise FileNotFoundError("3dSceneLayer.json not found in SLPK")

        # 解压或直接读取场景层文件
        if scene_layer_path.endswith('.gz'):
            import gzip
            with gzip.open(scene_layer_path, 'rb') as f_in:
                scene_layer_data = json.load(f_in)
            # 解压后保存为普通json以便修改
            uncompressed_path = os.path.join(temp_dir, "3dSceneLayer.json")
            with open(uncompressed_path, 'w') as f_out:
                json.dump(scene_layer_data, f_out, indent=2)
            scene_layer_path = uncompressed_path
        else:
            with open(scene_layer_path, 'r') as f:
                scene_layer_data = json.load(f)

        print("调整LOD过渡...")
        adjust_lod_transitions(scene_layer_data, lod_scale)

        # 保存修改后的场景层文件
        with open(scene_layer_path, 'w') as f:
            json.dump(scene_layer_data, f, indent=2)

        # 收集所有节点
        nodes_dir = os.path.join(temp_dir, "nodes")
        if not os.path.exists(nodes_dir):
            raise FileNotFoundError("nodes directory not found in SLPK")

        # 获取所有节点目录 - 更健壮的过滤
        node_ids = []
        for d in os.listdir(nodes_dir):
            full_path = os.path.join(nodes_dir, d)
            if os.path.isdir(full_path):
                # 接受数字形式或UUID形式的节点ID
                if d.isdigit() or (len(d) == 32 and all(c in "0123456789abcdef" for c in d.lower())):
                    node_ids.append(d)
                else:
                    print(f"跳过无效节点目录: {d} (非标准ID)")

        print(f"发现 {len(node_ids)} 个节点目录")

        # 空间聚类 (基于边界框中心)
        positions = []
        node_bboxes = []
        valid_nodes = 0
        invalid_nodes = 0

        for node_id in node_ids:
            node_dir = os.path.join(nodes_dir, node_id)
            index_path = os.path.join(node_dir, "3dNodeIndexDocument.json.gz")
            # 检查文件是否存在
            if not os.path.exists(index_path):
                invalid_nodes += 1
                continue
            if index_path.endswith('.gz'):
                import gzip
                with gzip.open(index_path, 'rb') as f_in:
                    index_path_data = json.load(f_in)
                # 解压后保存为普通json以便修改
                uncompressed_path = os.path.join(temp_dir, "3dSceneLayer.json")
                with open(uncompressed_path, 'w') as f_out:
                    json.dump(index_path_data, f_out, indent=2)
                index_path = uncompressed_path

            try:
                with open(index_path, 'r', encoding='utf-8') as f:
                    try:
                        node_data = json.load(f)
                    except json.JSONDecodeError:
                        # 尝试处理可能的GZIP压缩
                        with open(index_path, 'rb') as bin_f:
                            try:
                                decompressed = gzip.decompress(bin_f.read())
                                node_data = json.loads(decompressed.decode('utf-8'))
                            except:
                                print(f"节点 {node_id} 的索引文件格式无效")
                                invalid_nodes += 1
                                continue

                # 尝试不同格式的边界框
                center = None

                # 尝试获取OBB（定向边界框）
                if 'obb' in node_data:
                    obb = node_data['obb']
                    if 'center' in obb and len(obb['center']) >= 3:
                        center = np.array(obb['center'][:3])

                # 尝试获取MBS（最小包围球）
                elif 'mbs' in node_data:
                    mbs = node_data['mbs']
                    if isinstance(mbs, list) and len(mbs) >= 4:
                        center = np.array(mbs[:3])

                # 尝试获取几何范围
                if center is None and 'geometry' in node_data:
                    geom = node_data['geometry']
                    if 'bbox' in geom and len(geom['bbox']) >= 6:
                        # 计算边界框中心: [minx, miny, minz, maxx, maxy, maxz]
                        minx, miny, minz, maxx, maxy, maxz = geom['bbox'][:6]
                        center = np.array([(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2])

                if center is not None:
                    positions.append(center)
                    node_bboxes.append((node_id, node_data))
                    valid_nodes += 1
                else:
                    print(f"节点 {node_id} 缺少有效的边界框信息")
                    invalid_nodes += 1

            except Exception as e:
                print(f"处理节点 {node_id} 时出错: {str(e)}")
                invalid_nodes += 1

        print(f"有效节点: {valid_nodes}, 无效节点: {invalid_nodes}")

        if len(positions) == 0:
            print("未找到任何有效边界框信息")
            # repack_slpk(temp_dir, output_path)
            return

        positions = np.array(positions)

        # 空间聚类 (基于DBSCAN)
        print(f"执行空间聚类 (距离阈值: {merge_distance}米)...")
        dbscan = DBSCAN(eps=merge_distance, min_samples=1).fit(positions)
        labels = dbscan.labels_
        unique_clusters = len(set(labels))
        print(f"发现 {unique_clusters} 个空间聚类")

        # 按聚类分组节点
        clusters = {}
        for i, (node_id, node_data) in enumerate(node_bboxes):
            cluster_id = labels[i]
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append((node_id, node_data))

        # 创建新节点 - 移出循环外
        nodes_to_remove = []
        new_nodes_count = 0
        merged_nodes_count = 0
        processed_nodes = 0

        # 使用items()直接迭代
        for cluster_id, group in clusters.items():
            processed_nodes += len(group)

            # 小节点直接合并
            if len(group) > 1 or estimate_vertex_count(group[0][1]) < min_points:
                merged_nodes = [node_id for node_id, _ in group]
                new_node_id = f"merged_{cluster_id}"

                # 优化2: 批量计算中心点
                centers = [get_node_center(node_data) for _, node_data in group]
                cluster_center = np.mean(centers, axis=0)

                success = merge_i3s_nodes(temp_dir, merged_nodes, new_node_id,
                                          simplify_ratio, cluster_center)

                if success:
                    new_nodes_count += 1
                    merged_nodes_count += len(merged_nodes)
                    nodes_to_remove.extend(merged_nodes)
            else:
                # 大节点简化后保留
                node_id, node_data = group[0]
                node_center = get_node_center(node_data)
                success = simplify_i3s_node(temp_dir, node_id, simplify_ratio, node_center)
                if success:
                    new_nodes_count += 1

            # 批量删除节点（移出循环）
            if processed_nodes % 100 == 0 or processed_nodes == len(node_bboxes):
                print(f"已处理 {processed_nodes}/{len(node_bboxes)} 个节点")

        # 批量删除节点（提高效率）
        nodes_dir = os.path.join(temp_dir, "nodes")
        for node_id in nodes_to_remove:
            node_path = os.path.join(nodes_dir, node_id)
            if os.path.exists(node_path):
                shutil.rmtree(node_path)

        # 优化纹理
        print(f"优化纹理 (最大尺寸: {texture_max_size}px)...")
        optimize_i3s_textures(temp_dir, max_resolution=texture_max_size)

        # 重新打包为SLPK
        print("重新打包为SLPK...")
        repack_slpk(temp_dir, output_path)

        print(f"优化完成: 合并 {merged_nodes_count} 个节点为 {new_nodes_count} 个新节点")


def extract_slpk(slpk_path, output_dir):
    """解压SLPK文件"""
    try:
        with zipfile.ZipFile(slpk_path, 'r') as z:
            z.extractall(output_dir)
        return True
    except Exception as e:
        print(f"解压SLPK文件失败: {str(e)}")
        return False


def repack_slpk(source_dir, output_path):
    """重新打包为SLPK文件"""
    try:
        # 确保输出路径是字符串
        output_path = os.fspath(output_path)
        source_dir = os.fspath(source_dir)

        # 处理3dSceneLayer.json：压缩并重命名
        scene_layer_path = os.path.join(source_dir, "3dSceneLayer.json")
        if os.path.exists(scene_layer_path):
            # 创建临时压缩文件
            temp_gz = tempfile.NamedTemporaryFile(delete=False, suffix='.gz')
            try:
                # 读取原始JSON并压缩到临时文件
                with open(scene_layer_path, 'rb') as f_in, \
                        gzip.GzipFile(fileobj=temp_gz, mode='wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                temp_gz.close()  # 确保数据写入磁盘

                # 用压缩内容替换原文件（保持文件名不变）
                os.remove(scene_layer_path)
                shutil.move(temp_gz.name, scene_layer_path)
            finally:
                if os.path.exists(temp_gz.name):
                    os.unlink(temp_gz.name)  # 确保清理临时文件

        # 压缩所有节点索引文档
        nodes_dir = os.path.join(source_dir, "nodes")
        if os.path.exists(nodes_dir):
            for node_id in os.listdir(nodes_dir):
                node_dir = os.path.join(nodes_dir, node_id)
                if os.path.isdir(node_dir):
                    # 压缩3dNodeIndexDocument.json
                    index_path = os.path.join(node_dir, "3dNodeIndexDocument.json")
                    if os.path.exists(index_path):
                        # 压缩文件
                        with open(index_path, 'rb') as f_in:
                            compressed_data = gzip.compress(f_in.read())

                        # 写入压缩文件
                        gz_path = index_path + ".gz"
                        with open(gz_path, 'wb') as f_out:
                            f_out.write(compressed_data)

                        # 删除原始文件
                        os.remove(index_path)

                    # 压缩共享资源文件
                    shared_path = os.path.join(node_dir, "shared", "sharedResource.json")
                    if os.path.exists(shared_path):
                        # 压缩文件
                        with open(shared_path, 'rb') as f_in:
                            compressed_data = gzip.compress(f_in.read())

                        # 写入压缩文件
                        gz_path = shared_path + ".gz"
                        with open(gz_path, 'wb') as f_out:
                            f_out.write(compressed_data)

                        # 删除原始文件
                        os.remove(shared_path)

        # 创建ZIP包（自动处理文件路径）
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as z:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = str(os.path.join(root, file))
                    arcname = os.path.relpath(file_path, source_dir)

                    # 在ZIP中保留原始文件名（不包括.gz后缀）
                    if file.endswith('.gz'):
                        # 对于压缩文件，在ZIP中存储时不带.gz后缀
                        original_name = file[:-3]  # 移除.gz后缀
                        original_path = os.path.join(root, original_name)

                        # 检查是否已经存在未压缩版本（应该不存在）
                        if not os.path.exists(original_path):
                            # 直接添加压缩文件，但使用原始文件名
                            z.write(file_path, os.path.relpath(original_path, source_dir))
                        else:
                            # 如果存在冲突，优先使用压缩文件
                            z.write(file_path, arcname)
                    else:
                        # 对于非压缩文件，正常添加
                        z.write(file_path, arcname)
        return True
    except Exception as e:
        print(f"重新打包SLPK失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def get_node_center(node_data):
    """获取节点的中心点坐标（三维）"""
    if 'obb' in node_data:
        obb = node_data['obb']
        if 'center' in obb and len(obb['center']) >= 3:
            return np.array(obb['center'][:3])

    if 'mbs' in node_data:
        mbs = node_data['mbs']
        if isinstance(mbs, list) and len(mbs) >= 4:
            return np.array(mbs[:3])

    if 'geometry' in node_data:
        geom = node_data['geometry']
        if 'bbox' in geom and len(geom['bbox']) >= 6:
            minx, miny, minz, maxx, maxy, maxz = geom['bbox'][:6]
            return np.array([(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2])

    # 默认返回原点
    return np.array([0, 0, 0])


def estimate_vertex_count(node_data):
    """估算节点中的顶点数"""
    # 从几何资源中估算
    if 'geometryData' in node_data:
        for geom in node_data['geometryData']:
            if 'resource' in geom:
                resource = geom['resource']
                if 'vertexCount' in resource:
                    return resource['vertexCount']

    # 从特征数据中估算
    if 'featureData' in node_data:
        for feat in node_data['featureData']:
            if 'resource' in feat and 'vertexCount' in feat['resource']:
                return feat['resource']['vertexCount']

    return 0


def merge_i3s_nodes(temp_dir, node_ids, new_node_id, simplify_ratio, common_center):
    """合并多个I3S节点为一个新节点"""
    try:
        nodes_dir = os.path.join(temp_dir, "nodes")
        new_node_dir = os.path.join(nodes_dir, new_node_id)
        os.makedirs(new_node_dir, exist_ok=True)

        # 创建新节点目录结构
        for subdir in ['features', 'geometries', 'shared', 'textures']:
            os.makedirs(os.path.join(new_node_dir, subdir), exist_ok=True)

        # 合并边界框
        combined_obb = None

        # 合并几何和特征
        geometry_index = 0
        feature_index = 0
        texture_index = 0

        # 创建新节点索引文档
        new_index_data = {
            "version": "1.8",
            "obb": None,
            "geometryData": [],
            "featureData": [],
            "textureData": [],
            "attributeData": [],
            "sharedResource": {
                "resource": "sharedResource.json.gz"
            }
        }

        # 创建共享资源
        shared_resource = {
            "geometryDefinitions": [],
            "materialDefinitions": []
        }

        # 创建新节点的OBB（定向边界框）
        new_obb = {
            "center": common_center.tolist(),
            "halfSize": [10, 10, 10],  # 初始值，后续会更新
            "quaternion": [0, 0, 0, 1]  # 无旋转
        }
        new_index_data["obb"] = new_obb

        for node_id in node_ids:
            node_dir = os.path.join(nodes_dir, node_id)
            index_path = os.path.join(node_dir, "3dNodeIndexDocument.json")

            if not os.path.exists(index_path):
                continue

            with open(index_path, 'r') as f:
                node_data = json.load(f)

            # 合并边界框
            if 'obb' in node_data:
                if combined_obb is None:
                    combined_obb = node_data['obb'].copy()
                else:
                    # 简化：取并集
                    pass

            # 合并几何数据
            # 获取当前节点的局部坐标系
            node_center = get_node_center(node_data)

            # 计算从节点局部坐标系到公共坐标系的变换
            translation = node_center - common_center

            # 合并几何数据（带坐标系转换）
            if 'geometryData' in node_data:
                for geom in node_data['geometryData']:
                    geom_file = geom['resource']['href']
                    src_path = os.path.join(node_dir, geom_file)

                    if os.path.exists(src_path):
                        # 读取并转换几何数据
                        transformed_geom = transform_geometry_file(
                            src_path,
                            translation,
                            simplify_ratio
                        )

                        if transformed_geom:
                            new_geom_file = f"geometries/{geometry_index}.bin"
                            dst_path = os.path.join(new_node_dir, new_geom_file)
                            with open(dst_path, 'wb') as f_out:
                                f_out.write(transformed_geom)

                            # 更新几何资源
                            new_geom_resource = {
                                "href": new_geom_file,
                                "vertexCount": geom['resource']['vertexCount']  # 估算
                            }
                            new_index_data['geometryData'].append({
                                "resource": new_geom_resource
                            })
                            geometry_index += 1

            # 合并特征数据
            if 'featureData' in node_data:
                for feat in node_data['featureData']:
                    feat_file = feat['resource']['href']
                    src_path = os.path.join(node_dir, feat_file)
                    if os.path.exists(src_path):
                        new_feat_file = f"features/{feature_index}.json.gz"
                        dst_path = os.path.join(new_node_dir, new_feat_file)
                        shutil.copy(src_path, dst_path)

                        # 更新特征资源
                        new_feat_resource = feat['resource'].copy()
                        new_feat_resource['href'] = new_feat_file
                        new_index_data['featureData'].append({
                            "resource": new_feat_resource
                        })
                        feature_index += 1

            # 合并纹理
            if 'textureData' in node_data:
                for tex in node_data['textureData']:
                    tex_file = tex['resource']['href']
                    src_path = os.path.join(node_dir, tex_file)
                    if os.path.exists(src_path):
                        new_tex_file = f"textures/{texture_index}.jpg"
                        dst_path = os.path.join(new_node_dir, new_tex_file)
                        shutil.copy(src_path, dst_path)

                        # 更新纹理资源
                        new_tex_resource = tex['resource'].copy()
                        new_tex_resource['href'] = new_tex_file
                        new_index_data['textureData'].append({
                            "resource": new_tex_resource
                        })
                        texture_index += 1

        # 保存共享资源
        shared_resource_path = os.path.join(new_node_dir, "shared", "sharedResource.json")
        with open(shared_resource_path, 'w') as f:
            json.dump(shared_resource, f, indent=2)

        # 压缩共享资源
        import gzip
        with open(shared_resource_path, 'rb') as f_in:
            with gzip.open(shared_resource_path + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(shared_resource_path)
        new_index_data['sharedResource']['resource'] = "shared/sharedResource.json.gz"

        # 保存新节点索引文档
        index_path = os.path.join(new_node_dir, "3dNodeIndexDocument.json")
        with open(index_path, 'w') as f:
            json.dump(new_index_data, f, indent=2)

        return True

    except Exception as e:
        print(f"合并节点失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def transform_geometry_file(file_path, translation, simplify_ratio=0):
    """转换几何坐标系并简化（保留三维信息）"""
    try:
        with open(file_path, 'rb') as f:
            data = f.read()

        # 1. 解码几何数据
        if data.startswith(b'DRAC'):
            mesh = draco.decode(data)
            vertices = np.array(mesh.points)
            faces = np.array(mesh.faces).reshape(-1, 3) if mesh.faces else None
        else:
            # 尝试解析为I3S标准二进制格式
            vertices, faces = parse_i3s_binary(data)

        if vertices is None or len(vertices) == 0:
            return None

        # 2. 应用坐标系转换（三维平移）
        vertices = vertices + translation

        # 3. 三维感知的几何简化
        if 0 < simplify_ratio < 1:
            vertices, faces = simplify_mesh_3d(vertices, faces, simplify_ratio)

        # 4. 重新编码为Draco格式
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


def parse_i3s_binary(data):
    """解析I3S二进制格式（包含Z值）"""
    try:
        # I3S二进制格式参考：
        # 头部：4字节特征数量 + 4字节顶点数量
        # 顶点数据：每个顶点12字节（XYZ坐标） + 其他属性
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

        # 转换为numpy数组
        return np.array(vertices), None

    finally:
        return None, None


def simplify_mesh_3d(vertices, faces, ratio):
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


def simplify_i3s_node(temp_dir, node_id, simplify_ratio, node_center):
    """简化单个I3S节点（三维感知）"""
    try:
        nodes_dir = os.path.join(temp_dir, "nodes")
        node_dir = os.path.join(nodes_dir, node_id)
        index_path = os.path.join(node_dir, "3dNodeIndexDocument.json")
        geometry_modified = False

        if not os.path.exists(index_path):
            return False

        with open(index_path, 'r') as f:
            node_data = json.load(f)

        # 处理几何数据
        if 'geometryData' in node_data:
            for geom in node_data['geometryData']:
                geom_file = geom['resource']['href']
                src_path = os.path.join(node_dir, geom_file)

                if os.path.exists(src_path):
                    # 简化几何（使用三维感知方法）
                    simplified = transform_geometry_file(
                        src_path,
                        np.array([0, 0, 0]),  # 不转换位置
                        simplify_ratio
                    )

                    if simplified:
                        # 覆盖原始几何文件
                        with open(src_path, 'wb') as f_out:
                            f_out.write(simplified)
                        geometry_modified = True

        return geometry_modified

    except Exception as e:
        print(f"简化节点失败: {str(e)}")
        return False


def simplify_geometry_file(file_path, simplify_ratio):
    """简化几何文件 (Draco压缩格式)"""
    try:
        # 读取几何文件
        with open(file_path, 'rb') as f:
            data = f.read()

        # 如果是Draco格式，尝试解码
        if draco and data.startswith(b'DRAC'):
            # 使用draco-py解码
            mesh = draco.decode(data)

            # 简化网格 - 这里使用简单的顶点采样
            # 在实际应用中应使用网格简化算法
            num_vertices = len(mesh.points)
            num_to_keep = int(num_vertices * (1 - simplify_ratio))

            if num_to_keep < 3:
                return None

            # 随机采样保留顶点 (简化示例)
            indices = np.random.choice(num_vertices, num_to_keep, replace=False)
            indices.sort()

            # 创建新网格
            new_points = mesh.points[indices]

            # 更新面 - 需要更复杂的处理
            # 这里省略，实际应用中应使用网格简化库

            # 编码回Draco
            encoded = draco.encode(new_points, faces=None)
            return encoded

        return None

    except Exception as e:
        print(f"简化几何失败: {str(e)}")
        return None


def optimize_i3s_textures(temp_dir, max_resolution=1024):
    """优化I3S纹理"""
    nodes_dir = os.path.join(temp_dir, "nodes")
    if not os.path.exists(nodes_dir):
        return

    node_dirs = [d for d in os.listdir(nodes_dir)
                 if os.path.isdir(os.path.join(nodes_dir, d))]

    for node_dir in node_dirs:
        textures_dir = os.path.join(nodes_dir, node_dir, "textures")
        if not os.path.exists(textures_dir):
            continue

        for file in os.listdir(textures_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                file_path = os.path.join(textures_dir, file)
                try:
                    with Image.open(file_path) as img:
                        width, height = img.size
                        max_dim = max(width, height)

                        if max_dim > max_resolution:
                            # 计算新尺寸
                            ratio = max_resolution / max_dim
                            new_size = (int(width * ratio), int(height * ratio))

                            # 调整大小并保存
                            resized = img.resize(new_size, Image.Resampling.LANCZOS)

                            # 保存为JPEG以节省空间
                            output_file = os.path.splitext(file)[0] + ".jpg"
                            output_path = os.path.join(textures_dir, output_file)
                            resized.save(output_path, "JPEG", quality=85)

                            # 删除原始文件（如果是不同格式）
                            if file != output_file:
                                os.remove(file_path)

                            print(f"优化纹理: {file} {width}x{height} -> {new_size[0]}x{new_size[1]}")
                except Exception as e:
                    print(f"处理纹理 {file} 时出错: {str(e)}")


def adjust_lod_transitions(scene_layer_data, transition_scale=1.5):
    """调整LOD过渡阈值"""
    # 调整场景层的LOD设置
    if 'lodSelection' in scene_layer_data and isinstance(scene_layer_data['lodSelection'], list):
        for lod_sel in scene_layer_data['lodSelection']:
            if isinstance(lod_sel, dict) and 'maxError' in lod_sel:
                lod_sel['maxError'] = max(1, lod_sel['maxError'] * transition_scale)

    # 调整节点页的LOD设置 - 添加类型检查
    if 'nodePages' in scene_layer_data and isinstance(scene_layer_data['nodePages'], list):
        for node_page in scene_layer_data['nodePages']:
            # 确保 node_page 是字典类型
            if not isinstance(node_page, dict):
                continue  # 跳过非字典条目

            if 'lodSelection' in node_page and isinstance(node_page['lodSelection'], list):
                for lod_sel in node_page['lodSelection']:
                    if isinstance(lod_sel, dict) and 'maxError' in lod_sel:
                        # 确保新值至少为1
                        lod_sel['maxError'] = max(1, lod_sel['maxError'] * transition_scale)


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='I3S SLPK文件优化工具')

    # 位置参数（可选，带默认值）
    parser.add_argument('input', type=str, nargs='?', default="JP3.slpk",
                        help='输入SLPK文件路径（默认: %(default)s）')
    parser.add_argument('output', type=str, nargs='?', default="output.slpk",
                        help='输出SLPK文件路径（默认: %(default)s）')

    # 可选参数
    parser.add_argument('--min_points', '-m', type=int, default=200,
                        help='最小点数阈值（低于此值合并，默认: %(default)s）')
    parser.add_argument('--merge_distance', '-d', type=float, default=10.0,
                        help='节点合并距离阈值(米)（默认: %(default)sm）')
    parser.add_argument('--simplify_ratio', '-s', type=float, default=0.4,
                        help='几何简化比例(0-1)（默认: %(default)s）')
    parser.add_argument('--texture_size', '-t', type=int, default=1024,
                        help='纹理最大尺寸（默认: %(default)spx）')
    parser.add_argument('--lod_scale', '-l', type=float, default=1.8,
                        help='LOD过渡缩放因子（默认: %(default)s）')

    args = parser.parse_args()

    print("开始优化I3S SLPK文件...")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"参数: min_points={args.min_points}, merge_distance={args.merge_distance}m, "
          f"simplify_ratio={args.simplify_ratio}, texture_size={args.texture_size}px, "
          f"lod_scale={args.lod_scale}")

    optimize_slpk_nodes(
        args.input,
        args.output,
        min_points=args.min_points,
        merge_distance=args.merge_distance,
        simplify_ratio=args.simplify_ratio,
        texture_max_size=args.texture_size,
        lod_scale=args.lod_scale
    )

    print("优化完成!")
