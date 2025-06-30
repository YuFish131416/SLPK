import gzip
import json
import os
import shutil
from collections import defaultdict, deque

import numpy as np


class SLPKProcessor:
    """SLPK文件处理的基类"""

    def __init__(self, temp_dir):
        self.temp_dir = temp_dir
        self.nodes_dir = os.path.join(temp_dir, "nodes")
        self.scene_layer_path = self._find_scene_layer()

    @staticmethod
    def extract_node_id(node_ref):
        """从节点引用中提取节点ID"""
        if isinstance(node_ref, dict):
            node_id = node_ref.get('id')
        elif isinstance(node_ref, str):
            if node_ref and "/" in node_ref:
                return node_ref.split("/")[-1]  # 取路径最后一段
            node_id = node_ref
        else:
            return None
        if node_id is None:
            return None
            # 确保返回字符串
        return str(node_id)


    def _find_scene_layer(self):
        """查找场景层文件路径"""
        paths = [
            os.path.join(self.temp_dir, "3dSceneLayer.json.gz"),
            os.path.join(self.temp_dir, "3dSceneLayer.json")
        ]
        for path in paths:
            if os.path.exists(path):
                return path
        raise FileNotFoundError("3dSceneLayer.json not found in SLPK")

    def load_scene_layer(self):
        """加载和解压场景层数据"""
        if self.scene_layer_path.endswith('.gz'):
            with gzip.open(self.scene_layer_path, 'rb') as f_in:
                scene_layer_data = json.load(f_in)
            # 解压后保存为普通json以便修改
            uncompressed_path = os.path.join(self.temp_dir, "3dSceneLayer.json")
            with open(uncompressed_path, 'w') as f_out:
                json.dump(scene_layer_data, f_out, indent=2)
            self.scene_layer_path = uncompressed_path
            return scene_layer_data
        else:
            with open(self.scene_layer_path, 'r') as f:
                return json.load(f)

    def save_scene_layer(self, scene_layer_data):
        """保存场景层数据并压缩"""
        # 保存为未压缩JSON
        uncompressed_path = os.path.join(self.temp_dir, "3dSceneLayer.json")
        with open(uncompressed_path, 'w') as f:
            json.dump(scene_layer_data, f, indent=2)

        # 压缩为GZ
        gz_path = uncompressed_path + ".gz"
        with open(uncompressed_path, 'rb') as f_in:
            with gzip.open(gz_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # 删除未压缩文件
        os.remove(uncompressed_path)
        self.scene_layer_path = gz_path

    def get_node_ids(self):
        """获取所有有效的节点ID"""
        node_ids = []
        if not os.path.exists(self.nodes_dir):
            return node_ids

        for d in os.listdir(self.nodes_dir):
            full_path = os.path.join(self.nodes_dir, d)
            if os.path.isdir(full_path):
                # 接受数字形式或root形式的节点ID
                if d.isdigit() or d == 'root':
                    node_ids.append(d)
        return node_ids

    def get_node_center(self, node_data):
        """获取节点的中心点坐标（三维）"""
        # 尝试获取OBB（定向边界框）
        if 'obb' in node_data:
            obb = node_data['obb']
            if 'center' in obb and len(obb['center']) >= 3:
                return np.array(obb['center'][:3])

        # 尝试获取MBS（最小包围球）
        if 'mbs' in node_data:
            mbs = node_data['mbs']
            if isinstance(mbs, list) and len(mbs) >= 4:
                return np.array(mbs[:3])

        # 尝试获取几何范围
        if 'geometry' in node_data:
            geom = node_data['geometry']
            if 'bbox' in geom and len(geom['bbox']) >= 6:
                # 计算边界框中心: [minx, miny, minz, maxx, maxy, maxz]
                minx, miny, minz, maxx, maxy, maxz = geom['bbox'][:6]
                return np.array([(minx + maxx) / 2, (miny + maxy) / 2, (minz + maxz) / 2])

        # 默认返回原点
        return np.array([0, 0, 0])

    def get_node_extent(self, node_data):
        """获取节点的空间范围（OBB或MBS）"""
        # 优先使用OBB
        if 'obb' in node_data:
            obb = node_data['obb']
            if 'halfSize' in obb and len(obb['halfSize']) >= 3:
                half_size = np.array(obb['halfSize'][:3])
                return 2 * np.linalg.norm(half_size)  # 近似直径

        # 其次使用MBS
        if 'mbs' in node_data:
            mbs = node_data['mbs']
            if isinstance(mbs, list) and len(mbs) >= 4:
                return mbs[3] * 2  # 直径

        # 默认返回一个小的范围
        return 1.0

    def estimate_vertex_count(self, node_data):
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


class SLPKOptimizer(SLPKProcessor):
    """SLPK文件优化器 - 基于LOD层级保留策略"""

    def __init__(self, temp_dir, max_nodes=4096):
        super().__init__(temp_dir)
        self.max_nodes = max_nodes  # 最大保留节点数
        self.depth_map = {}  # 节点深度映射
        self.depth_count = defaultdict(int)  # 各深度节点计数

    def optimize(self):
        """执行优化流程 - 基于LOD层级保留策略"""
        # 加载场景层数据
        scene_layer_data = self.load_scene_layer()
        store_data = scene_layer_data.get("store", {})
        root_node_ref = store_data.get("rootNode")
        all_node_ids = self.get_node_ids()

        # 提取节点ID（假设extract_node_id能处理路径）
        root_node_id = self.extract_node_id(root_node_ref) if root_node_ref else None

        if not root_node_id:
            print("未找到根节点")
            return

        # 构建节点深度映射
        self.build_depth_map(root_node_id, all_node_ids)

        # 确定最大保留深度
        max_retained_depth = self.determine_max_retained_depth()
        print(f"最大保留深度: {max_retained_depth}")

        # 删除深度大于max_retained_depth的节点

        nodes_to_keep = [node_id for node_id, depth in self.depth_map.items() if depth < max_retained_depth]
        nodes_to_delete = list(set(all_node_ids) - set(nodes_to_keep))
        print(f"保留的节点数: {len(nodes_to_keep)} 需要删除的节点数: {len(nodes_to_delete)}")

        # 按深度降序排序（深度大的先删除）
        nodes_to_delete.sort(key=lambda node_id: self.depth_map[node_id], reverse=True)

        # 删除节点
        for node_id in nodes_to_delete:
            self.delete_node(node_id)

        # 调整保留节点的LOD范围
        for node_id, depth in self.depth_map.items():
            if depth == max_retained_depth:
                self.adjust_node_lod(node_id)

        print("优化完成")

    def build_depth_map(self, root_node_id, node_ids):
        """构建节点深度映射（通过父节点链遍历）"""
        self.depth_map = {}
        self.depth_count = defaultdict(int)

        # 初始化根节点
        self.depth_map[root_node_id] = 0
        self.depth_count[0] += 1

        # 步骤1: 构建父节点映射表 (node_id -> parent_id)
        parent_map = {}
        for node_id in node_ids:
            node_path = os.path.join(self.nodes_dir, node_id, "3dNodeIndexDocument.json")

            # 处理可能存在的压缩文件
            if not os.path.exists(node_path):
                gz_path = node_path + '.gz'
                if os.path.exists(gz_path):
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(node_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                else:
                    print(f"节点 {node_id} 的索引文件不存在，跳过")
                    continue

            try:
                with open(node_path, 'r') as f:
                    node_data = json.load(f)
            except Exception as e:
                print(f"加载节点 {node_id} 索引文件失败: {e}")
                continue

            # 提取父节点ID
            parent_ref = node_data.get("parentNode")
            parent_id = self.extract_node_id(parent_ref)
            parent_map[node_id] = parent_id

        # 步骤2: 遍历所有节点，向上查找父节点链
        for node_id in node_ids:
            if node_id in self.depth_map:  # 已处理的节点跳过
                continue

            current_path = []  # 存储从当前节点到已计算深度节点的路径
            current = node_id
            visited = set()  # 检测循环引用

            # 向上遍历直到找到已计算深度的节点或根节点
            while current and current not in self.depth_map and current not in visited:
                visited.add(current)
                current_path.append(current)
                current = parent_map.get(current)

            # 处理找到有效祖先节点的情况
            if current and current in self.depth_map:
                base_depth = self.depth_map[current]
                current_path.reverse()  # 反转路径: 祖先->父->当前

                # 沿路径设置节点深度
                current_depth = base_depth + 1
                for nid in current_path:
                    if nid not in self.depth_map:  # 防止重复设置
                        self.depth_map[nid] = current_depth
                        self.depth_count[current_depth] += 1
                        current_depth += 1

            # 处理孤立节点（无法连接到根节点）
            else:
                print(f"警告: 节点 {node_id} 无法连接到根节点，深度设置为0")
                self.depth_map[node_id] = 0
                self.depth_count[0] += 1

    def determine_max_retained_depth(self):
        """确定最大保留深度"""
        cumulative_count = 0
        max_retained_depth = 0

        # 按深度从小到大遍历
        depths = sorted(self.depth_count.keys())
        for depth in depths:
            count = self.depth_count[depth]

            # 检查加上当前深度节点是否会超过限制
            if cumulative_count + count > self.max_nodes:
                # 当前深度加上后会超过限制，保留到上一深度
                break

            cumulative_count += count
            max_retained_depth = depth

        return max_retained_depth

    def delete_node(self, node_id):
        """删除节点及其所有资源"""
        # 获取父节点ID
        node_path = os.path.join(self.nodes_dir, node_id, "3dNodeIndexDocument.json")
        if not os.path.exists(node_path):
            return

        try:
            with open(node_path, 'r') as f:
                node_data = json.load(f)
        except:
            return

        parent_ref = node_data.get("parentNode")
        parent_id = self.extract_node_id(parent_ref)

        # 更新父节点：从父节点的children中移除该节点
        if parent_id:
            parent_path = os.path.join(self.nodes_dir, parent_id, "3dNodeIndexDocument.json")
            if os.path.exists(parent_path):
                try:
                    with open(parent_path, 'r') as f:
                        parent_data = json.load(f)

                    children_refs = parent_data.get("children", [])
                    new_children = [
                        ref for ref in children_refs
                        if self.extract_node_id(ref) != node_id
                    ]
                    parent_data["children"] = new_children

                    with open(parent_path, 'w') as f:
                        json.dump(parent_data, f, indent=2)
                except Exception as e:
                    print(f"更新父节点 {parent_id} 失败: {e}")

        # 删除节点目录（递归删除所有资源）
        node_dir = os.path.join(self.nodes_dir, node_id)
        if os.path.exists(node_dir):
            try:
                shutil.rmtree(node_dir)
            except Exception as e:
                print(f"删除节点目录 {node_dir} 失败: {e}")

    def adjust_node_lod(self, node_id):
        """调整节点的LOD范围，使其覆盖更高细节级别"""
        node_path = os.path.join(self.nodes_dir, node_id, "3dNodeIndexDocument.json")
        if not os.path.exists(node_path):
            return

        try:
            with open(node_path, 'r') as f:
                node_data = json.load(f)
        except:
            return

        lod_selection = node_data.get("lodSelection", [])
        if lod_selection:
            # 找到最后一个LOD范围
            last_lod = lod_selection[-1]

            # 修改最大误差为极大值，使该节点覆盖所有更高细节级别
            if "maxError" in last_lod:
                last_lod["maxError"] = 1e9

            # 保存修改
            with open(node_path, 'w') as f:
                json.dump(node_data, f, indent=2)
