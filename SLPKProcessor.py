import collections
import glob
import gzip
import json
import os
import shutil
from collections import defaultdict


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


class SLPKOptimizer(SLPKProcessor):
    """SLPK文件优化器 - 基于LOD层级保留策略"""

    def __init__(self, temp_dir, max_nodes=4096):
        super().__init__(temp_dir)
        self.parent_map = {}
        self.max_nodes = max_nodes  # 最大保留节点数
        self.depth_map = {}  # 节点深度映射
        self.depth_count = defaultdict(int)  # 各深度节点计数
        self.nodepages_dir = os.path.join(temp_dir, "nodepages")  # 节点页目录
        self.root_dir = os.path.join(self.nodes_dir, 'root')
        self.nodepage_files = self._find_nodepage_files()  # 所有节点页文件
        self.node_id_to_index = {}  # 节点ID到索引的映射
        self.index_to_node_id = {}  # 索引到节点ID的映射

    def _find_nodepage_files(self):
        """查找所有节点页文件"""
        if not os.path.exists(self.nodepages_dir):
            return []

        # 查找所有节点页JSON文件（包括压缩格式）
        json_files = glob.glob(os.path.join(self.nodepages_dir, "**", "*.json"), recursive=True)
        gz_files = glob.glob(os.path.join(self.nodepages_dir, "**", "*.json.gz"), recursive=True)
        return json_files + gz_files

    def build_node_index_mapping(self):
        """构建节点ID与节点页索引之间的映射关系"""
        # 清空现有映射
        self.node_id_to_index = {}
        self.index_to_node_id = {}

        # 遍历所有节点页文件
        for nodepage_file in self.nodepage_files:
            try:
                nodepage_data = self.get_json_data(nodepage_file)

                # 处理每个节点条目
                for node_entry in nodepage_data.get("nodes", []):
                    node_index = node_entry.get("index")
                    if node_index is None:
                        continue

                    # 尝试从资源引用中获取节点ID
                    node_id = None

                    # 1. 从mesh资源中获取节点ID
                    if "mesh" in node_entry:
                        mesh = node_entry["mesh"]

                        # 从geometry资源获取
                        if "geometry" in mesh and "resource" in mesh["geometry"]:
                            resource_id = mesh["geometry"]["resource"]
                            if isinstance(resource_id, int):
                                # 资源ID对应节点文件夹的ID
                                node_id = str(resource_id)

                        # 从material资源获取（如果geometry未提供）
                        if not node_id and "material" in mesh and "resource" in mesh["material"]:
                            resource_id = mesh["material"]["resource"]
                            if isinstance(resource_id, int):
                                node_id = str(resource_id)

                    # 2. 如果mesh资源未提供节点ID，尝试从父节点引用获取
                    if not node_id and "parentIndex" in node_entry:
                        parent_index = node_entry["parentIndex"]
                        if parent_index in self.index_to_node_id:
                            # 获取父节点数据
                            parent_id = self.index_to_node_id[parent_index]
                            parent_data = self.get_node_data(parent_id)

                            # 从父节点的children中查找匹配项
                            if parent_data:
                                children = parent_data.get("children", [])
                                for child_ref in children:
                                    child_id = self.extract_node_id(child_ref)
                                    if child_id and child_id in self.get_node_ids():
                                        # 检查此子节点是否在同一个节点页中
                                        if child_id not in self.node_id_to_index:
                                            # 尚未映射，假设当前节点为此子节点
                                            node_id = child_id
                                            break

                    # 3. 如果仍无法确定，使用节点索引作为节点ID（最后手段）
                    if not node_id:
                        continue

                    # 保存映射关系
                    self.node_id_to_index[node_id] = node_index
                    self.index_to_node_id[node_index] = node_id

            except Exception as e:
                print(f"处理节点页 {nodepage_file} 时出错: {str(e)}")

    def optimize(self):
        """执行优化流程 - 基于LOD层级保留策略"""
        # 加载场景层数据
        scene_layer_data = self.load_scene_layer()
        store_data = scene_layer_data.get("store", {})
        root_node_ref = store_data.get("rootNode")
        all_node_ids = self.get_node_ids()

        # 构建节点索引映射
        # self.build_node_index_mapping()

        # 提取节点ID（假设extract_node_id能处理路径）
        root_node_id = self.extract_node_id(root_node_ref) if root_node_ref else None

        if not root_node_id:
            print("未找到根节点")
            return

        # 构建节点深度映射
        self.build_depth_map(all_node_ids)

        # 确定最大保留深度
        max_retained_depth = self.determine_max_retained_depth()
        print(f"最大保留深度: {max_retained_depth}")

        # 获取需要保留的节点（包括共享资源中引用的节点）
        nodes_to_keep = self.get_nodes_to_keep(max_retained_depth+1)
        nodes_to_delete = list(set(all_node_ids) - set(nodes_to_keep))
        print(f"保留的节点数: {len(nodes_to_keep)} 需要删除的节点数: {len(nodes_to_delete)}")

        # 按深度降序排序（深度大的先删除）
        nodes_to_delete.sort(key=lambda node_id: self.depth_map[node_id], reverse=True)

        # 删除节点
        for node_id in nodes_to_delete:
            self.delete_node(node_id)

        # # 获取要删除节点的索引集合
        # indices_to_delete = set()
        # for node_id in nodes_to_delete:
        #     if node_id in self.node_id_to_index:
        #         indices_to_delete.add(self.node_id_to_index[node_id])
        #
        # # 处理所有节点页文件
        # for nodepage_file in self.nodepage_files:
        #     self.process_nodepage_file(nodepage_file, indices_to_delete)

        # for node_id in nodes_to_keep:
        #     self.adjust_node_lod(node_id)

        print("优化完成")

    def get_node_data(self, node_id):
        """加载节点数据（3dNodeIndexDocument.json）"""
        node_dir = os.path.join(self.nodes_dir, node_id)
        node_file = os.path.join(node_dir, "3dNodeIndexDocument.json")

        return self.get_json_data(node_file)

    @staticmethod
    def get_json_data(file):
        """加载json数据"""
        if not os.path.exists(file):
            # 检查压缩版本
            gz_path = file + ".gz"
            if os.path.exists(gz_path):
                file = gz_path
            else:
                return None

        if file.endswith('.gz'):
            with gzip.open(file, 'rb') as f_in:
                return json.load(f_in)
        else:
            # 普通JSON文件使用utf-8编码
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)

    def find_texture_nodes(self, node_id):
        """递归查找节点及其所有关联的纹理节点、祖先节点和共享资源节点"""
        # 存储所有相关节点的集合
        all_related_nodes = set()
        # 记录已访问节点防止循环引用
        visited_nodes = set()
        # 使用队列进行BFS遍历
        queue = collections.deque([str(node_id)])

        # 构建父节点映射关系（需在类初始化时完成）
        if not hasattr(self, 'parent_map'):
            self.build_parent_mapping()

        while queue:
            current_id = queue.popleft()

            # 跳过已访问节点
            if current_id in visited_nodes:
                continue
            visited_nodes.add(current_id)
            all_related_nodes.add(current_id)

            # 1. 添加当前节点的所有祖先节点
            parent_id = self.parent_map.get(current_id)
            while parent_id is not None and parent_id != -1:
                if parent_id not in visited_nodes:
                    all_related_nodes.add(parent_id)
                    visited_nodes.add(parent_id)
                parent_id = self.parent_map.get(parent_id)

            # 2. 查找当前节点的直接纹理节点
            direct_texture_nodes = self._get_direct_texture_nodes(current_id)
            for tex_node in direct_texture_nodes:
                if tex_node not in visited_nodes:
                    queue.append(tex_node)
                    all_related_nodes.add(tex_node)

        return all_related_nodes

    def _get_direct_texture_nodes(self, node_id):
        """辅助方法：获取节点直接引用的纹理节点"""
        texture_node_ids = set()
        shared_resource_path = os.path.join(
            self.nodes_dir, node_id, "shared", "sharedResource.json"
        )

        if not os.path.exists(shared_resource_path):
            return texture_node_ids

        try:
            with open(shared_resource_path, 'r') as f:
                shared_data = json.load(f)
        except Exception as e:
            print(f"Error loading shared resource: {e}")
            return texture_node_ids

        texture_defs = shared_data.get("textureDefinitions", {})
        for tex_def in texture_defs.values():
            images = tex_def.get("images", [])
            for image in images:
                image_id = image.get("id")
                if image_id is not None:
                    texture_node_ids.add(str(image_id))

        return texture_node_ids

    def build_parent_mapping(self):
        """构建全量父节点映射关系（在类初始化时调用）"""
        self.parent_map = {}
        index_path = os.path.join(self.root_dir, "3dNodeIndexDocument.json")

        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
        except Exception as e:
            print(f"Error loading node index: {e}")
            return

        # 构建ID->父ID的映射
        for node in index_data.get("nodes", []):
            node_id = str(node["index"])
            parent_idx = node.get("parentIndex", -1)
            self.parent_map[node_id] = str(parent_idx) if parent_idx != -1 else None

    def get_nodes_to_keep(self, max_retained_depth):
        """获取需要保留的节点ID集合（包括直接保留的节点和共享资源中引用的节点）"""
        # 根据深度保留节点
        nodes_to_keep = set()
        for node_id, depth in self.depth_map.items():
            if depth <= max_retained_depth:
                nodes_to_keep.add(node_id)

                # 查找并添加共享资源中引用的节点
                # texture_nodes = self.find_texture_nodes(node_id)
                # nodes_to_keep.update(texture_nodes)

        return nodes_to_keep

    def build_depth_map(self, node_ids):
        """构建节点深度映射 - 直接使用节点中的level字段"""
        self.depth_map = {}
        self.depth_count = defaultdict(int)

        for node_id in node_ids:
            # 获取节点数据
            node_data = self.get_node_data(node_id)
            if not node_data:
                self.depth_map[node_id] = 0
                self.depth_count[0] += 1
                continue

            # 直接从节点数据中获取level字段
            level = node_data.get("level")
            if level is None:
                # 如果level字段不存在，使用默认值0
                level = 0

            # 记录深度信息
            self.depth_map[node_id] = level
            self.depth_count[level] += 1

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
        node_data = self.get_node_data(node_id)

        if not node_data:
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

                    lod_selection = [{
                        "metricType": "maxScreenThreshold",
                        "maxError": 100000000
                    }]

                    parent_data["children"] = []
                    # parent_data["lodSelection"] = lod_selection

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

    def process_nodepage_file(self, nodepage_file, indices_to_delete):
        """处理单个节点页文件"""
        try:
            # 读取节点页数据
            if nodepage_file.endswith('.gz'):
                with gzip.open(nodepage_file, 'rb') as f_in:
                    nodepage_data = json.load(f_in)
                is_gz = True
            else:
                with open(nodepage_file, 'r') as f:
                    nodepage_data = json.load(f)
                is_gz = False

            modified = False
            nodes = nodepage_data.get("nodes", [])
            new_nodes = []

            # 处理每个节点
            for node_entry in nodes:
                node_index = node_entry.get("index")

                # 1. 检查节点是否在删除列表中
                if node_index in indices_to_delete:
                    # 跳过删除节点
                    modified = True
                    continue

                # 2. 检查子节点是否包含要删除的节点
                children = node_entry.get("children", [])
                if children:
                    # 找出所有要删除的子节点索引
                    children_to_remove = [idx for idx in children if idx in indices_to_delete]

                    if children_to_remove:
                        # 移除所有子节点引用
                        node_entry["children"] = []

                        # 修改lodThreshold
                        node_entry["lodThreshold"] = 100000000

                        # 标记为已修改
                        modified = True

                # 添加到新节点列表
                new_nodes.append(node_entry)

            # 如果有修改，保存回文件
            if modified:
                nodepage_data["nodes"] = new_nodes

                if is_gz:
                    # 先写临时文件，再压缩
                    temp_file = nodepage_file.replace(".gz", ".tmp")
                    with open(temp_file, 'w') as f_out:
                        json.dump(nodepage_data, f_out, indent=2)
                    with open(temp_file, 'rb') as f_in:
                        with gzip.open(nodepage_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(temp_file)
                else:
                    with open(nodepage_file, 'w') as f:
                        json.dump(nodepage_data, f, indent=2)

        except Exception as e:
            print(f"处理节点页 {nodepage_file} 时出错: {str(e)}")

    def adjust_node_lod(self, node_id):
        """调整节点的LOD范围，使其覆盖所有缩放级别"""
        # 获取节点数据
        node_dir = os.path.join(self.nodes_dir, node_id)
        node_path = os.path.join(node_dir, "3dNodeIndexDocument.json")
        node_data = self.get_node_data(node_id)

        if not node_data:
            return

        children_refs = node_data.get("children", [])

        if children_refs:
            return

        lod_selection = [
            {
                "metricType": "maxScreenThresholdSQ",
                "maxError": 100000000
            },
            {
                "metricType": "maxScreenThreshold",
                "maxError": 100000
            }
        ],

        node_data["lodSelection"] = lod_selection

        # 移除子节点和共享资源引用
        # node_data.pop("children", None)
        # node_data.pop("sharedResource", None)

        # 保存修改
        try:
            with open(node_path, 'w') as f:
                json.dump(node_data, f, indent=2)
        except Exception as e:
            print(f"保存节点 {node_id} 修改失败: {e}")
