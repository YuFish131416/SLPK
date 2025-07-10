import gzip
import json
import os
import shutil

from Tools.DepthMapTools import DepthMap


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

    def __init__(self, temp_dir, max_nodes):
        super().__init__(temp_dir)
        self.parent_map = {}
        self.max_nodes = max_nodes  # 最大保留节点数

    def optimize(self):
        """执行优化流程"""
        # 加载场景层数据
        scene_layer_data = self.load_scene_layer()
        store_data = scene_layer_data.get("store", {})
        root_node_ref = store_data.get("rootNode")
        all_node_ids = self.get_node_ids()

        total_nodes = len(all_node_ids)
        min_nodes = max(1, round(0.02 * total_nodes))  # 至少保留1个节点
        max_nodes = round(0.2 * total_nodes)

        # 确保max_nodes在[min_nodes, max_nodes]区间内
        if self.max_nodes < min_nodes:
            self.max_nodes = min_nodes
        elif self.max_nodes > max_nodes:
            self.max_nodes = max_nodes

        # 构建节点索引映射
        # self.build_node_index_mapping()

        # 提取节点ID
        root_node_id = self.extract_node_id(root_node_ref) if root_node_ref else None

        if not root_node_id:
            print("未找到根节点")
            return

        # 构建节点深度映射
        dep_map = DepthMap(self.nodes_dir, all_node_ids)

        # 确定最大保留深度
        max_retained_depth = dep_map.determine_max_retained_depth(self.max_nodes)
        print(f"最大保留深度: {max_retained_depth}")

        # 获取需要保留的节点（包括共享资源中引用的节点）
        nodes_to_keep = self.get_nodes_to_keep(max_retained_depth, dep_map.get_depth_items())
        nodes_to_delete = list(set(all_node_ids) - set(nodes_to_keep))
        print(f"保留的节点数: {len(nodes_to_keep)} 需要删除的节点数: {len(nodes_to_delete)}")

        # 按深度降序排序（深度大的先删除）
        nodes_to_delete.sort(key=lambda noid: dep_map.get_depth_data(noid), reverse=True)

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

    @staticmethod
    def get_nodes_to_keep(max_retained_depth, depth_items):
        """获取需要保留的节点ID集合（包括直接保留的节点和共享资源中引用的节点）"""
        # 根据深度保留节点
        nodes_to_keep = set()
        # texture_nodes_tool = TextureNodes(self.nodes_dir)
        for node_id, depth in depth_items:
            if depth <= max_retained_depth:
                nodes_to_keep.add(node_id)

                # 查找并添加共享资源中引用的节点
                # texture_nodes = texture_nodes_tool.find_texture_nodes(node_id)
                # nodes_to_keep.update(texture_nodes)

        return nodes_to_keep

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

                    # lod_selection = [{
                    #     "metricType": "maxScreenThreshold",
                    #     "maxError": 100000000
                    # }]

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
