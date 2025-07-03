import collections
import json
import os


class TextureNodes:

    def __init__(self, nodes_dir):
        self.parent_map = {}
        self.nodes_dir = nodes_dir
        self.root_dir = os.path.join(self.nodes_dir, 'root')

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
