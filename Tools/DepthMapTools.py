import gzip
import json
import os
from collections import defaultdict


class DepthMap:
    """SLPK文件节点深度工具类"""

    def __init__(self, nodes_dir, node_ids):
        self.depth_map = {}
        self.depth_count = defaultdict(int)
        self.nodes_dir = nodes_dir
        self.node_ids = node_ids
        self._build_depth_map()

    def _build_depth_map(self):
        """构建节点深度映射 - 直接使用节点中的level字段"""
        for node_id in self.node_ids:
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

    def determine_max_retained_depth(self, max_nodes):
        """确定最大保留深度"""
        cumulative_count = 0
        max_retained_depth = 0

        # 按深度从小到大遍历
        depths = sorted(self.depth_count.keys())
        for depth in depths:
            count = self.depth_count[depth]

            # 检查加上当前深度节点是否会超过限制
            if cumulative_count + count > max_nodes:
                # 当前深度加上后会超过限制，保留到上一深度
                break

            cumulative_count += count
            max_retained_depth = depth

        return max_retained_depth

    def get_depth_data(self, node_id):
        return self.depth_map[node_id]

    def get_depth_items(self):
        return self.depth_map.items()

    def get_node_data(self, node_id):
        """加载节点数据（3dNodeIndexDocument.json）"""
        node_dir = str(os.path.join(self.nodes_dir, node_id))
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


