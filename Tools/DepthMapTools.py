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
        """确定最大保留深度（保留节点总数最接近阈值，可以超过）并打印各深度节点数"""
        cumulative_count = 0
        max_retained_depth = 0
        best_count = 0  # 最接近阈值的节点数
        best_diff = float('inf')  # 当前最小差值
        depths = sorted(self.depth_count.keys())

        # 打印表头
        print("深度\t节点数\t累积节点数\t与阈值差值")

        # 记录所有深度的信息用于后续打印
        depth_info = []

        # 遍历每个深度
        for depth in depths:
            count = self.depth_count[depth]
            new_cumulative = cumulative_count + count

            # 计算当前累积与阈值的差值
            diff = abs(new_cumulative - max_nodes)

            # 记录当前深度的信息
            depth_info.append((depth, count, new_cumulative, diff))

            # 检查是否更接近阈值
            if diff < best_diff:
                best_diff = diff
                best_count = new_cumulative
                max_retained_depth = depth

            # 如果当前累积已超过阈值且差值开始增大，提前终止
            if new_cumulative > max_nodes and diff > best_diff:
                break

            cumulative_count = new_cumulative

        # 打印所有深度的信息
        for d, count, cumul, diff in depth_info:
            status = ""
            if cumul == best_count:
                status = " <-- 最接近"
            print(f"{d}\t{count}\t{cumul}\t{abs(cumul - max_nodes)}{status}")

        # 打印最终结果
        print(f"\n最终保留深度: {max_retained_depth}")
        print(f"保留节点总数: {best_count} (阈值: {max_nodes}, 差值: {best_count - max_nodes})")
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


