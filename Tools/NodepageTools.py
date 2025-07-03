import glob
import gzip
import json
import os
import shutil


class Nodepages:
    def __init__(self, temp_dir):
        self.nodes_dir = os.path.join(temp_dir, "nodes")
        self.nodepages_dir = os.path.join(temp_dir, "nodepages")  # 节点页目录
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

    @staticmethod
    def process_nodepage_file(nodepage_file, indices_to_delete):
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
