import os
import json
import gzip
import struct
from typing import Dict, List, Tuple, Optional, Any


class I3SReader:
    """I3S 文件读取基类"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None

    def load(self):
        """加载文件内容"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        if self.file_path.endswith('.gz'):
            with gzip.open(self.file_path, 'rb') as f:
                content = f.read()
                # 尝试解析为JSON
                try:
                    self.data = json.loads(content)
                    return
                except json.JSONDecodeError:
                    # 如果不是JSON，则作为二进制处理
                    self.data = content
        else:
            # 尝试作为JSON文件读取
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                    return
            except UnicodeDecodeError:
                # 如果是二进制文件
                with open(self.file_path, 'rb') as f:
                    self.data = f.read()


class SceneLayerReader(I3SReader):
    """3dSceneLayer.json 文件读取器"""

    @property
    def spatial_reference(self) -> Dict:
        """获取空间参考信息"""
        return self.data.get('spatialReference', {})

    @property
    def full_extent(self) -> Dict:
        """获取完整空间范围"""
        return self.data.get('fullExtent', {})

    @property
    def store(self) -> Dict:
        """获取存储配置信息"""
        return self.data.get('store', {})

    @property
    def node_pages(self) -> Dict:
        """获取节点页配置"""
        return self.data.get('nodePages', {})

    @property
    def geometry_definitions(self) -> List[Dict]:
        """获取几何定义"""
        return self.data.get('geometryDefinitions', [])

    @property
    def material_definitions(self) -> List[Dict]:
        """获取材质定义"""
        return self.data.get('materialDefinitions', [])

    @property
    def texture_set_definitions(self) -> List[Dict]:
        """获取纹理集定义"""
        return self.data.get('textureSetDefinitions', [])

    @property
    def root_node_path(self) -> str:
        """获取根节点路径"""
        return self.store.get('rootNode', './nodes/root')

    def get_geometry_schema(self) -> Dict:
        """获取默认几何模式"""
        return self.store.get('defaultGeometrySchema', {})

    def get_vertex_attributes(self) -> Dict:
        """获取顶点属性定义"""
        schema = self.get_geometry_schema()
        return schema.get('vertexAttributes', {})

    def get_feature_attributes(self) -> Dict:
        """获取特征属性定义"""
        schema = self.get_geometry_schema()
        return schema.get('featureAttributes', {})


import json
from typing import Dict, List, Optional, Any


class NodeReference:
    """表示节点引用结构"""

    def __init__(self, data: Dict):
        self.data = data

    @property
    def id(self) -> str:
        """引用的节点ID"""
        return self.data.get('id', '')

    @property
    def href(self) -> str:
        """节点资源的相对路径"""
        return self.data.get('href', '')

    @property
    def mbs(self) -> List[float]:
        """最小包围球 (x, y, z, radius)"""
        return self.data.get('mbs', [])

    @property
    def obb(self) -> Dict:
        """定向边界框信息"""
        return self.data.get('obb', {})

    def get_center(self) -> Optional[Tuple[float, float, float]]:
        """获取节点中心点 (优先使用OBB中心)"""
        if 'center' in self.obb and len(self.obb['center']) >= 3:
            return tuple(self.obb['center'][:3])

        if len(self.mbs) >= 3:
            return tuple(self.mbs[:3])

        return None

    def __repr__(self) -> str:
        return f"<NodeReference(id={self.id}, href={self.href})>"


class Resource:
    """表示资源引用结构"""

    def __init__(self, data: Dict):
        self.data = data

    @property
    def href(self) -> str:
        """资源路径"""
        return self.data.get('href', '')

    @property
    def feature_range(self) -> List[int]:
        """特征范围 [start, end] (如果适用)"""
        return self.data.get('featureRange', [])

    @property
    def texture_set_definition_id(self) -> int:
        """纹理集定义ID (如果适用)"""
        return self.data.get('textureSetDefinitionId', -1)

    def __repr__(self) -> str:
        return f"<Resource(href={self.href})>"


class LodSelection:
    """表示LOD选择标准"""

    def __init__(self, data: Dict):
        self.data = data

    @property
    def metric_type(self) -> str:
        """度量类型 (screenSpaceRelative, maxScreenThreshold, etc.)"""
        return self.data.get('metricType', '')

    @property
    def max_error(self) -> float:
        """最大误差值"""
        return self.data.get('maxError', 0.0)

    def __repr__(self) -> str:
        return f"<LodSelection(type={self.metric_type}, max_error={self.max_error})>"


class OBB:
    """表示定向边界框"""

    def __init__(self, data: Dict):
        self.data = data

    @property
    def center(self) -> List[float]:
        """中心点坐标 [x, y, z]"""
        return self.data.get('center', [0.0, 0.0, 0.0])

    @property
    def half_size(self) -> List[float]:
        """半尺寸 [x, y, z]"""
        return self.data.get('halfSize', [0.0, 0.0, 0.0])

    @property
    def quaternion(self) -> List[float]:
        """表示方向的四元数 [x, y, z, w]"""
        return self.data.get('quaternion', [0.0, 0.0, 0.0, 1.0])

    def get_rotation_matrix(self) -> List[List[float]]:
        """将四元数转换为旋转矩阵 (3x3)"""
        # 实际实现需要四元数到矩阵的转换逻辑
        # 这里返回单位矩阵作为示例
        return [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]

    def __repr__(self) -> str:
        return f"<OBB(center={self.center}, size={self.half_size})>"


class NodeIndexReader:
    """3dNodeIndexDocument.json 文件读取器 (I3S 1.8 规范)"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_file()

    def _load_file(self) -> Dict:
        """加载文件内容"""
        try:
            if self.file_path.endswith('.gz'):
                import gzip
                with gzip.open(self.file_path, 'rt', encoding='utf-8') as f:
                    return json.load(f)
            else:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            raise ValueError(f"无法解析节点索引文件: {str(e)}")

    @property
    def id(self) -> str:
        """节点的唯一标识符 (必需)"""
        return self.data.get('id', '')

    @property
    def level(self) -> int:
        """节点在索引树中的层级 (必需)"""
        return self.data.get('level', 0)

    @property
    def version(self) -> str:
        """节点的版本 (store update session ID)"""
        return self.data.get('version', '')

    @property
    def mbs(self) -> List[float]:
        """最小包围球 [x, y, z, radius] (必需)"""
        return self.data.get('mbs', [])

    @property
    def obb(self) -> OBB:
        """定向边界框信息 (必需)"""
        return OBB(self.data.get('obb', {}))

    @property
    def created(self) -> str:
        """节点创建时间 (ISO 8601 格式)"""
        return self.data.get('created', '')

    @property
    def expires(self) -> str:
        """节点过期时间 (ISO 8601 格式)"""
        return self.data.get('expires', '')

    @property
    def transform(self) -> List[float]:
        """4x4 变换矩阵 (列优先顺序)"""
        return self.data.get('transform', [])

    @property
    def parent_node(self) -> Optional[NodeReference]:
        """父节点引用"""
        parent_data = self.data.get('parentNode')
        return NodeReference(parent_data) if parent_data else None

    @property
    def children(self) -> List[NodeReference]:
        """子节点引用列表"""
        return [NodeReference(child) for child in self.data.get('children', [])]

    @property
    def neighbors(self) -> List[NodeReference]:
        """相邻节点引用列表 (同一层级)"""
        return [NodeReference(neighbor) for neighbor in self.data.get('neighbors', [])]

    @property
    def shared_resource(self) -> Optional[Resource]:
        """共享资源引用"""
        res_data = self.data.get('sharedResource')
        return Resource(res_data) if res_data else None

    @property
    def feature_data(self) -> List[Resource]:
        """特征数据资源引用列表"""
        return [Resource(fd) for fd in self.data.get('featureData', [])]

    @property
    def geometry_data(self) -> List[Resource]:
        """几何数据资源引用列表"""
        return [Resource(gd) for gd in self.data.get('geometryData', [])]

    @property
    def texture_data(self) -> List[Resource]:
        """纹理数据资源引用列表"""
        return [Resource(td) for td in self.data.get('textureData', [])]

    @property
    def attribute_data(self) -> List[Resource]:
        """属性数据资源引用列表"""
        return [Resource(ad) for ad in self.data.get('attributeData', [])]

    @property
    def lod_selection(self) -> List[LodSelection]:
        """LOD选择标准列表 (必需)"""
        return [LodSelection(ls) for ls in self.data.get('lodSelection', [])]

    @property
    def features(self) -> List[Dict]:
        """特征信息列表 (已弃用)"""
        return self.data.get('features', [])

    def get_center(self) -> Optional[Tuple[float, float, float]]:
        """获取节点中心点 (优先使用OBB中心)"""
        if self.obb.center:
            return tuple(self.obb.center[:3])

        if len(self.mbs) >= 3:
            return tuple(self.mbs[:3])

        return None

    def get_geometry_resource_paths(self) -> List[str]:
        """获取几何资源路径列表"""
        return [res.href for res in self.geometry_data if res.href]

    def get_texture_resource_paths(self) -> List[str]:
        """获取纹理资源路径列表"""
        return [res.href for res in self.texture_data if res.href]

    def get_lod_metric_types(self) -> List[str]:
        """获取所有LOD度量类型"""
        return [ls.metric_type for ls in self.lod_selection]

    def get_lod_by_metric(self, metric_type: str) -> Optional[LodSelection]:
        """根据度量类型获取LOD选择标准"""
        for ls in self.lod_selection:
            if ls.metric_type == metric_type:
                return ls
        return None

    def is_leaf_node(self) -> bool:
        """检查是否为叶节点 (没有子节点)"""
        return len(self.children) == 0

    def validate(self) -> bool:
        """验证节点文档是否符合规范 (检查必需字段)"""
        # 必需字段: id, level, mbs, obb, lodSelection
        if not self.id:
            return False

        if self.level <= 0:
            return False

        if len(self.mbs) < 4:  # 需要4个值: x,y,z,radius
            return False

        if not self.obb.center or not self.obb.half_size or not self.obb.quaternion:
            return False

        if not self.lod_selection:
            return False

        return True

    def __repr__(self) -> str:
        return (f"<NodeIndexReader(id={self.id}, level={self.level}, "
                f"children={len(self.children)}, geometries={len(self.geometry_data)})>")


class SharedResourceReader(I3SReader):
    """sharedResource.json 文件读取器"""

    @property
    def geometry_definitions(self) -> List[Dict]:
        """获取几何定义"""
        return self.data.get('geometryDefinitions', [])

    @property
    def material_definitions(self) -> List[Dict]:
        """获取材质定义"""
        return self.data.get('materialDefinitions', [])

    @property
    def texture_definitions(self) -> List[Dict]:
        """获取纹理定义"""
        return self.data.get('textureDefinitions', [])

    def get_geometry_definition(self, index: int) -> Optional[Dict]:
        """获取特定索引的几何定义"""
        if 0 <= index < len(self.geometry_definitions):
            return self.geometry_definitions[index]
        return None

    def get_material_definition(self, index: int) -> Optional[Dict]:
        """获取特定索引的材质定义"""
        if 0 <= index < len(self.material_definitions):
            return self.material_definitions[index]
        return None


class FeatureDataReader(I3SReader):
    """特征数据读取器 (通常为 features/{index}.json)"""

    @property
    def attributes(self) -> Dict:
        """获取特征属性"""
        return self.data.get('attributes', {})

    @property
    def geometry(self) -> Dict:
        """获取几何引用"""
        return self.data.get('geometry', {})

    @property
    def material(self) -> Dict:
        """获取材质引用"""
        return self.data.get('material', {})

    def get_attribute(self, name: str) -> Any:
        """获取特定属性值"""
        return self.attributes.get(name)

    def get_geometry_resource(self) -> Optional[str]:
        """获取几何资源路径"""
        if 'resource' in self.geometry:
            return self.geometry['resource'].get('href')
        return None


class GeometryResourceReader(I3SReader):
    """几何资源读取器 (通常为 geometries/{index}.bin)"""

    def parse_header(self) -> Dict:
        """解析几何资源头部信息"""
        if not self.data or len(self.data) < 8:
            return {}

        # 头部: 4字节顶点数 + 4字节特征数
        vertex_count = struct.unpack('<I', self.data[0:4])[0]
        feature_count = struct.unpack('<I', self.data[4:8])[0]

        return {
            'vertex_count': vertex_count,
            'feature_count': feature_count,
            'header_size': 8
        }

    def get_vertex_data(self, offset: int, count: int, component: int, dtype: str) -> List:
        """获取顶点数据

        :param offset: 数据偏移量
        :param count: 顶点数量
        :param component: 每个顶点的组件数
        :param dtype: 数据类型 ('Float32', 'UInt8', etc.)
        :return: 顶点数据列表
        """
        if not self.data:
            return []

        # 计算数据大小
        dtype_size = self._get_dtype_size(dtype)
        data_size = count * component * dtype_size

        # 检查数据范围
        if offset + data_size > len(self.data):
            raise ValueError("请求的数据超出文件范围")

        # 解析数据
        data_format = self._get_struct_format(dtype, component)
        values = []
        for i in range(count):
            start = offset + i * component * dtype_size
            end = start + component * dtype_size
            chunk = self.data[start:end]
            values.append(struct.unpack(data_format, chunk))

        return values

    def _get_dtype_size(self, dtype: str) -> int:
        """获取数据类型大小"""
        sizes = {
            'Float32': 4,
            'UInt32': 4,
            'UInt16': 2,
            'UInt8': 1,
            'Int32': 4,
            'Int16': 2,
            'Int8': 1
        }
        return sizes.get(dtype, 4)  # 默认为4字节

    def _get_struct_format(self, dtype: str, component: int) -> str:
        """获取struct解包格式"""
        dtype_map = {
            'Float32': 'f',
            'UInt32': 'I',
            'UInt16': 'H',
            'UInt8': 'B',
            'Int32': 'i',
            'Int16': 'h',
            'Int8': 'b'
        }

        fmt_char = dtype_map.get(dtype, 'f')  # 默认为float
        return f'<{component}{fmt_char}'  # 小端字节序


class TextureReader:
    """纹理读取器 (支持多种图像格式)"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def read(self) -> bytes:
        """读取纹理数据"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"纹理文件不存在: {self.file_path}")

        with open(self.file_path, 'rb') as f:
            return f.read()

    def get_format(self) -> str:
        """获取纹理格式"""
        ext = os.path.splitext(self.file_path)[1].lower()
        return ext if ext else '.unknown'


class SLPKExplorer:
    """SLPK 文件包浏览器"""

    def __init__(self, slpk_path: str):
        """
        :param slpk_path: SLPK文件路径或解压后的目录路径
        """
        self.slpk_path = slpk_path
        self.scene_layer = None
        self.nodes = {}  # 节点ID到节点读取器的映射
        self.shared_resources = {}  # 共享资源路径到读取器的映射

        # 确定是压缩包还是解压目录
        if os.path.isfile(slpk_path) and slpk_path.endswith('.slpk'):
            self.is_archive = True
            # 在实际应用中，这里应该实现解压逻辑
            # 为简化示例，我们假设已解压到同名目录
            self.base_dir = os.path.splitext(slpk_path)[0]
            if not os.path.exists(self.base_dir):
                raise FileNotFoundError("SLPK文件需要解压，但解压目录不存在")
        else:
            self.is_archive = False
            self.base_dir = slpk_path

        # 加载场景层文件
        scene_layer_path = os.path.join(self.base_dir, '3dSceneLayer.json.gz')
        if not os.path.exists(scene_layer_path):
            scene_layer_path = os.path.join(self.base_dir, '3dSceneLayer.json')

        if os.path.exists(scene_layer_path):
            self.scene_layer = SceneLayerReader(scene_layer_path)
            self.scene_layer.load()
        else:
            raise FileNotFoundError("3dSceneLayer.json 文件未找到")

    def load_node(self, node_id: str):
        """加载指定节点"""
        node_dir = os.path.join(self.base_dir, 'nodes', node_id)
        index_path = os.path.join(node_dir, '3dNodeIndexDocument.json')

        # 检查压缩版本
        if not os.path.exists(index_path):
            gz_path = index_path + '.gz'
            if os.path.exists(gz_path):
                index_path = gz_path
            else:
                raise FileNotFoundError(f"节点索引文件未找到: {node_id}")

        reader = NodeIndexReader(index_path)
        reader.load()
        self.nodes[node_id] = reader
        return reader

    def load_shared_resource(self, resource_path: str):
        """加载共享资源"""
        full_path = os.path.join(self.base_dir, resource_path)
        if not os.path.exists(full_path):
            # 检查压缩版本
            gz_path = full_path + '.gz'
            if os.path.exists(gz_path):
                full_path = gz_path
            else:
                raise FileNotFoundError(f"共享资源未找到: {resource_path}")

        reader = SharedResourceReader(full_path)
        reader.load()
        self.shared_resources[resource_path] = reader
        return reader

    def get_node_geometry(self, node_id: str, geometry_index: int = 0) -> Optional[GeometryResourceReader]:
        """获取节点的几何资源"""
        if node_id not in self.nodes:
            self.load_node(node_id)

        node = self.nodes[node_id]
        geom_resources = node.get_geometry_resources()

        if geometry_index < len(geom_resources):
            geom_path = os.path.join(self.base_dir, 'nodes', node_id, geom_resources[geometry_index])
            reader = GeometryResourceReader(geom_path)
            reader.load()
            return reader

        return None

    def get_node_texture(self, node_id: str, texture_index: int = 0) -> Optional[TextureReader]:
        """获取节点的纹理资源"""
        if node_id not in self.nodes:
            self.load_node(node_id)

        node = self.nodes[node_id]
        texture_resources = node.get_texture_resources()

        if texture_index < len(texture_resources):
            texture_path = os.path.join(self.base_dir, 'nodes', node_id, texture_resources[texture_index])
            return TextureReader(texture_path)

        return None

    def get_root_node(self) -> Optional[NodeIndexReader]:
        """获取根节点"""
        root_path = self.scene_layer.root_node_path
        if root_path.startswith('./'):
            root_path = root_path[2:]

        # 提取节点ID (假设路径格式为 'nodes/{node_id}')
        parts = root_path.split('/')
        if len(parts) >= 2 and parts[0] == 'nodes':
            node_id = parts[1]
            return self.load_node(node_id)

        return None

    def traverse_nodes(self, start_node_id: str = None) -> List[str]:
        """遍历所有节点ID"""
        nodes_dir = os.path.join(self.base_dir, 'nodes')
        if not os.path.exists(nodes_dir):
            return []

        node_ids = []
        for entry in os.listdir(nodes_dir):
            entry_path = os.path.join(nodes_dir, entry)
            if os.path.isdir(entry_path):
                # 检查是否有节点索引文件
                index_path = os.path.join(entry_path, '3dNodeIndexDocument.json')
                gz_path = index_path + '.gz'
                if os.path.exists(index_path) or os.path.exists(gz_path):
                    node_ids.append(entry)

        return node_ids

    def get_node_center(self, node_id: str) -> Tuple[float, float, float]:
        """获取节点中心点坐标"""
        if node_id not in self.nodes:
            self.load_node(node_id)

        node = self.nodes[node_id]

        # 优先使用OBB中心
        obb = node.obb
        if obb and 'center' in obb and len(obb['center']) >= 3:
            return tuple(obb['center'][:3])

        # 其次使用MBS中心
        mbs = node.mbs
        if mbs and len(mbs) >= 3:
            return tuple(mbs[:3])

        # 默认返回原点
        return (0.0, 0.0, 0.0)

    def get_node_lod_info(self, node_id: str) -> Dict:
        """获取节点的LOD信息"""
        if node_id not in self.nodes:
            self.load_node(node_id)

        node = self.nodes[node_id]
        lod_info = {'level': node.level, 'lod_selection': node.lod_selection, 'geometry_count': len(node.geometry_data)}

        # 添加几何资源数量

        return lod_info