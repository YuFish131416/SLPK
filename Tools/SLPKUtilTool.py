import gzip
import os
import shutil
import zipfile


class SLPKUtils:
    """SLPK文件处理工具类（修复版）"""

    @staticmethod
    def extract_slpk(slpk_path, output_dir):
        """解压SLPK文件并处理压缩文件"""
        try:
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 解压ZIP文件
            with zipfile.ZipFile(slpk_path, 'r') as z:
                z.extractall(output_dir)

            # 解压所有.gz文件（保留原始文件）
            # SLPKUtils.decompress_gz_files(output_dir)

            return True
        except Exception as e:
            print(f"解压SLPK文件失败: {str(e)}")
            return False

    @staticmethod
    def decompress_gz_files(directory):
        """解压目录中的所有.gz文件"""
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.gz'):
                    gz_path = os.path.join(root, file)
                    json_path = gz_path[:-3]  # 移除.gz扩展名

                    # 跳过已经解压的文件
                    if os.path.exists(json_path):
                        continue

                    # 解压.gz文件
                    with gzip.open(gz_path, 'rb') as f_in:
                        with open(json_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    os.remove(gz_path)

    @staticmethod
    def should_compress(path, root_dir):
        """判断文件是否需要压缩"""
        # 根目录的3dSceneLayer.json压缩
        if os.path.basename(path) == "3dSceneLayer.json" and os.path.dirname(path) == root_dir:
            return True

        # 节点目录中的特定文件需要压缩
        if "nodes" in path:
            filename = os.path.basename(path)
            if filename in ["3dNodeIndexDocument.json", "features.json"]:
                return True

        # 其他JSON文件默认压缩
        if path.endswith('.json'):
            return True

        # Bin文件默认压缩
        if path.endswith('.bin'):
            return True

        return False

    @staticmethod
    def compress_for_repack(source_dir):
        """为重新打包准备文件（压缩需要压缩的文件）"""
        # 处理所有文件
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)

                # 忽略已压缩的文件
                if file.endswith('.gz'):
                    pass

                # 判断是否需要压缩
                if SLPKUtils.should_compress(file_path, source_dir):
                    gz_path = file_path + '.gz'

                    # 压缩文件
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(gz_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)

                    # 删除原始文件
                    os.remove(file_path)

    @staticmethod
    def repack_slpk(source_dir, output_path):
        """重新打包为SLPK文件（修复版）"""
        try:
            # 为重新打包准备文件
            SLPKUtils.compress_for_repack(source_dir)

            # 创建ZIP包 - 修改为仅存储模式 (ZIP_STORED)
            with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_STORED) as z:  # 关键修改
                for root, _, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)

                        # 计算在ZIP中的相对路径
                        arcname = os.path.relpath(file_path, source_dir)

                        # 添加到ZIP
                        z.write(file_path, arcname)

            return True
        except Exception as e:
            print(f"重新打包SLPK失败: {str(e)}")
            return False
