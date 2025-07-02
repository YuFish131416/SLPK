import argparse
import os.path
import shutil
import subprocess

from SLPKProcessor import SLPKOptimizer
from Tools.SLPKUtilTool import SLPKUtils


def optimize_slpk_nodes(slpk_path, output_path,
                        max_nodes=4096):  # 添加道格拉斯-普克容差参数
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
    dp_tolerance: 道格拉斯-普克算法容差
    """
    # 创建临时工作目录
    temp_dir = os.path.abspath("Temp")
    print(f"解压SLPK文件到临时目录: {temp_dir}")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    # 解压SLPK文件
    if not SLPKUtils.extract_slpk(slpk_path, temp_dir):
        return

    # 创建优化器并执行优化
    optimizer = SLPKOptimizer(
        temp_dir,
        max_nodes=max_nodes
    )
    optimizer.optimize()

    # 重新打包为SLPK
    print("重新打包为SLPK...")
    if SLPKUtils.repack_slpk(temp_dir, output_path):
        shutil.rmtree(temp_dir)
        print("重新打包完成！")
    else:
        print("重新打包失败!")

    i3s_converter_path = r"i3s_converter.exe"
    command = [i3s_converter_path, output_path]
    try:
        subprocess.run(
            command,
            check=True,  # 检查返回状态码
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.PIPE,  # 捕获错误输出
            text=False  # 以文本形式返回结果
        )
        print("LOD细节层次转换成功！")
    except subprocess.CalledProcessError as e:
        print(f"LOD细节层次转换成功 (状态码 {e.returncode}):")
        print("错误信息:", e.stderr)
    except Exception as e:
        print("发生意外错误:", str(e))


# ====================== 主程序入口 ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='I3S SLPK文件优化工具')

    # 位置参数（可选，带默认值）
    parser.add_argument('input', type=str, nargs='?', default="TestData1.slpk",
                        help='输入SLPK文件路径（默认: %(default)s）')
    parser.add_argument('output', type=str, nargs='?', default="output.slpk",
                        help='输出SLPK文件路径（默认: %(default)s）')

    # 可选参数
    parser.add_argument('--max_nodes', '-m', type=int, default=4096,
                        help='最大节点数阈值（默认: %(default)s）')

    args = parser.parse_args()

    print("开始优化I3S SLPK文件...")
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"参数: max_nodes={args.max_nodes}")

    optimize_slpk_nodes(
        args.input,
        args.output,
        max_nodes=args.max_nodes  # 添加容差参数
    )
