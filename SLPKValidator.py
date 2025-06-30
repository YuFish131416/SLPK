import os
import shutil
import stat
import time
import ctypes
import sys


def is_admin():
    """检查是否以管理员权限运行"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False


def force_delete(path):
    """尝试强制删除文件或文件夹"""
    try:
        # 首先尝试正常删除
        if os.path.isfile(path):
            os.remove(path)
            return True
        elif os.path.isdir(path):
            shutil.rmtree(path)
            return True
    except Exception as e:
        # 如果正常删除失败，尝试修改权限
        try:
            if os.path.isfile(path):
                # 移除只读属性
                os.chmod(path, stat.S_IWRITE)
                os.remove(path)
                return True
            elif os.path.isdir(path):
                # 递归修改文件夹内所有文件的权限
                for root, dirs, files in os.walk(path):
                    for name in files:
                        file_path = os.path.join(root, name)
                        os.chmod(file_path, stat.S_IWRITE)
                    for name in dirs:
                        dir_path = os.path.join(root, name)
                        os.chmod(dir_path, stat.S_IWRITE)
                # 然后删除整个文件夹
                shutil.rmtree(path)
                return True
        except:
            return False
    return False


def delete_temp_contents():
    """删除Temp目录下的所有内容"""
    temp_dir = r"C:\Users\23531\AppData\Local\Temp"

    if not os.path.exists(temp_dir):
        print(f"目录不存在: {temp_dir}")
        return

    print(f"开始清理目录: {temp_dir}")
    total_count = 0
    success_count = 0
    failed_count = 0
    skipped_count = 0

    # 先处理文件
    for filename in os.listdir(temp_dir):
        item_path = os.path.join(temp_dir, filename)
        total_count += 1

        try:
            # 跳过系统重要文件
            if filename.lower() in ['desktop.ini', 'thumbs.db']:
                skipped_count += 1
                continue

            # 尝试删除
            if force_delete(item_path):
                success_count += 1
                print(f"✓ 已删除: {filename}")
            else:
                # 最后尝试使用Windows API强制删除
                try:
                    if os.path.isfile(item_path):
                        ctypes.windll.kernel32.SetFileAttributesW(item_path, 0)
                        os.remove(item_path)
                        success_count += 1
                        print(f"✓ (强制)已删除: {filename}")
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                        success_count += 1
                        print(f"✓ (强制)已删除文件夹: {filename}")
                    else:
                        failed_count += 1
                        print(f"✗ 删除失败: {filename} (未知类型)")
                except:
                    failed_count += 1
                    print(f"✗ 无法删除: {filename} (可能正在使用)")
        except Exception as e:
            failed_count += 1
            print(f"✗ 删除错误: {filename} - {str(e)}")

    # 总结报告
    print("\n清理完成!")
    print(f"总计处理: {total_count} 项")
    print(f"成功删除: {success_count} 项")
    print(f"跳过项目: {skipped_count} 项")
    print(f"失败项目: {failed_count} 项")

    if failed_count > 0:
        print("\n提示: 某些文件可能被其他程序占用，请关闭所有程序后重试")
        print("      或重启电脑后再次运行此程序")


if __name__ == "__main__":
    delete_temp_contents()
