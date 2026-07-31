import os

def parse_road_id(road_id):
    """
    从road_id解析拓扑信息。
    假设格式为 road_src_dst_lane (例如 road_4_0_1)
    返回: (source_intersection, destination_intersection)
    """
    try:
        parts = road_id.split('_')
        # road_4_0_1 -> src=4, dst=0
        # 兼容不同长度的ID，通常第二个和第三个部分是路口ID
        if len(parts) >= 3:
            return parts[1], parts[2]
        return None, None
    except Exception as e:
        print(f"Error parsing road_id {road_id}: {e}")
        return None, None

def ensure_dir(directory):
    """确保目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)