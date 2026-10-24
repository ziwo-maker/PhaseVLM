import os
import json
from data_loader import load_json_data
from intersection_analyzer import IntersectionAnalyzer
from utils import ensure_dir

def main():
    # 1. 定义文件路径 (请根据实际位置修改)
    data_dir = "./" # 假设json文件在当前目录或上级目录
    flow_file = "/home/code/LLMTSCS/data/Hangzhou/4_4/anon_4_4_hangzhou_real.json"
    roadnet_file =  "/home/code/LLMTSCS/data/Hangzhou/4_4/anon_4_4_hangzhou_real.json"
    
    output_dir = "./output"
    output_file = os.path.join(output_dir, "intersection_evolution.json")

    # 2. 读取数据
    print("Loading data...")
    try:
        # 为了演示，如果文件不存在，这里不会报错而是提示
        # 实际运行时请确保json文件存在
        if not os.path.exists(flow_file):
            print(f"Warning: {flow_file} not found. Please place the dataset files.")
            return

        vehicles = load_json_data(flow_file)
        roadnet = load_json_data(roadnet_file)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 3. 执行分析
    print("Processing traffic flow...")
    analyzer = IntersectionAnalyzer(vehicles, roadnet)
    result_data = analyzer.analyze()

    # 4. 输出结果
    ensure_dir(output_dir)
    print(f"Saving results to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # indent=2 方便人类阅读，如果文件过大建议去掉 indent
        json.dump(result_data, f, indent=2) 

    print("Analysis complete.")

if __name__ == "__main__":
    main()