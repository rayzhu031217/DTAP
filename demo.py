import os
import json
import config
from memory_builder import MemoryBuilder
from memory_query import MemoryQuery

def main():
    print("="*50)
    print("DTAP 价值筛选演示（RGB-D 版）")  # 修改为 DTAP
    print("="*50)

    # 1. 构建记忆
    print("\n[1] 开始构建记忆...")
    builder = MemoryBuilder()
    rgb_txt = "data/rgb.txt"
    depth_txt = "data/depth.txt"
    groundtruth = "data/groundtruth.txt"
    if not os.path.exists(rgb_txt):
        print(f"错误：找不到 {rgb_txt}")
        print("请确保 data/ 目录下包含 TUM 格式的 rgb.txt, depth.txt, groundtruth.txt 以及对应的图像文件夹 rgb/ 和 depth/")
        return
    builder.build(rgb_txt, depth_txt, groundtruth)

    # 2. 打印存储的记忆
    print("\n[2] 已存储的高价值记忆：")
    from pymilvus import Collection, connections
    connections.connect(alias='default', host='localhost', port='19530')
    collection = Collection(config.COLLECTION_NAME)
    collection.load()
    results = collection.query(expr="id >= 0", limit=5, output_fields=["id", "caption", "value"])
    for r in results:
        print(f"  ID {r['id']} [价值={r['value']:.2f}] {r['caption'][:60]}...")

    # 3. 查询演示
    print("\n[3] 开始查询演示...")
    querier = MemoryQuery()
    questions = [
        "Where did I see a chair?",
        "What happened near the entrance?",
        "How long ago did someone drop a box?",
        "Was there any unusual event in the last 10 minutes?"
    ]
    if os.path.exists("data/questions.txt"):
        with open("data/questions.txt", "r") as f:
            questions = [line.strip() for line in f if line.strip()]

    for i, q in enumerate(questions, 1):
        print(f"\n  问题 {i}: {q}")
        answer = querier.query(q)
        print(f"  答案: {json.dumps(answer, indent=2, ensure_ascii=False)}")

    print("\n演示完成！更多信息请查看 README.md")

if __name__ == "__main__":
    main()