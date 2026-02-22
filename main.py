import argparse
import json
from memory_builder import MemoryBuilder
from memory_query import MemoryQuery

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['build', 'query'], required=True)
    parser.add_argument('--rgb_txt', help='Path to rgb.txt for building')
    parser.add_argument('--depth_txt', help='Path to depth.txt for building')
    parser.add_argument('--groundtruth', help='Path to groundtruth.txt for building (optional)')
    parser.add_argument('--question', help='Question for query mode')
    args = parser.parse_args()

    if args.mode == 'build':
        if not args.rgb_txt or not args.depth_txt:
            print("Please provide --rgb_txt and --depth_txt")
            return
        builder = MemoryBuilder()
        builder.build(args.rgb_txt, args.depth_txt, args.groundtruth)
    elif args.mode == 'query':
        if not args.question:
            print("Please provide a question with --question")
            return
        querier = MemoryQuery()
        answer = querier.query(args.question)
        print(json.dumps(answer, indent=2))

if __name__ == '__main__':
    main()