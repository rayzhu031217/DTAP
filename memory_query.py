import json
from pymilvus import Collection, connections
from utils import embed_text, call_llm, parse_function_call
import config

class MemoryQuery:
    def __init__(self):
        connections.connect(alias='default', host=config.MILVUS_HOST, port=config.MILVUS_PORT)
        self.collection = Collection(config.COLLECTION_NAME)
        self.collection.load()
        self.context_memories = []
        self.iteration = 0

    def query(self, question):
        self.iteration = 0
        self.context_memories = []
        system_prompt = """You are a robot assistant with access to a memory database of your past experiences.
You can retrieve memories by calling functions. The available functions are:
- retrieve_text(query: str) -> list: retrieve memories by semantic similarity to the query.
- retrieve_position(x: float, y: float, radius: float) -> list: retrieve memories near a position.
- retrieve_time_range(start: float, end: float) -> list: retrieve memories within a time range.

Each memory contains: caption (text), position (x,y), timestamp (seconds), and a value score (0-1) indicating importance.
When you have enough information, answer the question. If unsure, you can retrieve more memories (up to 3 iterations).

Output your answer as a JSON with keys: "answer", "reasoning" (optional), "used_memories" (list of memory IDs).
"""
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}]

        while self.iteration < config.MAX_ITERATIONS:
            response = call_llm(messages, model=config.LLM_MODEL, temperature=config.LLM_TEMPERATURE)
            function_call = parse_function_call(response)
            if function_call:
                memories = self._execute_function(function_call)
                self.context_memories.extend(memories)
                mem_text = self._format_memories(memories)
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Retrieved memories:\n{mem_text}\nContinue."})
                self.iteration += 1
            else:
                try:
                    answer_json = json.loads(response)
                    return answer_json
                except:
                    return {"answer": response.strip()}

        return {"answer": "Unable to answer within max iterations."}

    def _execute_function(self, func_call):
        name = func_call['name']
        args = func_call['arguments']
        if name == 'retrieve_text':
            query = args['query']
            emb = embed_text(query, model=config.EMBED_MODEL)
            results = self.collection.search(
                data=[emb.tolist()],
                anns_field='vector',
                param={'metric_type': 'IP', 'params': {'nprobe': 10}},
                limit=config.RETRIEVAL_K,
                output_fields=['caption', 'position_x', 'position_y', 'timestamp', 'value']
            )
        elif name == 'retrieve_position':
            x, y, radius = args['x'], args['y'], args['radius']
            # 优化：使用 query 进行标量过滤，然后按价值排序
            expr = f"position_x >= {x-radius} and position_x <= {x+radius} and position_y >= {y-radius} and position_y <= {y+radius}"
            results = self.collection.query(
                expr=expr,
                output_fields=["id", "caption", "position_x", "position_y", "timestamp", "value"],
                limit=config.RETRIEVAL_K * 2
            )
            # 按价值降序
            results.sort(key=lambda x: x['value'], reverse=True)
            memories = []
            for hit in results[:config.FINAL_K]:
                memories.append({
                    'id': hit['id'],
                    'caption': hit['caption'],
                    'position': (hit['position_x'], hit['position_y']),
                    'timestamp': hit['timestamp'],
                    'value': hit['value'],
                    'similarity': 1.0  # 位置检索无相似度，设为默认
                })
            return memories
        elif name == 'retrieve_time_range':
            start, end = args['start'], args['end']
            expr = f"timestamp >= {start} and timestamp <= {end}"
            results = self.collection.query(
                expr=expr,
                output_fields=["id", "caption", "position_x", "position_y", "timestamp", "value"],
                limit=config.RETRIEVAL_K * 2
            )
            results.sort(key=lambda x: x['value'], reverse=True)
            memories = []
            for hit in results[:config.FINAL_K]:
                memories.append({
                    'id': hit['id'],
                    'caption': hit['caption'],
                    'position': (hit['position_x'], hit['position_y']),
                    'timestamp': hit['timestamp'],
                    'value': hit['value'],
                    'similarity': 1.0
                })
            return memories
        else:
            return []

        # 对于文本检索，需要将 Milvus 结果转换为记忆列表
        memories = []
        for hits in results:
            for hit in hits:
                mem = {
                    'id': hit.id,
                    'caption': hit.entity.get('caption'),
                    'position': (hit.entity.get('position_x'), hit.entity.get('position_y')),
                    'timestamp': hit.entity.get('timestamp'),
                    'value': hit.entity.get('value'),
                    'similarity': hit.score
                }
                memories.append(mem)

        # 重排序：结合相似度和价值
        for mem in memories:
            mem['combined_score'] = 0.7 * mem['similarity'] + 0.3 * mem['value']
        memories.sort(key=lambda x: x['combined_score'], reverse=True)

        return memories[:config.FINAL_K]

    def _format_memories(self, memories):
        lines = []
        for m in memories:
            lines.append(f"- ID {m['id']} [value={m['value']:.2f}] at ({m['position'][0]:.1f},{m['position'][1]:.1f}) time={m['timestamp']:.1f}s: {m['caption']}")
        return "\n".join(lines)