import asyncio
import json

class AgentBackend:
    """
    后端只负责干活，返回数据。
    """
    
    # 步骤 1: 意图分析
    async def intent_analyzer(self, text: str):
        await asyncio.sleep(1) # 模拟耗时
        # 返回结构化数据
        return {
            "intent": "generate_image",
            "keywords": ["赛博朋克", "孙悟空", "机甲"]
        }

    # 步骤 2: 关键词优化 (依赖步骤 1 的结果)
    async def keyword_generation(self, keywords: list):
        await asyncio.sleep(1)
        # 模拟生成更详细的 Prompt
        refined_prompt = f"Masterpiece, 8k, {' '.join(keywords)}, neon lights, metallic texture"
        return refined_prompt

    # 步骤 3: 图像生成 (依赖步骤 2 的 Prompt)
    async def image_generation(self, prompt: str):
        await asyncio.sleep(2)
        # 模拟生成，返回本地图片路径或 URL
        return "https://picsum.photos/800/600"