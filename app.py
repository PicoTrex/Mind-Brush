import chainlit as cl
from backend import AgentBackend

# 初始化后端
backend = AgentBackend()

@cl.on_chat_start
async def start():
    await cl.Message(content="我是分步执行 Agent，请输入内容。").send()

@cl.on_message
async def main(message: cl.Message):
    """
    前端编排逻辑：串行调用后端函数
    """
    user_input = message.content
    
    # ==============================
    # 阶段 1: 调用意图分析
    # ==============================
    step1_result = None # 用于存储这一步的结果，传给下一步
    
    async with cl.Step(name="Step 1: 意图分析", type="llm") as step:
        step.input = user_input # 在 UI 上显示输入是什么
        
        # 1. 调用后端
        step1_result = await backend.intent_analyzer(user_input)
        
        # 2. 格式化输出到 UI
        # 比如把字典转成 JSON 字符串显示
        import json
        step.output = f"```json\n{json.dumps(step1_result, indent=2, ensure_ascii=False)}\n```"

    # ==============================
    # 阶段 2: 调用关键词生成
    # ==============================
    step2_result = None
    
    # 我们可以在这里加逻辑判断，比如意图不对就不执行了
    if step1_result["intent"] == "generate_image":
        
        async with cl.Step(name="Step 2: 提示词优化", type="tool") as step:
            #以此步的输入作为上一步的输出
            step.input = str(step1_result["keywords"]) 
            
            # 1. 调用后端 (传入上一步的数据)
            step2_result = await backend.keyword_generation(step1_result["keywords"])
            
            # 2. 更新 UI
            step.output = step2_result
            
    # ==============================
    # 阶段 3: 最终图像生成
    # ==============================
    final_image_url = None
    
    async with cl.Step(name="Step 3: 图像生成", type="run") as step:
        step.input = step2_result
        
        # 1. 调用后端
        final_image_url = await backend.image_generation(step2_result)
        
        # 2. 更新 UI (显示结果文本)
        step.output = "生成成功！"
        
        # 3. 将图片挂载到这个步骤里 (可选)
        step.elements = [cl.Image(url=final_image_url, name="preview", display="inline")]

    # ==============================
    # 发送最终的大图消息
    # ==============================
    await cl.Message(
        content="任务全部完成，这是最终结果：",
        elements=[cl.Image(url=final_image_url, name="result", display="inline")]
    ).send()