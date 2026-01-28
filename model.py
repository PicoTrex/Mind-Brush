import json
from openai import AsyncOpenAI
from typing import Optional


class Response:
    def __init__(self, content: str, tool_calls: list[dict], thinking_content: str = None):
        self.content = content
        self.thinking_content = thinking_content
        self.tool_calls = tool_calls

    def __str__(self):
        out = {
            'content': self.content,
            'thinking_content': self.thinking_content,
            'tool_calls': str(self.tool_calls)
        }
        return json.dumps(out, ensure_ascii=False, indent=4)

    def __repr__(self):
        return self.__str__()


class Function:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments

    def __str__(self):
        return json.dumps({'name': self.name, 'arguments': self.arguments}, ensure_ascii=False, indent=4)

    def __repr__(self):
        return self.__str__()


class Tool_Call:
    def __init__(self, name: str, arguments: dict, id: str = None):
        self.function = Function(name, json.dumps(arguments, ensure_ascii=False))
        self.id = id

    def __str__(self):
        return json.dumps({'function': str(self.function), 'id': self.id}, ensure_ascii=False, indent=4)

    def __repr__(self):
        return self.__str__()


class Model:
    def __init__(
        self,
        model_name: Optional[str] = 'gpt-4o',
        system_prompt: Optional[str] = 'You are a helpful assistant.',
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[str] = 'openai',
    ):
        self.model_name = model_name
        self.provider = provider
        self.system_prompt = system_prompt

        if self.provider == 'openai':
            if api_key is None:
                raise ValueError("Please provide api_key for OpenAI.")
            self.model = AsyncOpenAI(base_url=base_url, api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def openai_to_format(self, response):
        return Response(
            content=response.message.content,
            tool_calls=response.message.tool_calls,
        )

    async def chat(self, messages: list[dict], tools: list[dict] = []):
        # Inject system prompt
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        response = await self.model.chat.completions.create(
            model=self.model_name,
            messages=full_messages,
            tools=tools,
        )
        return [self.openai_to_format(_) for _ in response.choices]


if __name__ == "__main__":
    import asyncio

    async def test():
        model = Model(
            model_name='gpt-4o',
            provider='openai',
            api_key='your-api-key-here'
        )
        res = await model.chat([{"role": "user", "content": "What is the capital of France?"}])
        print(res[0])

    asyncio.run(test())
