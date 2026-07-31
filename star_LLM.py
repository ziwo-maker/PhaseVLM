from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import uvicorn
import time
import json
import base64
import requests
from io import BytesIO

# 初始化 FastAPI 应用
app = FastAPI(title="Qwen3-VL OpenAI Compatible API")

# 全局变量存储模型和处理器
llm_path="/home/data/model/Qwen3-VL-32B/"
model = None
processor = None

# OpenAI 兼容的数据模型
class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: Union[str, List[dict]]

class ChatCompletionRequest(BaseModel):
    model: str = "qwen3-vl-32b"
    messages: List[Message]
    max_tokens: Optional[int] = Field(default=128, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    top_p: Optional[float] = Field(default=0.9, ge=0, le=1)
    stream: Optional[bool] = False

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Usage

def load_model():
    """加载模型和处理器"""
    global model, processor
    print("正在加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
       llm_path, 
        dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(llm_path)
    print("模型加载完成！")

def convert_openai_messages_to_qwen_format(messages: List[Message]) -> List[dict]:
    """将 OpenAI 格式的消息转换为 Qwen 格式"""
    qwen_messages = []
    
    for msg in messages:
        qwen_msg = {"role": msg.role, "content": []}
        
        if isinstance(msg.content, str):
            # 纯文本消息
            qwen_msg["content"].append({"type": "text", "text": msg.content})
        elif isinstance(msg.content, list):
            # 多模态消息
            for item in msg.content:
                if item.get("type") == "text":
                    qwen_msg["content"].append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    image_url = item["image_url"]["url"]
                    # 支持 URL 和 base64
                    qwen_msg["content"].append({"type": "image", "image": image_url})
        
        qwen_messages.append(qwen_msg)
    
    return qwen_messages

@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型"""
    load_model()

@app.get("/")
async def root():
    return {"message": "Qwen3-VL OpenAI Compatible API", "status": "running"}

@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-vl-32b",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "qwen"
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI 兼容的聊天补全接口"""
    try:
        if model is None or processor is None:
            raise HTTPException(status_code=503, detail="模型未加载")
        
        # 转换消息格式
        qwen_messages = convert_openai_messages_to_qwen_format(request.messages)
        
        # 准备输入
        inputs = processor.apply_chat_template(
            qwen_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # 生成参数
        gen_kwargs = {
            "max_new_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "do_sample": request.temperature > 0
        }
        
        # 流式响应
        if request.stream:
            return StreamingResponse(
                stream_generator(inputs, gen_kwargs, request.model),
                media_type="text/event-stream"
            )
        
        # 非流式响应
        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_kwargs)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        output_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        # 构造响应
        response = ChatCompletionResponse(
            id=f"chatcmpl-{int(time.time())}",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=output_text),
                    finish_reason="stop"
                )
            ],
            usage=Usage(
                prompt_tokens=inputs.input_ids.shape[1],
                completion_tokens=len(generated_ids_trimmed[0]),
                total_tokens=inputs.input_ids.shape[1] + len(generated_ids_trimmed[0])
            )
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def stream_generator(inputs, gen_kwargs, model_name):
    """流式生成器"""
    try:
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        streamer = TextIteratorStreamer(
            processor.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        generation_kwargs = {**inputs, **gen_kwargs, "streamer": streamer}
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            chunk = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None
                    }
                ]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
        
        # 发送结束标记
        final_chunk = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "internal_error"
            }
        }
        yield f"data: {json.dumps(error_chunk)}\n\n"

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=9997)