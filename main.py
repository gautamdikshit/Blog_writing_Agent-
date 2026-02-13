from fastapi import FastAPI
from pydantic import BaseModel
from agent import run_blog_agent
import asyncio

app = FastAPI()

class BlogRequest(BaseModel):
    topic: str

@app.post("/generate-blog")
async def create_blog(request: BlogRequest):
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, run_blog_agent, request.topic)
    return {"blog": result}