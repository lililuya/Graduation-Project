from fastapi import FastAPI

# 创建 FastAPI 实例
app = FastAPI()

# 定义路由和处理函数
@app.get("/")
async def read_root():
    return {"message": "Hello, FastAPI!"}

# 启动 FastAPI 应用并监听指定端口
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
