import asyncio
import os

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

# 导入子应用
from discord_bot.bot import app as discord_app, client, TOKEN
from wecom_bot.demo_server import app as wecom_app

# 加载环境变量
load_dotenv()

# 创建主应用
main_app = FastAPI(title="Bot Service")


# 挂载子应用
main_app.mount("/discord", discord_app)
main_app.mount("/wecom", wecom_app)


@main_app.get("/")
async def root():
    """服务状态"""
    return {
        "status": "running",
        "services": {
            "discord": {
                "ready": client.is_ready(),
                "name": client.user.name if client.user else None,
            },
            "wecom": "available",
        }
    }


@main_app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


async def main():
    # Railway 使用 PORT 环境变量
    port = int(os.getenv("PORT", 8000))
    config = uvicorn.Config(main_app, host="0.0.0.0", port=port, log_level="debug")
    server = uvicorn.Server(config)

    print(f"Starting web server on http://0.0.0.0:{port}")
    print("  - Discord dashboard: http://localhost:{port}/discord")
    print(f"  - Wecom callback: http://localhost:{port}/wecom/ai-bot/callback/demo/{{botid}}")
    print("Starting Discord bot...")

    await asyncio.gather(
        server.serve(),
        client.start(TOKEN),
    )


if __name__ == "__main__":
    asyncio.run(main())
