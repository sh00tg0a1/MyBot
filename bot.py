import asyncio
import os
import random

import discord
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Please set DISCORD_BOT_TOKEN")

# 1. 声明你要监听哪些事件（Intents）
intents = discord.Intents.default()
intents.message_content = True   # 接收消息内容
intents.members = False          # 不需要成员列表可关

# 2. 创建 Discord Client（Gateway 客户端）
client = discord.Client(intents=intents)

# 3. 创建 FastAPI 应用
app = FastAPI(title="Discord Bot API")


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "bot_ready": client.is_ready()}


@app.get("/bot/info")
async def bot_info():
    """获取 bot 信息"""
    if client.user:
        return {
            "name": client.user.name,
            "id": client.user.id,
            "guilds": len(client.guilds),
        }
    return {"error": "Bot not ready"}


# === Gateway 事件 1：READY ===
@client.event
async def on_ready():
    print("====== READY EVENT ======")
    print(f"Bot logged in as {client.user}")
    print(f"Bot id: {client.user.id}")
    print("=========================")


# === Gateway 事件 2：MESSAGE_CREATE ===
@client.event
async def on_message(message: discord.Message):
    print("====== MESSAGE EVENT ======")
    print(f"Guild   : {message.guild}")
    print(f"Channel : {message.channel}")
    print(f"Author  : {message.author}")
    print(f"Content : {message.content}")
    print(f"Attachments: {len(message.attachments)}")
    for att in message.attachments:
        print(f"  - {att.filename} ({att.content_type}, {att.size} bytes)")
        print(f"    URL: {att.url}")
    print("===========================")

    # 忽略 bot 自己
    if message.author == client.user:
        return

    # 将用户的话直接回复
    if message.content:
        async with message.channel.typing():
            await asyncio.sleep(random.uniform(1, 2))
            await message.channel.send(message.content)

    # 把用户发的文件再发回去
    for att in message.attachments:
        file = await att.to_file()
        async with message.channel.typing():
            await asyncio.sleep(random.uniform(1, 2))
            await message.channel.send(f"收到文件: {att.filename}", file=file)


# 4. 同时运行 Web 服务器和 Discord Bot
async def main():
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)

    print("Starting web server on http://0.0.0.0:8000")
    print("Starting Discord bot...")

    await asyncio.gather(
        server.serve(),
        client.start(TOKEN),
    )


if __name__ == "__main__":
    asyncio.run(main())
