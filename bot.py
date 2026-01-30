import asyncio
import os
import random
from datetime import datetime

import discord
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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

# 记录启动时间
start_time = datetime.now()


@app.get("/", response_class=HTMLResponse)
async def homepage():
    """Bot 状态首页"""
    uptime = datetime.now() - start_time
    hours, remainder = divmod(int(uptime.total_seconds()), 3600)
    minutes, seconds = divmod(remainder, 60)
    uptime_str = f"{hours}h {minutes}m {seconds}s"

    bot_status = "🟢 在线" if client.is_ready() else "🔴 离线"
    bot_name = client.user.name if client.user else "未连接"
    bot_id = client.user.id if client.user else "N/A"
    bot_avatar = client.user.avatar.url if client.user and client.user.avatar else ""
    guild_count = len(client.guilds) if client.is_ready() else 0

    # 获取服务器列表
    guilds_html = ""
    if client.is_ready():
        for guild in client.guilds:
            member_count = guild.member_count or 0
            guilds_html += f"""
            <div class="guild-card">
                <div class="guild-name">{guild.name}</div>
                <div class="guild-info">成员: {member_count}</div>
            </div>
            """

    html = f"""
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Discord Bot Dashboard</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            body {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 40px 20px;
            }}
            .container {{
                max-width: 800px;
                margin: 0 auto;
            }}
            .card {{
                background: white;
                border-radius: 16px;
                padding: 30px;
                margin-bottom: 20px;
                box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            }}
            .header {{
                display: flex;
                align-items: center;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .avatar {{
                width: 80px;
                height: 80px;
                border-radius: 50%;
                background: #5865F2;
            }}
            .bot-name {{
                font-size: 28px;
                font-weight: bold;
                color: #2c3e50;
            }}
            .status {{
                font-size: 18px;
                margin-top: 5px;
            }}
            .stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            .stat-item {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 12px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 24px;
                font-weight: bold;
                color: #5865F2;
            }}
            .stat-label {{
                color: #666;
                margin-top: 5px;
            }}
            .section-title {{
                font-size: 20px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 15px;
            }}
            .guilds {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }}
            .guild-card {{
                background: #f8f9fa;
                padding: 15px;
                border-radius: 10px;
            }}
            .guild-name {{
                font-weight: 600;
                color: #2c3e50;
            }}
            .guild-info {{
                color: #666;
                font-size: 14px;
                margin-top: 5px;
            }}
            .refresh-btn {{
                background: #5865F2;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                margin-top: 20px;
            }}
            .refresh-btn:hover {{
                background: #4752c4;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card">
                <div class="header">
                    {"<img src='" + bot_avatar + "' class='avatar'>" if bot_avatar else "<div class='avatar'></div>"}
                    <div>
                        <div class="bot-name">{bot_name}</div>
                        <div class="status">{bot_status}</div>
                    </div>
                </div>
                <div class="stats">
                    <div class="stat-item">
                        <div class="stat-value">{guild_count}</div>
                        <div class="stat-label">服务器</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{uptime_str}</div>
                        <div class="stat-label">运行时间</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value">{bot_id}</div>
                        <div class="stat-label">Bot ID</div>
                    </div>
                </div>
                <button class="refresh-btn" onclick="location.reload()">刷新状态</button>
            </div>

            <div class="card">
                <div class="section-title">已加入的服务器</div>
                <div class="guilds">
                    {guilds_html if guilds_html else "<p style='color:#666'>暂无服务器</p>"}
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    return html


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
    # Railway 使用 PORT 环境变量
    port = int(os.getenv("PORT", 8000))
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
    server = uvicorn.Server(config)

    print(f"Starting web server on http://0.0.0.0:{port}")
    print("Starting Discord bot...")

    await asyncio.gather(
        server.serve(),
        client.start(TOKEN),
    )


if __name__ == "__main__":
    asyncio.run(main())
