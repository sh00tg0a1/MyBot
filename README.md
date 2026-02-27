# Test Bot

多平台聊天机器人服务，支持 Discord 和企业微信（WeCom），集成 OpenAI / Anthropic 大模型。

## 项目结构

```
test_bot/
├── main.py                  # 主入口，同时运行 Web 服务器 + Discord Bot
├── Procfile                 # Railway 部署配置
├── pyproject.toml           # 项目依赖
├── .env                     # 环境变量（不提交到 Git）
├── discord_bot/
│   └── bot.py               # Discord Bot + 状态页面
├── wecom_bot/
│   ├── demo_server.py       # 企业微信智能机器人回调服务
│   ├── WXBizJsonMsgCrypt.py # 企微消息加解密
│   └── ierror.py            # 错误码定义
└── utils/
    └── llm.py               # 大模型调用（OpenAI / Anthropic）
```

## 功能

### Discord Bot (`discord_bot/bot.py`)

- 消息回复（echo）+ 文件转发
- 带 typing 效果（随机 1-2 秒延迟）
- Web 状态页面（Bot 信息、服务器列表）
- API 端点：`/discord/health`、`/discord/bot/info`

### 企业微信智能机器人 (`wecom_bot/demo_server.py`)

- 接收文本消息 → 调用大模型 → 流式回复
- 接收图片消息 → 解密并原图回复
- 支持 Mock LLM 模式用于测试
- API 端点：`/wecom/ai-bot/callback/demo/{botid}`

### 大模型工具 (`utils/llm.py`)

- `LLM` - OpenAI 兼容接口（支持任何 OpenAI API 兼容的服务）
- `AnthropicLLM` - Anthropic 接口（支持 thinking 流式输出）
- 支持单轮对话、流式对话、多轮对话

## 环境变量

```env
# Discord
DISCORD_BOT_TOKEN=你的Discord Bot Token

# 企业微信
Token=企微回调Token
EncodingAESKey=企微回调EncodingAESKey

# 大模型 - OpenAI 兼容
OPENAI_API_KEY=你的API Key
OPENAI_BASE_URL=https://api.minimax.chat/v1
OPENAI_MODEL=MiniMax-M1

# 大模型 - Anthropic
ANTHROPIC_API_KEY=你的API Key
ANTHROPIC_BASE_URL=https://api.minimax.chat/v1
ANTHROPIC_MODEL=MiniMax-M2.1

# 切换
USE_MOCK_LLM=false       # true 使用模拟大模型
USE_ANTHROPIC=false       # true 使用 Anthropic 接口

# 服务
PORT=8000                 # Railway 自动设置
```

## 快速开始

```bash
# 安装依赖
uv sync

# 运行
uv run main.py
```

启动后：

- 主页: http://localhost:8000/
- Discord 状态页: http://localhost:8000/discord/
- 企微回调: http://localhost:8000/wecom/ai-bot/callback/demo/{botid}
- API 文档: http://localhost:8000/docs

## 部署（Railway）

1. 将代码推送到 GitHub
2. 在 Railway 创建项目，连接 GitHub 仓库
3. 设置环境变量
4. Railway 自动检测 `Procfile` 并部署

## 参考资料

- [Discord Developer Portal](https://discord.com/developers/applications/) - Discord 应用管理、Bot Token 获取、Intent 配置
- [企业微信智能机器人 API 文档](https://developer.work.weixin.qq.com/document/path/101039) - 回调配置、消息格式、加解密方案
- [discord.py 文档](https://discordpy.readthedocs.io/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)
- [OpenAI API 文档](https://platform.openai.com/docs/)
- [Anthropic API 文档](https://docs.anthropic.com/)
