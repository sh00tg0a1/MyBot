#!/usr/bin/env python
# coding=utf-8
# 文档：https://developer.work.weixin.qq.com/document/path/101039

import asyncio
import base64
import hashlib
import json
import logging
import os
import random
import string
import time
import threading

import requests
import uvicorn
from Crypto.Cipher import AES
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response

from wecom_bot.WXBizJsonMsgCrypt import WXBizJsonMsgCrypt

app = FastAPI(title="Wecom Bot API")


@app.get("/")
async def wecom_root():
    """Wecom 服务状态"""
    return {
        "service": "wecom_bot",
        "status": "running",
        "endpoints": {
            "verify": "GET /ai-bot/callback/demo/{botid}",
            "message": "POST /ai-bot/callback/demo/{botid}",
        }
    }


@app.get("/test")
async def test_endpoint():
    """测试端点 - 检查配置是否正确"""
    import os
    token = os.getenv('Token', '')
    encoding_aes_key = os.getenv('EncodingAESKey', '')
    return {
        "token_configured": bool(token),
        "encoding_aes_key_configured": bool(encoding_aes_key),
        "token_length": len(token) if token else 0,
        "encoding_aes_key_length": len(encoding_aes_key) if encoding_aes_key else 0,
    }


# 导入大模型
from utils.llm import LLM, AnthropicLLM

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _generate_random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))


def _process_encrypted_image(image_url, aes_key_base64):
    """
    下载并解密加密图片

    参数:
        image_url: 加密图片的URL
        aes_key_base64: Base64编码的AES密钥(与回调加解密相同)

    返回:
        tuple: (status: bool, data: bytes/str) 
               status为True时data是解密后的图片数据，
               status为False时data是错误信息
    """
    try:
        # 1. 下载加密图片
        logger.info("开始下载加密图片: %s", image_url)
        response = requests.get(image_url, timeout=15)
        response.raise_for_status()
        encrypted_data = response.content
        logger.info("图片下载成功，大小: %d 字节", len(encrypted_data))

        # 2. 准备AES密钥和IV
        if not aes_key_base64:
            raise ValueError("AES密钥不能为空")

        # Base64解码密钥 (自动处理填充)
        aes_key = base64.b64decode(aes_key_base64 + "=" * (-len(aes_key_base64) % 4))
        if len(aes_key) != 32:
            raise ValueError("无效的AES密钥长度: 应为32字节")

        iv = aes_key[:16]  # 初始向量为密钥前16字节

        # 3. 解密图片数据
        cipher = AES.new(aes_key, AES.MODE_CBC, iv)
        decrypted_data = cipher.decrypt(encrypted_data)

        # 4. 去除PKCS#7填充 (Python 3兼容写法)
        pad_len = decrypted_data[-1]  # 直接获取最后一个字节的整数值
        if pad_len > 32:  # AES-256块大小为32字节
            raise ValueError("无效的填充长度 (大于32字节)")

        decrypted_data = decrypted_data[:-pad_len]
        logger.info("图片解密成功，解密后大小: %d 字节", len(decrypted_data))

        return True, decrypted_data

    except requests.exceptions.RequestException as e:
        error_msg = f"图片下载失败 : {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    except ValueError as e:
        error_msg = f"参数错误 : {str(e)}"
        logger.error(error_msg)
        return False, error_msg

    except Exception as e:
        error_msg = f"图片处理异常 : {str(e)}"
        logger.error(error_msg)
        return False, error_msg


def MakeTextStream(stream_id, content, finish):
    plain = {
                "msgtype": "stream",
                "stream": {
                    "id": stream_id,
                    "finish": finish,
                    "content": content
                }
            }
    return json.dumps(plain, ensure_ascii=False)


def MakeImageStream(stream_id, image_data, finish):
    image_md5 = hashlib.md5(image_data).hexdigest()
    image_base64 = base64.b64encode(image_data).decode('utf-8')

    plain = {
                "msgtype": "stream",
                "stream": {
                    "id": stream_id,
                    "finish": finish,
                    "msg_item": [
                        {
                            "msgtype": "image",
                            "image": {
                                "base64": image_base64,
                                "md5": image_md5 
                            }
                        }
                    ]
                }
            }
    return json.dumps(plain)


def EncryptMessage(receiveid, nonce, timestamp, stream):
    logger.info("开始加密消息，receiveid=%s, nonce=%s, timestamp=%s", receiveid, nonce, timestamp)
    logger.debug("发送流消息: %s", stream)

    wxcpt = WXBizJsonMsgCrypt(os.getenv('Token', ''), os.getenv('EncodingAESKey', ''), receiveid)
    ret, resp = wxcpt.EncryptMsg(stream, nonce, timestamp)
    if ret != 0:
        logger.error("加密失败，错误码: %d", ret)
        return

    stream_id = json.loads(stream)['stream']['id']
    finish = json.loads(stream)['stream']['finish']
    logger.info("回调处理完成, 返回加密的流消息, stream_id=%s, finish=%s", stream_id, finish)
    logger.debug("加密后的消息: %s", resp)

    return resp


# ============================================================
# Mock LLM - 模拟大模型，用于测试流式协议（不依赖外部服务）
# ============================================================

CACHE_DIR = "/tmp/llm_demo_cache"
MAX_STEPS = 10


class MockLLM:
    """模拟大模型，分步返回结果，用于测试"""

    def __init__(self):
        self.cache_dir = CACHE_DIR
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def invoke(self, question):
        stream_id = _generate_random_string(10)
        cache_file = os.path.join(self.cache_dir, "%s.json" % stream_id)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'question': question,
                'created_time': time.time(),
                'current_step': 0,
                'max_steps': MAX_STEPS
            }, f)
        return stream_id

    def get_answer(self, stream_id):
        cache_file = os.path.join(self.cache_dir, "%s.json" % stream_id)
        if not os.path.exists(cache_file):
            return "任务不存在或已过期"

        with open(cache_file, 'r', encoding='utf-8') as f:
            task_data = json.load(f)

        current_step = task_data['current_step'] + 1
        task_data['current_step'] = current_step
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(task_data, f)

        response = '收到问题：%s\n' % task_data['question']
        for i in range(current_step):
            response += '处理步骤 %d: 已完成\n' % (i)

        return response

    def is_task_finish(self, stream_id):
        cache_file = os.path.join(self.cache_dir, "%s.json" % stream_id)
        if not os.path.exists(cache_file):
            return True

        with open(cache_file, 'r', encoding='utf-8') as f:
            task_data = json.load(f)

        return task_data['current_step'] >= task_data['max_steps']


# ============================================================
# Real LLM - 流式调用 OpenAI 大模型
# ============================================================

# 内存缓存：保存 LLM 流式回复
# { stream_id: { "content": str, "sent_length": int, "finished": bool, "mock": bool } }
# sent_length 记录已发送给客户端的内容长度，用于计算增量
_llm_cache: dict[str, dict] = {}

# 通过环境变量 USE_MOCK_LLM=true 切换为 MockLLM
USE_MOCK_LLM = os.getenv("USE_MOCK_LLM", "false").lower() == "true"
# 通过环境变量 USE_ANTHROPIC=true 使用 AnthropicLLM，否则使用 OpenAI LLM
USE_ANTHROPIC = os.getenv("USE_ANTHROPIC", "false").lower() == "true"


def _create_llm():
    """根据环境变量创建对应的 LLM 实例"""
    if USE_ANTHROPIC:
        return AnthropicLLM()
    return LLM()


def _stream_worker(stream_id: str, question: str):
    """
    后台线程：流式调用大模型，逐步累积内容到缓存
    """
    try:
        llm = _create_llm()
        logger.info("开始流式生成, stream_id=%s, provider=%s", stream_id, "anthropic" if USE_ANTHROPIC else "openai")

        for chunk in llm.chat_stream(question):
            _llm_cache[stream_id]["content"] += chunk

        _llm_cache[stream_id]["finished"] = True
        logger.info(
            "流式生成完成, stream_id=%s, total_length=%d",
            stream_id,
            len(_llm_cache[stream_id]["content"]),
        )
    except Exception as e:
        logger.error("LLM 流式调用失败: %s", e)
        _llm_cache[stream_id]["content"] += f"\n抱歉，AI 服务异常: {str(e)}"
        _llm_cache[stream_id]["finished"] = True


def llm_invoke(question: str) -> tuple[str, str, bool]:
    """
    调用大模型获取回复（根据 USE_MOCK_LLM 切换真实/模拟）

    真实模式：启动后台线程流式生成，首次返回 finish=False
    模拟模式：使用 MockLLM 分步返回

    Args:
        question: 用户问题

    Returns:
        (stream_id, answer, finish)
    """
    if USE_MOCK_LLM:
        mock = MockLLM()
        stream_id = mock.invoke(question)
        answer = mock.get_answer(stream_id)
        finish = mock.is_task_finish(stream_id)
        _llm_cache[stream_id] = {"mock": True}
        return stream_id, answer, finish

    stream_id = _generate_random_string(10)

    # 初始化缓存
    _llm_cache[stream_id] = {
        "content": "",
        "sent_length": 0,
        "finished": False,
        "mock": False,
    }

    # 启动后台线程进行流式生成
    thread = threading.Thread(
        target=_stream_worker,
        args=(stream_id, question),
        daemon=True,
    )
    thread.start()

    # 等待直到有内容生成（最多等 5 秒）
    for _ in range(50):
        time.sleep(0.1)
        if _llm_cache[stream_id]["content"] or _llm_cache[stream_id]["finished"]:
            break

    content = _llm_cache[stream_id]["content"]
    finished = _llm_cache[stream_id]["finished"]

    # 取增量部分
    delta = content[_llm_cache[stream_id]["sent_length"]:]
    _llm_cache[stream_id]["sent_length"] = len(content)

    if not delta:
        finished = False

    logger.info(
        "首次返回, stream_id=%s, delta_length=%d, finished=%s",
        stream_id, len(delta), finished,
    )
    return stream_id, delta, finished


def llm_get_cached(stream_id: str) -> tuple[str, bool]:
    """
    获取缓存中的 LLM 回复（用于 stream 轮询）

    Args:
        stream_id: 流ID

    Returns:
        (answer, finish)
    """
    cached = _llm_cache.get(stream_id)
    if not cached:
        return "任务不存在或已过期", True

    # MockLLM 使用文件缓存
    if cached.get("mock"):
        mock = MockLLM()
        answer = mock.get_answer(stream_id)
        finish = mock.is_task_finish(stream_id)
        return answer, finish

    content = cached["content"]
    finished = cached["finished"]

    # 取增量：只返回上次发送之后新生成的内容
    delta = content[cached["sent_length"]:]
    cached["sent_length"] = len(content)

    if not delta and not finished:
        # 还在生成但暂时没有新内容
        return "", False

    return delta, finished


@app.get("/ai-bot/callback/demo/{botid}")
async def verify_url(
    request: Request,
    botid: str,
    msg_signature: str,
    timestamp: str,
    nonce: str,
    echostr: str
):
    # 企业创建的自能机器人的 VerifyUrl 请求, receiveid 是空串
    receiveid = ''
    wxcpt = WXBizJsonMsgCrypt(os.getenv('Token', ''), os.getenv('EncodingAESKey', ''), receiveid)

    ret, echostr = wxcpt.VerifyURL(
        msg_signature,
        timestamp,
        nonce,
        echostr
    )

    if ret != 0:
        echostr = "verify fail"

    return Response(content=echostr, media_type="text/plain")


@app.post("/ai-bot/callback/demo/{botid}")
async def handle_message(
    request: Request,
    botid: str,
    msg_signature: str = None,
    timestamp: str = None,
    nonce: str = None
):
    query_params = dict(request.query_params)
    if not all([msg_signature, timestamp, nonce]):
        raise HTTPException(status_code=400, detail="缺少必要参数")
    logger.info("收到消息，botid=%s, msg_signature=%s, timestamp=%s, nonce=%s", botid, msg_signature, timestamp, nonce)

    post_data = await request.body()

    # 智能机器人的 receiveid 是空串
    receiveid = ''
    wxcpt = WXBizJsonMsgCrypt(os.getenv('Token', ''), os.getenv('EncodingAESKey', ''), receiveid)

    ret, msg = wxcpt.DecryptMsg(
        post_data,
        msg_signature,
        timestamp,
        nonce
    )

    if ret != 0:
        raise HTTPException(status_code=400, detail="解密失败")

    data = json.loads(msg)
    logger.debug('Decrypted data: %s', data)
    if 'msgtype' not in data:
        logger.info("不认识的事件: %s", data)
        return Response(content="success", media_type="text/plain")

    msgtype = data['msgtype']
    if (msgtype == 'text'):
        content = data['text']['content']
        logger.info("收到文本消息: %s", content)

        # 调用大模型获取回复
        stream_id, answer, finish = llm_invoke(content)

        stream = MakeTextStream(stream_id, answer, finish)
        resp = EncryptMessage(receiveid, nonce, timestamp, stream)
        return Response(content=resp, media_type="text/plain")
    elif (msgtype == 'stream'):  # case stream
        # 获取缓存中的回复
        stream_id = data['stream']['id']
        answer, finish = llm_get_cached(stream_id)

        stream = MakeTextStream(stream_id, answer, finish)
        resp = EncryptMessage(receiveid, nonce, timestamp, stream)
        return Response(content=resp, media_type="text/plain")
    elif (msgtype == 'image'):
        # 从环境变量获取AES密钥
        aes_key = os.getenv('EncodingAESKey', '')

        # 调用图片处理函数
        success, result = _process_encrypted_image(data['image']['url'], aes_key)
        if not success:
            logger.error("图片处理失败: %s", result)
            return

        # 这里简单处理直接原图回复
        decrypted_data = result
        stream_id = _generate_random_string(10)
        finish = True

        stream = MakeImageStream(stream_id, decrypted_data, finish)
        resp = EncryptMessage(receiveid, nonce, timestamp, stream)
        return Response(content=resp, media_type="text/plain")
    elif (msgtype == 'mixed'):
        # TODO 处理图文混排消息
        logger.warning("需要支持mixed消息类型")
    elif (msgtype == 'event'):
        # TODO 一些事件的处理
        logger.warning("需要支持event消息类型: %s", data)
        return
    else:
        logger.warning("不支持的消息类型: %s", msgtype)
        return

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
