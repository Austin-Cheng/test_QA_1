# -*- coding: utf-8 -*-

import os

import uvicorn
from dotenv import load_dotenv

from common.logger import logging

# 自动搜索.env文件
load_dotenv(dotenv_path=('.envs'), verbose=True)
# 设置项目路径
os.environ['PROJECT_PATH'] = os.path.dirname(os.path.abspath(__file__))


def run_api_server():
    """
    启动服务
    """
    logging.info(f'Run api server...')
    port: int = os.getenv('PORT', 8080)
    logging.info(f"bind ip is http://0.0.0.0:{port}")

    # start server
    uvicorn.run(
        'app.fastapi_app:app',
        host='0.0.0.0',
        port=int(port),
    )


if __name__ == '__main__':
    run_api_server()
