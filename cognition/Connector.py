# -*- coding:utf-8 -*-


import base64
import json
import aiohttp
from nebula2.gclient.net import ConnectionPool
from nebula2.Config import Config
from typing import List


class RequestNebula:
    def __init__(self, ips: List[str], ports: List[str], user: str, password: str):
        self.ips = ips
        self.ports = ports
        self.user = user
        self.password = password
        config = Config()
        config.max_connection_pool_size = 3
        self.connect_pool = ConnectionPool()
        assert self.connect_pool.init([(ip, int(port)) for ip, port in zip(self.ips, self.ports)], config)

    async def execute(self, sql):
        with self.connect_pool.session_context(self.user, self.password) as client:
            result = client.execute(sql)
        return self._parse_result(result)

    async def execute_json(self, sql):
        with self.connect_pool.session_context(self.user, self.password) as client:
            result = client.execute_json(sql)
        return result

    @staticmethod
    def _parse_result(result):
        records = []
        error_msg = result.error_msg()
        if error_msg:
            raise Exception(error_msg)
        for record in result:
            records.append(record.values())
        return records

    def __del__(self):
        self.connect_pool.close()


class RequestOrientDB:
    headers = {
        "Accept-Encoding": "gzip,deflate",
        "Content-Type": "application/json",
        "Connection": "close"
    }

    def __init__(self, ips: list, ports: list, user: str, password: str):
        self.ip = ips[0]
        self.ip = "10.2.174.230"
        self.port = ports[0]
        pwd = base64.b64decode(password).decode()
        self.user = user
        self.password = pwd
        self.url = 'http://{ip}:{port}/command/{space}/sql/'
        self.class_info_url = 'http://{ip}:{port}/class/{space}/'

        self.space = None

    def set_space(self, space):
        self.space = space

    async def execute(self, body, timeout=300.0):
        timeout = aiohttp.ClientTimeout(total=timeout)
        auth = aiohttp.BasicAuth(login=self.user, password=self.password)
        async with aiohttp.ClientSession(auth=auth) as session:
            url = self.url.format(ip=self.ip, port=self.port, space=self.space)
            response = await session.post(url, json=body, headers=self.headers, timeout=timeout)
            result = await response.text()
        return self._parse_result(result)

    async def get_class_info(self, space, _class, timeout=300.0):
        timeout = aiohttp.ClientTimeout(total=timeout)
        auth = aiohttp.BasicAuth(login=self.user, password=self.password)
        async with aiohttp.ClientSession(auth=auth) as session:
            url = self.class_info_url.format(ip=self.ip, port=self.port, space=space) + _class
            response = await session.get(url, headers=self.headers, timeout=timeout)
            result = await response.text()
        return self._parse_result(result)

    @staticmethod
    def _parse_result(res):
        res = json.loads(res, strict=False)
        if 'errors' in res:
            raise Exception(res['errors'][0]['content'])
        else:
            return res['result']


class RequestOpenSearch:
    def __init__(self, ips: list, ports: list, user: str, password: str):
        self.ip = ips[0]
        self.port = ports[0]
        self.user = user
        self.password = password
        self.headers = {
            "Accept-Encoding": "gzip,deflate",
            "Content-Type": "application/json",
            "Connection": "close"
        }
        self.pre_url = 'http://{ip}:{port}/'.format(ip=self.ip, port=self.port)

    async def get(self, url, body=None, timeout=300.0):
        timeout = aiohttp.ClientTimeout(total=timeout)
        auth = aiohttp.BasicAuth(login=self.user, password=self.password)
        async with aiohttp.ClientSession(auth=auth) as session:
            url = self.pre_url + url
            if body:
                response = await session.get(url, timeout=timeout, json=body, verify_ssl=False, headers=self.headers)
            else:
                response = await session.get(url, timeout=timeout, verify_ssl=False, headers=self.headers)
            result = await response.content.read()
            result = json.loads(result.decode(), strict=False)
        return result


if __name__ == "__main__":
    """
        使用示例
    """
    import asyncio
    import time

    nebula = RequestNebula(
        ips=["10.4.131.25"],
        ports=["9669"],
        user="root",
        password="root"
    )
    start_time = time.time()


    async def test():
        # run three tasks concurrently
        task1 = asyncio.create_task(nebula.get("document230", "show tags;"))
        task2 = asyncio.create_task(nebula.get("document230", "show tags;"))
        task3 = asyncio.create_task(nebula.get("document230", "show tags;"))
        res1 = await task1
        res2 = await task2
        res3 = await task3
        print(res1)
        print(res2)
        print(res3)
        # run a task separately
        res4 = await nebula.get("document230", "show tags;")
        print(res4)


    # run top-level entry point test()
    asyncio.run(test())

    end_time = time.time()
    print(end_time - start_time)
