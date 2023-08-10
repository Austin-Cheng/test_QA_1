# -*- coding: utf-8 -*-


from fastapi import APIRouter, Request
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError, FastAPIError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import JSONResponse

from common.errorcode.codes import search as code
from common.errorcode.sview import Sview
from .services import search, health


# 初始化FastAPI
def init_app():
    fastApp: FastAPI = FastAPI()
    router = APIRouter()
    router.include_router(search.router)
    router.include_router(health.router)
    fastApp.include_router(router, prefix='/api/v1')
    return fastApp


# create fast_app
app: FastAPI = init_app()

"""
自定义全局HTTP异常处理
"""


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(Sview.TErrorreturn(code.APP_HANDLERS_SERVICE_ALG_SEARCH_NO_RESULT))


"""
自定义全局参数异常处理
"""


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(Sview.TErrorreturn(code.APP_HANDLERS_SERVICE_SEARCH_PARAMS_ERROR))

