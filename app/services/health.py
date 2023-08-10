# -*- coding: utf-8 -*-
from fastapi import APIRouter


router = APIRouter(prefix='/health', tags=['check'])


@router.get('/ready')
async def ready():
    return "success"


@router.get('/alive')
async def alive():
    return "success"
