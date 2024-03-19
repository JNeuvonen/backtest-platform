from fastapi import APIRouter, Response, status

from context import HttpResponseContext
from query_code_preset import CodePresetQuery
from request_types import BodyCreateCodePreset


router = APIRouter()


class RoutePaths:
    CODE_PRESET = "/"
    FETCH_ALL_BY_CATEGORY = "/all/{category}"
    BY_ID = "/{id}"


@router.get(RoutePaths.FETCH_ALL_BY_CATEGORY)
async def route_fetch_by_preset_id(category):
    with HttpResponseContext():
        code_presets = CodePresetQuery.retrieve_all(category)
        return {"data": code_presets}


@router.post(RoutePaths.CODE_PRESET)
async def route_create_code_preset(body: BodyCreateCodePreset):
    with HttpResponseContext():
        id = CodePresetQuery.create_entry(body.model_dump())
        return {"id": id}


@router.get(RoutePaths.BY_ID)
async def route_fetch_by_id(id):
    with HttpResponseContext():
        code_preset = CodePresetQuery.fetch_one_by_id(id)
        return {"data": code_preset}
