from fastapi import APIRouter, Response, status

from context import HttpResponseContext
from query_data_transformation import DataTransformationQuery
from request_types import BodyCreateDataTransformation


router = APIRouter()


class RoutePaths:
    DATA_TRANSFORMATION = "/"


@router.get(RoutePaths.DATA_TRANSFORMATION)
async def route_get_data_transformations():
    with HttpResponseContext():
        data_transformations = (
            DataTransformationQuery.fetch_all_data_transformations_without_dataset()
        )
        return {"data": data_transformations}


@router.post(RoutePaths.DATA_TRANSFORMATION)
async def route_post_data_transformation(body: BodyCreateDataTransformation):
    with HttpResponseContext():
        id = DataTransformationQuery.create_entry(body.model_dump())
        return Response(
            content=f"{str(id)}",
            media_type="text/plain",
            status_code=status.HTTP_200_OK,
        )
