from fastapi import APIRouter, Depends

from app.routers import COMMON_ERROR_RESPONSES
from app.schemas import RecommendSizeRequest, RecommendSizeResponse
from app.security import require_api_key
from app.services.sizing import recommend_size as score_sizes

router = APIRouter(tags=["sizing"], dependencies=[Depends(require_api_key)])


@router.post(
    "/recommend-size",
    response_model=RecommendSizeResponse,
    responses=COMMON_ERROR_RESPONSES,
    summary="Best size + a fit note for every size in the chart",
    description=(
        "Scores every size in `size_chart` against the measurements. Chart fields used: "
        "`chest`, `waist`, `shoulder`, `length` (tops/dresses) and `inseam` (lower body); "
        "others are ignored. `fit_preference` shifts what counts as ideal "
        "(slim = snug, loose = roomy)."
    ),
)
def recommend_size(body: RecommendSizeRequest) -> RecommendSizeResponse:
    # Plain `def` (not async): FastAPI runs it in a worker thread. It's quick maths anyway.
    return score_sizes(body)
