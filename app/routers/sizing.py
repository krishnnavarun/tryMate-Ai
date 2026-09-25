from fastapi import APIRouter, Depends

from app.routers import COMMON_ERROR_RESPONSES
from app.schemas import RecommendSizeRequest, RecommendSizeResponse, SizeFit
from app.security import require_api_key

router = APIRouter(tags=["sizing"], dependencies=[Depends(require_api_key)])


@router.post(
    "/recommend-size",
    response_model=RecommendSizeResponse,
    responses=COMMON_ERROR_RESPONSES,
    summary="Best size + a fit note for every size in the chart",
)
async def recommend_size(body: RecommendSizeRequest) -> RecommendSizeResponse:
    # ---- PHASE 1 STUB -------------------------------------------------------
    # Ignores the measurements: picks the middle size of the chart and scores the
    # others lower the further away they are. Phase 4 replaces this with real scoring.
    sizes = list(body.size_chart.keys())
    middle = len(sizes) // 2

    per_size: dict[str, SizeFit] = {}
    for index, size in enumerate(sizes):
        distance = index - middle
        score = max(0.1, 0.93 - 0.25 * abs(distance))
        if distance == 0:
            note = "Good fit"
        elif distance < 0:
            note = "Stub: likely tight"
        else:
            note = "Stub: likely loose"
        per_size[size] = SizeFit(score=round(score, 2), note=note)

    return RecommendSizeResponse(recommended_size=sizes[middle], per_size=per_size)
