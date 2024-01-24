from typing import List

from pydantic import BaseModel


class Point(BaseModel):
    joint: List[List[float]]

    model_config = {
        "json_schema_extra": {
            "examples": [{"joint": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]}]
        }
    }
