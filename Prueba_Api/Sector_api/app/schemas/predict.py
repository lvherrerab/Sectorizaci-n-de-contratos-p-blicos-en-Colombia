from typing import Any, Optional

from pydantic import BaseModel
#from model.processing.validation import DataInputSchema

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    cluster_label: Optional[int]

# Esquema para inputs múltiples


class SingleDataInput(BaseModel):
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "Ejemplo de texto para clusterización."
            }
        }

# Esquema para el resultado del cluster
class ClusterResult(BaseModel):
    cluster_label: Optional[int]