from typing import Optional

from pydantic import BaseModel

# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
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
