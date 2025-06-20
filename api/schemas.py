from pydantic import BaseModel
from typing import List

class TransactionInput(BaseModel):
    features: List[float]  # Must be 29 features: V1-V28 + Amount
