from pydantic import BaseModel

class BankMarketingFeatures(BaseModel):
    poutcome: str
    contact: str
    duration: int
    housing: str
    month: str
    previous: int
    pdays: int
    loan: str
    age: int
    day: int