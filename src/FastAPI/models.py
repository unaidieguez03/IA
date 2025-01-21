from fastapi import FastAPI
from pydantic import BaseModel

class Data(BaseModel):
	id: int
	disease: str

