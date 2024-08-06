from typing import Annotated

import uvicorn
from fastapi import Body, FastAPI  # , Header
from pydantic import BaseModel

app = FastAPI()


class HelloRequest(BaseModel):
    name: str


class HelloResponse(BaseModel):
    text: str
    name: str


@app.post("/hello")
def hello(data: Annotated[HelloRequest, Body()]) -> HelloResponse:
    return HelloResponse(text=f"Hello, {data.name}", name=data.name)


def main():
    uvicorn.run(app)


if __name__ == '__main__':
    main()
