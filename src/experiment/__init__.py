from abc import ABCMeta
from pydantic import BaseModel, Field


class Experiment(metaclass=ABCMeta):
    desc: str

    def plot(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        raise NotImplementedError


class Record(BaseModel):
    desc: str = Field(..., description='Description of the experiment')
    model_name: str = Field(..., description='Name of the model')
    params: dict = Field(default={}, description='Parameters of the experiment')
    metrics: list = Field(default=[], description='Metrics of the experiment')


class RecordContainer(BaseModel):
    desc: str = Field(..., description='Description of the experiment')
    results: list[Record] = Field(default=[], description='Results of the experiment')
    gloabl_params: dict = Field(..., description='Global parameters of the experiment')
