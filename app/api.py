from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import Dict, Optional

from fastapi import FastAPI,Request

from app.schema import PredictPayload 
from ml_scripts import config,main,predict
from ml_scripts.config import logger



#Define the Application 
app= FastAPI(
    title ="AI Tags",
    description="Predict relavent tags given a text input",
    version="0.1"
)

@app.on_event("startup")
def load_artifacts():
    global artifacts
    artifacts=main.load_artifacts(model_dir=config.MODEL_DIR)
    logger.info("Ready for Inferrence")


def construct_response(f):
    """ 
    Construct a JSON response fro an endpoint's results .
    """
    @wraps(f)
    def wrap(request:Request,*args,**kwargs):
        results=f(request,*args,**kwargs)

        reponse={
            "message":results["message"],
            "method" : request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url":request.url._url,
        }

        #Adding Data
        if "data" in results:
            reponse["data"]=results["data"]

        return response 

    return wrap 


@app.get("/",tags=["General"])
@construct_response
def _index(request :Request):
    """
    Health Check

    
    """
    response={
        "message":HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data":{},
    }

    return response

@app.post("/predict",tags=["Prediction"])
@construct_response
def _predict(request:Request,payload : PredictPayload)->Dict:
    """
    Predicts tags for a list of text using best run time 

    """
    texts=[item.text for item in payload.texts]
    predictions=predict.predict(texts=texts,artifacts=artifacts)
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predictions": predictions},
    }

    return response

@app.get("/params", tags=["Parameters"])
@construct_response
def _params(request: Request) -> Dict:
    """Get parameter values used for a run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "params": vars(artifacts["params"]),
        },
    }
    return response


@app.get("/params/{param}", tags=["Parameters"])
@construct_response
def _param(request: Request, param: str) -> Dict:
    """Get a specific parameter's value used for a run."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "params": {
                param: vars(artifacts["params"]).get(param, ""),
            }
        },
    }
    return response


@app.get("/performance", tags=["Performance"])
@construct_response
def _performance(request: Request, filter: Optional[str] = None) -> Dict:
    """Get the performance metrics for a run."""
    performance = artifacts["performance"]
    if filter:
        for key in filter.split("."):
            performance = performance.get(key, {})
        data = {"performance": {filter: performance}}
    else:
        data = {"performance": performance}
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": data,
    }
    return response
