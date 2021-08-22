import uvicorn

from nlpapi import app


def cli():
    """Start the server

    TODO: Random port?
    """
    uvicorn.run(app, host="0.0.0.0", port=8000)
