import sys
sys.path.append("/home/jovyan")
sys.path.append("/home/jovyan/work")

from outboxml.main_predict import app
import uvicorn




def run_service(host="127.0.0.1", port=8000):
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_service()