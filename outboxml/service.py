
import config
from outboxml.main_predict import app
import uvicorn

from outboxml.main_release import MLFLowRelease



def main(host="127.0.0.1", port=8000, group_name: str = 'example'):
    MLFLowRelease(config=config).load_model_to_source_from_mlflow(group_name=group_name)
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()