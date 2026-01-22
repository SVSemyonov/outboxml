import pickle
import io

from loguru import logger
from minio import Minio
from minio.error import S3Error


class Client:
    def load_ds_result(self):
        pass
    def save_ds_result(self):
        pass

class  Minio(Client):
    def __init__(self,
                endpoint="localhost:9000",
                access_key:str="minio",
                secret_key:str="Strong#Pass#2022",
                secure: bool=False
                ):
        self.client = Minio(endpoint=endpoint,
                            access_key=access_key,
                            secret_key=secret_key,
                            secure=secure)

    def load_ds_result(self,):
        try:
            response = self.client.get_object(bucket_name, result_name)
            data = response.read()
            response.close()
            response.release_conn()
            return pickle.loads(data)
        except S3Error as exc:
            logger.error('Error while loading result from Minio||'+ str(exc))
