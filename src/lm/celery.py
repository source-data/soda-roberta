from celery import Celery

app = Celery(
    'src.lm.dataprep',
    backend='rpc',
    broker='pyamqp://guest@rabbitmq:5672',
    # broker='pyamqp://guest@localhost:5672'
    include=['src.lm.dataprep']
)
