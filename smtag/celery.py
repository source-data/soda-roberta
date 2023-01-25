from celery import Celery
app = Celery(
    'smtag',
    backend='rpc',
    broker='pyamqp://guest@rabbitmq:5672',
    include=['smtag.extract', 'smtag.dataprep']
)


if __name__ == '__main__':
    app.start()
