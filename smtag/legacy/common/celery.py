from celery import Celery

app = Celery(
    'common',
    backend='rpc',
    broker='pyamqp://guest@rabbitmq:5672',
    include=['common.tasks']
)


if __name__ == '__main__':
    app.start()
