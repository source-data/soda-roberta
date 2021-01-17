FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update
RUN pip install --upgrade pip setuptools
RUN pip install python-dotenv
RUN pip install nltk
RUN pip install sklearn
RUN pip install transformers
RUN pip install datasets
# uninstalling TensorBoard 1.15.0+nv is making problem with 'cannot convert 0 to DType'
# RUN pip uninstall -y tensorflow
RUN pip uninstall -y nvidia-tensorboard
RUN pip install tensorflow
RUN pip install seqeval
RUN pip install celery
RUN pip install flower
RUN pip install spacy
# download language models for spacy
RUN python -m spacy download en_core_web_sm