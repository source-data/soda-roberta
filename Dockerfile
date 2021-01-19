FROM nvcr.io/nvidia/pytorch:20.12-py3

RUN apt-get update
RUN pip install --upgrade pip setuptools
RUN pip install python-dotenv==0.15.0
RUN pip install nltk==3.5
RUN pip install scikit-learn==0.24.0
RUN pip install transformers==4.2.1
RUN pip install datasets==1.2.1
# uninstalling TensorBoard 1.15.0+nv is making problem with 'cannot convert 0 to DType'
# RUN pip uninstall -y tensorflow
RUN pip uninstall -y nvidia-tensorboard
RUN pip install tensorflow==1.15.0
RUN pip install seqeval==1.2.2
RUN pip install celery==5.0.5
RUN pip install flower==0.9.7
RUN pip install spacy==2.3.5
RUN pip install lxml==4.6.2
# download language models for spacy
RUN python -m spacy download en_core_web_sm