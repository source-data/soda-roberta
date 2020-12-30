FROM nvcr.io/nvidia/pytorch:20.12-py3


RUN apt-get update
RUN pip install --upgrade pip setuptools
RUN pip install python-dotenv
RUN pip install nltk
RUN pip install sklearn
RUN pip install transformers
# uninstalling TensorBoard 1.15.0+nv is making problem with 'cannot convert 0 to DType'
RUN pip uninstall -y tensorflow
RUN pip uninstall -y nvidia-tensorboard
RUN pip install tensorflow
# RUN pip install nvidia-pyindex
# RUN pip install nvidia-tensorflow
# RUN python -c "import nltk; nltk.download('averaged_perceptron_tagger')" \
