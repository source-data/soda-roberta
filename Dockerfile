FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN apt-get update \
&& pip install --upgrade pip setuptools \
&& pip install --upgrade pip \
# reinstall numpy with version compatible with nvcr.io/nvidia/pytorch:22.01-py3
&& pip install numpy==1.22.2 \
# && pip install tensorflow==2.4.0 \
&& pip install python-dotenv==0.15.0 \
&& pip install nltk==3.5 \
&& pip install scikit-learn==0.24.0 \
# && pip install transformers==4.1.1 \
# && pip install transformers==4.15.0 \
&& pip install transformers==4.20.0 \
# && pip install datasets==1.3.0 \
&& pip install datasets==1.17.0 \
&& pip install seqeval==1.2.2 \
&& pip install celery==5.0.5 \
&& pip install flower==0.9.7 \
&& pip install spacy==2.3.5 \
&& pip install lxml==4.6.2 \
&& pip install neo4j==4.1.1 \
# download language models for spacy \
&& python -m spacy download en_core_web_sm \
# apparently need separate installation for progress bar stuff in jupyter
# https://ipywidgets.readthedocs.io/en/stable/user_install.html
&& pip install ipywidgets
# optional for plotting
RUN pip install plotly \
&& dash=2.5.0 \
&& jupyter-dash \
&& pandas

# Clear cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*1
