# nvidia-docker run --shm-size 8G --rm -it -v /raid/lemberge/vsearch:/workspace/vsearch -p12345:6005 vsearch:dev
# https://download.nvidia.com/XFree86/Linux-x86_64/396.51/README/editxconfig.html
# https://www.pugetsystems.com/labs/hpc/NVIDIA-Docker2-with-OpenGL-and-X-Display-Output-1527/
# https://stackoverflow.com/questions/48235040/run-x-application-in-a-docker-container-reliably-on-a-server-connected-via-ssh-w
# 
# nvidia-docker run --shm-size 8G --rm -it -v /raid/lemberge/vsearch:/workspace/vsearch -p12345:6005 -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all vsearch:dev
FROM nvcr.io/nvidia/pytorch:20.12-py3


COPY . /app
RUN apt-get update && \
    pip install --upgrade pip setuptools && \
    pip install -r /app/requirements.txt && \
    python -c "import nltk; nltk.download('averaged_perceptron_tagger')" \
    rm -Rf /app

WORKDIR /app