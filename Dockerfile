FROM python:3.8-slim-buster

RUN apt-get update && yes | apt-get upgrade
RUN apt-get install -y python-pip
RUN pip install --upgrade pip

RUN pip install requests
RUN pip install Flask
RUN pip install numpy
RUN pip install mysql-connector-python



ADD quality_master /quality_master
WORKDIR /quality_master
EXPOSE 8081
CMD ["python", "wsgi_api_quality.py"]

