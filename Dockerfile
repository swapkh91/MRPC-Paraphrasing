FROM python:3.7-slim-stretch

MAINTAINER Swapnesh Khare <swapkh91@gmail.com>

# This prevents Python from writing out pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# This keeps Python from buffering stdin/stdout
ENV PYTHONUNBUFFERED 1

# set work directory
WORKDIR /src/app

# copy requirements.txt
COPY ./requirements.txt /src/app/requirements.txt

# install system dependencies
RUN apt-get update \
    && apt-get -y --no-install-recommends install gcc make \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# copy project
COPY . .

# Move models and related files
WORKDIR /src/app/ML_Model
COPY ./xgb_clf.pickle.dat /src/app/ML_Model/xgb_clf.pickle.dat
COPY ./SiameseLSTM.h5 /src/app/ML_Model/SiameseLSTM.h5
COPY ./mrpc.w2v /src/app/ML_Model/mrpc.w2v
COPY ./weights.json /src/app/ML_Model/weights.json

# set work directory
WORKDIR /src/app

# set app port
EXPOSE 4002

ENTRYPOINT [ "python" ] 

# Run app.py when the container launches
CMD [ "flask_app.py","run","--host","0.0.0.0"] 