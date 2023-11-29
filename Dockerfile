FROM python:3.8.13

## root dir
WORKDIR /

##
COPY . .

##install requirements
RUN pip install --no-cache-dir --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt


CMD ["python3", "start.py"]