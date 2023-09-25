FROM python:latest
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
#RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
CMD ["python", "client_node.py"]
