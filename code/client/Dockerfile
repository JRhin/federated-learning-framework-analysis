FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 80
CMD ["python", "client_node.py"]
