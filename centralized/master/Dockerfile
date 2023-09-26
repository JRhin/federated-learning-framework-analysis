FROM python:latest
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 80
#RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
CMD ["python", "master_node.py"]
