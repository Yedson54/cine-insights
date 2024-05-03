FROM ubuntu:22.04
WORKDIR ${HOME}/cine_insights
# Install Python
RUN apt-get -y update && \
    apt-get install -y python3-pip
# Install project dependencies
COPY main.py .
COPY src ./src
COPY requirements.txt .
RUN pip install -r requirements.txt
# Run the application
CMD ["python3", "main.py"]