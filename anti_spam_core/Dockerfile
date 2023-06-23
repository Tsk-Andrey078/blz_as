FROM python:3.9
WORKDIR /anti_spam_core
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /anti_spam_core
CMD ["python3", "main.py"]

