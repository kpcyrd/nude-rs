FROM python:3
RUN pip install --upgrade nudepy
COPY data/ /data
COPY python/bench.py .
CMD ["./bench.py", "/data/*"]
