FROM python:3.10-slim as base

RUN mkdir /app
WORKDIR /app
ADD . .

FROM base as builder

RUN apt-get update && apt-get install -y apt-utils make gcc
RUN make install

FROM base

COPY --from=builder /app/.venv /app/.venv

EXPOSE 8888

ENTRYPOINT . /app/.venv/bin/activate; \
  jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --notebook-dir=/app/notebooks --allow-root
