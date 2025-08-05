FROM python:3.12-slim


WORKDIR /app


RUN useradd --create-home --shell /bin/bash app
USER app


COPY --chown=app:app requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt

ENV PATH="/home/app/.local/bin:${PATH}"

COPY --chown=app:app . .

EXPOSE 7860

CMD ["python", "app.py"]