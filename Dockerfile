FROM python:3.13
WORKDIR /code
COPY ./backend/requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt
COPY ./backend/transaction_anomalies_dataset.csv /code/
COPY ./backend/preprocessing.py /code/preprocessing.py
COPY ./backend/generate_model.py /code/generate_model.py
COPY ./backend/app /code/app
RUN python generate_model.py
RUN mv rfr_v1.pkl app/rfr_v1.pkl
RUN rm /code/transaction_anomalies_dataset.csv
CMD ["fastapi", "run", "app/main.py", "--port", "80"]