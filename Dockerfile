# Use the Python 3.13 base image
FROM python:3.13

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file and install dependencies
COPY ./backend/requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

# Copy the dataset and all necessary Python files
COPY ./backend/transaction_anomalies_dataset.csv /code/
COPY ./backend/preprocessing.py /code/preprocessing.py
COPY ./backend/generate_model.py /code/generate_model.py
COPY ./backend/app /code/app

# Set PYTHONPATH to ensure modules are discoverable
ENV PYTHONPATH="/code"

# Run the script to generate and save the model
RUN python generate_model.py

# Move the trained model to the app directory
RUN mv rfr_v1.pkl /code/app/rfr_v1.pkl

# Remove the dataset after generating the model to clean up
RUN rm /code/transaction_anomalies_dataset.csv

# # Set the command to run the FastAPI application
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]

CMD ["fastapi", "run", "app/main.py", "--port", "80"]