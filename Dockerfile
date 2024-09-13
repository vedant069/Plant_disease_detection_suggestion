# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt requirements.txt

# Install the necessary packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app's code into the container
COPY . .

# Expose Streamlit's default port
EXPOSE 8501

# First run the script to download the model, then start the Streamlit app
CMD ["bash", "-c", "python download_model.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]

