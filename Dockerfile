FROM python:3.11-slim


# 2. Set the working directory inside the container
WORKDIR /app


# 3. Copy requirements and install dependencies COPY
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt



# 4. Copy the rest of the application code (including the .joblib model)
COPY . .


# 5. Expose port 5000 for the Flask API
EXPOSE 5000


# 6. Command to run the API
CMD ["python", "app.py"] 
