# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run train.py when the container launches
CMD ["python", "train.py"]