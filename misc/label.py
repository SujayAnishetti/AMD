import requests
import os

# Label Studio API info
label_studio_url = "http://localhost:8080/projects/"  # Your Label Studio instance URL
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA1NDE3MDM0OCwiaWF0IjoxNzQ2OTcwMzQ4LCJqdGkiOiJjOWQxMmIwM2Y5Yjg0N2RhOWEzYmQ4ZjM1MTJjOThhNCIsInVzZXJfaWQiOjF9.rQFG5oGyTZtUEc5VCT1aK8HRTEi2is_vlgx04G-U4OA"  # Replace with your actual API key
project_id = 2  # Replace with your project ID

# Directory containing images and masks
image_dir = 'images/'
mask_dir = 'masks/'

# Function to upload tasks to Label Studio
def upload_task(image_filename, mask_filename):
    # Prepare the task data
    task_data = {
        "data": {
            "image": f"http://localhost:8080/{image_filename}",  # Link to the image
            "RPE": f"http://localhost:8080/{mask_filename}",  # Link to the RPE mask
            # You can add other fields if necessary for other masks or data
        }
    }

    headers = {
        "Authorization": f"Token {api_key}",
    }

    # POST request to upload the task
    response = requests.post(
        f"{label_studio_url}/api/projects/{project_id}/tasks/",
        json=task_data,
        headers=headers
    )
    
    if response.status_code == 201:
        print(f"Task {image_filename} uploaded successfully!")
    else:
        print(f"Failed to upload task {image_filename}: {response.status_code} - {response.text}")

# Loop through your images and upload tasks with corresponding masks
for image_name in os.listdir(image_dir):
    if image_name.endswith(".png"):  # Ensure you're matching your image types
        mask_name = image_name.replace(".png", "_mask.png")  # Assuming mask file names match
        upload_task(image_name, mask_name)

# Get all tasks for the project to verify upload
response = requests.get(
    f"{label_studio_url}/api/projects/{project_id}/tasks/",
    headers={"Authorization": f"Token {api_key}"}
)

if response.status_code == 200:
    tasks = response.json()
    print(f"Tasks uploaded: {len(tasks)}")
else:
    print(f"Error: {response.status_code}")
