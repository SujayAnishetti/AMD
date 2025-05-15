import requests
import json
import os

def upload_annotations_to_label_studio(json_file_path, project_id, api_key):
    url = f"http://localhost:8080/api/projects/{project_id}/tasks/bulk/"
    
    with open(json_file_path, 'r') as f:
        annotations = json.load(f)
    
    tasks = []
    for annotation in annotations:
        tasks.append({
            "data": {
                "image": f"http://localhost:8080/media/{annotation['images/11.PNG']}"  # Provide image URL
            },
            "annotations": [{
                "result": [{
                    "from_name": "RPE",  # Label name
                    "to_name": "image",
                    "type": "polygon",
                    "points": annotation["points"]
                }]
            }]
        })
    
    headers = {'Authorization': f"Token {api_key}", 'Content-Type': 'application/json'}
    
    response = requests.post(url, headers=headers, json=tasks)
    
    if response.status_code == 201:
        print(f"✅ Annotations uploaded successfully.")
    else:
        print(f"❌ Failed to upload annotations. Status code: {response.status_code}")

# Example usage
json_dir = "annotations"  # Replace with your directory containing JSON files
project_id = 2  # Replace with your project ID
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA1NDE3MDM0OCwiaWF0IjoxNzQ2OTcwMzQ4LCJqdGkiOiJjOWQxMmIwM2Y5Yjg0N2RhOWEzYmQ4ZjM1MTJjOThhNCIsInVzZXJfaWQiOjF9.rQFG5oGyTZtUEc5VCT1aK8HRTEi2is_vlgx04G-U4OA"  # Replace with your Label Studio API key

for json_file in os.listdir(json_dir):
    if json_file.endswith("_annotations.json"):
        json_file_path = os.path.join(json_dir, json_file)
        upload_annotations_to_label_studio(json_file_path, project_id, api_key)
