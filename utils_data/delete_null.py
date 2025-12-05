import json

# Load the original JSON data
with open('/home/zwding/ly/M3D/Data/all_T2A.json', 'r') as f:
    data = json.load(f)
# print(data)

# Filter out entries where 'image' or 'conversations' are empty
filtered_data = [entry for entry in data['train'] if entry.get('image') and entry.get('conversations')]

# Save the filtered data into a new file
with open('/home/zwding/ly/M3D/Data/all_T2A2.json', 'w') as f:
    json.dump(filtered_data, f, indent=4)
