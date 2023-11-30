import os
import json
import shutil
import urllib.request
import time

# API configuration
http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
key = "ILUS28ltigP0UirSewQPZmKLCqFQEtg1"
secret = "vMLb1ufvKfjIgIcLShervHy723o7cdLI"
boundary = '----------%s' % hex(int(time.time() * 1000))

# Input and output directories
input_directory = "cuhk_images"
output_directory_detected = "cuhk_images_portrait"
output_directory_undetected = "face_not_detected"
output_multiple_faces = "multiple_faces"

# Loop through JPG images in the input directory
for filename in os.listdir(input_directory):

    # Check if the file already exists in any of the output folders
    output_paths = [
        os.path.join(output_directory_detected, filename),
        os.path.join(output_directory_undetected, filename),
        os.path.join(output_multiple_faces, filename)
    ]

    if any(os.path.exists(output_path) for output_path in output_paths):
        print(f"{filename} already exists in output folders, skipping...")
        continue

    filepath = os.path.join(input_directory, filename)

    # Prepare data for the API request
    data = []
    data.append(b'--%s' % boundary.encode('utf-8'))
    data.append(b'Content-Disposition: form-data; name="%s"\r\n' % b'api_key')
    data.append(key.encode('utf-8'))
    data.append(b'--%s' % boundary.encode('utf-8'))
    data.append(b'Content-Disposition: form-data; name="%s"\r\n' % b'api_secret')
    data.append(secret.encode('utf-8'))
    data.append(b'--%s' % boundary.encode('utf-8'))
    fr = open(filepath, 'rb')
    data.append(b'Content-Disposition: form-data; name="%s"; filename="%s"' % (b'image_file', filename.encode('utf-8')))
    data.append(b'Content-Type: %s\r\n' % b'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append(b'--%s--\r\n' % boundary.encode('utf-8'))

    http_body = b'\r\n'.join(data)

    # Build http request
    req = urllib.request.Request(http_url)
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
    req.data = http_body

    try:
        # Post data to the server
        resp = urllib.request.urlopen(req, timeout=5)
        # Get response
        qrcont = resp.read()
        response_data = json.loads(qrcont.decode('utf-8'))

        # Check the number of detected faces
        face_num = response_data.get('face_num', 0)
        if face_num == 1:
            output_path = os.path.join(output_directory_detected, filename)
        elif face_num > 1:
            output_path = os.path.join(output_multiple_faces, filename)
        else:
            output_path = os.path.join(output_directory_undetected, filename)

        # Copy the image to the appropriate output folder
        shutil.copy(filepath, output_path)
        print(f"{filename} copied to {output_path}")

    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.read().decode('utf-8'))

print("Completed face detection using Face++ API.")