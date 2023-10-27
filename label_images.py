import csv
import warnings

# -*- coding: utf-8 -*-
import urllib.request
import time
import json

with warnings.catch_warnings():
    import os
    import numpy as np
    from PIL import Image

# loop through the image directory
# get file name
# get bbox with face++
# compute for the offsets!!
# put in an array [filename, x1, x2, y1, y2]
# append to training_data array

portrait_data = []
img_folder = "training_images"  # the path to the folder of images!!
filename_list = os.listdir(img_folder)  # returns a list of all files in directory


def get_offset(wa, ha, bbox):
    # wa (width of aesthetic area)
    # ha (height of aesthetic area)

    # coordinates of the corners of the aesthetic area
    # x1a = 0, x2a = ha, y1a = 0, y2a = wa

    # divide by ha to normalize
    o1 = float(bbox[0]) / float(ha)  # in relation to x1a
    o2 = float(ha - bbox[1]) / float(ha)  # in relation to x2a
    o3 = float(bbox[2]) / float(wa)  # in relation to x1a
    o4 = float(wa - bbox[3]) / float(wa)  # in relation to x2a
    # print("wa:", wa)
    # print("ha:", ha)
    # print("bbox:", bbox)
    # f"not normalized: {[bbox[0], float(ha - bbox[1]) / float(ha), (bbox[2]), float(wa - bbox[3]) / float(wa)]}")

    offset = [o1, o2, o3, o4]
    # print("offset", offset)
    return offset


def add_offset(w, h, bbox, offset):
    crop_h = int(h * (float(bbox[1]) - float(bbox[0])))
    crop_w = int(w * (float(bbox[3]) - float(bbox[2])))

    new_w = crop_w / (1 - float(offset[2]) - float(offset[3]) + 1e-10)
    new_h = crop_h / (1 - float(offset[0]) - float(offset[1]) + 1e-10)

    r_w = min(w, max(0, new_w))
    r_h = min(h, max(0, new_h))

    x1 = max(0, h * float(bbox[0]) - r_h * float(offset[0]))
    x2 = min(h, x1 + r_h)
    y1 = max(0, w * float(bbox[2]) - r_w * float(offset[2]))
    y2 = min(w, y1 + r_w)

    bbox_aes = [x1 / float(h), x2 / float(h), y1 / float(w), y2 / float(w)]

    return bbox_aes


def get_shape(img, w, h, scale, ratio):
    size = (w, h)
    if ratio:
        if scale is not None:
            size = (scale, scale)
    else:
        if scale is not None:
            if w <= h:
                size = (scale, scale * h // w)
            else:
                size = (scale * w // h, scale)
    return img.resize(size, Image.ANTIALIAS)


# IMAGE PRE PROCESSING!!!

def face_plus_plus(filepath):
    # API configuration
    key = "ILUS28ltigP0UirSewQPZmKLCqFQEtg1"
    secret = "vMLb1ufvKfjIgIcLShervHy723o7cdLI"

    face_http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
    humanbody_http_url = 'https://api-us.faceplusplus.com/humanbodypp/v1/detect'

    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = []
    data.append(b'--%s' % boundary.encode('utf-8'))
    data.append(b'Content-Disposition: form-data; name="%s"\r\n' % b'api_key')
    data.append(key.encode('utf-8'))
    data.append(b'--%s' % boundary.encode('utf-8'))
    data.append(b'Content-Disposition: form-data; name="%s"\r\n' % b'api_secret')
    data.append(secret.encode('utf-8'))
    data.append(b'--%s' % boundary.encode('utf-8'))
    fr = open(filepath, 'rb')
    data.append(b'Content-Disposition: form-data; name="%s"; filename="12263.jpg"' % b'image_file')
    data.append(b'Content-Type: %s\r\n' % b'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append(b'--%s--\r\n' % boundary.encode('utf-8'))

    # Join data as bytes
    http_body = b'\r\n'.join(data)

    # Build HTTP request
    req = urllib.request.Request(humanbody_http_url)

    # Header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
    req.data = http_body

    try:
        # Post data to server
        resp = urllib.request.urlopen(req, timeout=5)
        # Get response
        qrcont = resp.read()
        # Parse the JSON response
        parsed_response = json.loads(qrcont)

        # Check if human body detection was successful
        if len(parsed_response['humanbodies']) != 0:
            human_body_rectangle = parsed_response['humanbodies'][0]['humanbody_rectangle']

            # Extract and print the face rectangle details
            top = human_body_rectangle['top']
            left = human_body_rectangle['left']
            width = human_body_rectangle['width']
            height = human_body_rectangle['height']
            right = left + width
            bottom = top + height

            return [top, bottom, left, right]

        else:
            req = urllib.request.Request(face_http_url)

            # Header
            req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
            req.data = http_body
            # Post data to server
            resp = urllib.request.urlopen(req, timeout=5)
            # Get response
            qrcont = resp.read()
            # Parse the JSON response
            parsed_response = json.loads(qrcont)
            if len(parsed_response['faces']) != 0:
                face_rectangle = parsed_response['faces'][0]['face_rectangle']

                # Extract and print the face rectangle details
                top = face_rectangle['top']
                left = face_rectangle['left']
                width = face_rectangle['width']
                height = face_rectangle['height']
                right = left + width
                bottom = top + height

                return [top, bottom, left, right]
            else:
                return None

    except urllib.error.HTTPError as e:
        print(e.read())
        return None


with open('portrait_data.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    # Loop through the images in the original directory
    for filename in filename_list:
        image_path = os.path.join(img_folder, filename)

        # Check if the resized image already exists in the 'resized_training' directory
        resized_image_path = os.path.join("resized_training", filename)

        if os.path.exists(resized_image_path):
            print(f"Skipping {filename} - Already resized")
            continue

        # GET THE IMAGE HEIGHT AND WIDTH
        img_object = Image.open(image_path)
        w3, h3 = img_object.size
        img_object = img_object.convert('RGB')
        img_reshape = get_shape(img_object, w3, h3, 224, False)
        img_reshape.save(resized_image_path)
        print(f"Resized and saved: {resized_image_path}")

        image = np.asarray(img_reshape)
        h2, w2 = (image.shape[0] // 16 + 1) * 16, (image.shape[1] // 16 + 1) * 16

        # get values of bounding box
        bbox = face_plus_plus(resized_image_path)
        if bbox is None:
            print(f"Human body or face could not be detected. Deleted {resized_image_path}")
            os.remove(resized_image_path)
            continue
        else:
            # get the values of the offset
            o1, o2, o3, o4 = get_offset(w2 - 1, h2 - 1, bbox)
            portrait_data.append([filename, o1, o2, o3, o4])

            # Write the data to the CSV file immediately
            writer.writerow([filename, o1, o2, o3, o4])

            # run add_offset
            # print("add_offset", add_offset(w2 - 1, h2 - 1, bbox, get_offset(w2 - 1, h2 - 1, bbox)))
