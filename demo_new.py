import urllib.request
import time
import json
import os
import sys
import cv2
from utils import *
import numpy as np
from models import config
from models import model as M
from PIL import Image, ImageDraw
from modify_model import new_model
import tensorflow as tf

# Load configuration settings from the custom Config class
C = config.Config()


# Resize the input image while maintaining its aspect ratio
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


def face_plus_plus(filepath):
    # API configuration
    key = "ILUS28ltigP0UirSewQPZmKLCqFQEtg1"
    secret = "vMLb1ufvKfjIgIcLShervHy723o7cdLI"

    face_http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
    humanbody_http_url = 'https://api-us.faceplusplus.com/humanbodypp/v1/detect'

    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = [b'--%s' % boundary.encode('utf-8'), b'Content-Disposition: form-data; name="%s"\r\n' % b'api_key',
            key.encode('utf-8'), b'--%s' % boundary.encode('utf-8'),
            b'Content-Disposition: form-data; name="%s"\r\n' % b'api_secret', secret.encode('utf-8'),
            b'--%s' % boundary.encode('utf-8')]
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
            bottom = top + height
            right = left + width

            return [top, left, height, width]
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
                bottom = top + height
                right = left + width

                return [top, left, height, width]
            else:
                return None

    except urllib.error.HTTPError as e:
        print(e.read())
        return None


# Main function to process images
def run(images):
    if C.log:
        result = []
    if os.path.isdir(images):
        test_db = os.listdir(images)
        test_db = [os.path.join(images, i) for i in test_db]
    elif os.path.isfile(images) and images.endswith(C.pic_extend):
        test_db = [images]
    else:
        raise Exception('Image file or directory not exist.')
    for image_name in test_db:
        img_object = Image.open(image_name)
        # print("image_name", image_name)
        w3, h3 = img_object.size
        img_object = img_object.convert('RGB')
        img_reshape = get_shape(img_object, w3, h3, C.scale, C.ratio)

        if not os.path.exists("resized_testing"):
            os.makedirs("resized_testing")
        file_name = os.path.basename(image_name)
        img_reshape.save(os.path.join("resized_testing", "resized_" + file_name))

        image = np.asarray(img_reshape)
        h1, w1 = image.shape[0], image.shape[1]
        h2, w2 = (image.shape[0] // 16 + 1) * 16, (image.shape[1] // 16 + 1) * 16

        image = cv2.copyMakeBorder(image, top=0, bottom=h2 - h1, left=0, right=w2 - w1, borderType=cv2.BORDER_CONSTANT,
                                   value=0)

        # Normalize image pixel values to [0, 1]
        image = image.astype('float32') / 255.0
        # Add a batch dimension to the image
        image = np.expand_dims(image, axis=0)
        # Run face_plus_plus
        fpp_box = face_plus_plus(os.path.join("resized_testing", "resized_" + file_name))
        # fpp_box = [156, 55, 132, 96]
        fpp_input = tf.convert_to_tensor(fpp_box)
        fpp_input = tf.reshape(fpp_input, (1, 4))
        fpp_input = tf.cast(fpp_input, dtype='float32')
        fpp_input = fpp_input / 16.0
        # Predict bounding boxes using the model
        offset = new_model.predict([image, fpp_input], batch_size=1, verbose=0)
        offset = offset[0]
        print("offset:", offset)
        print("fpp_box:", fpp_box)
        fpp_box[2] = fpp_box[0] + fpp_box[2]
        fpp_box[3] = fpp_box[1] + fpp_box[3]
        print("fpp_box top, left, bottom, right:", fpp_box)
        # Rearrange fpp_box coordinates to [x1, x2, y1, y2] format
        fpp_box = [fpp_box[0], fpp_box[2], fpp_box[1], fpp_box[3]]  # x1, x2, y1, y2
        print("rearranged fpp_box:", fpp_box)
        # Convert offset to a NumPy array
        offset = np.array(offset)
        print("converted offset:", offset)
        # Normalize box coordinates
        fpp_box = normalization(w2 - 1, h2 - 1, fpp_box)
        print("normalized fpp_box:", fpp_box)
        # Add offset to box to get aes_bbox
        aes_bbox = add_offset(w2 - 1, h2 - 1, fpp_box, offset)
        print("aes_bbox breakdown:")
        print("w2:", w2)
        print("h2: ", h2)
        print("aes_bbox:", aes_bbox)

        img_name = image_name.split('\\')[-1]
        if C.log:
            to_file = ' '.join([img_name] + [str(u) for u in fpp_box] + [str(y) for y in aes_bbox])
            result.append(to_file)
        if C.draw:
            if not os.path.isdir(C.box_out_path):
                os.makedirs(C.box_out_path)
            aes_box = recover_from_normalization_with_order(w3 - 1, h3 - 1, aes_bbox)  # [int]
            fpp_box = recover_from_normalization_with_order(w3 - 1, h3 - 1, fpp_box)
            img_draw = img_object.copy()
            draw = ImageDraw.Draw(img_draw)
            draw.rectangle(fpp_box, None, C.saliency_box_color)
            draw.rectangle(aes_box, None, C.aesthetics_box_color)
            # print(os.path.join(C.box_out_path, img_name))
            img_draw.save(os.path.join(C.box_out_path, img_name))
        if C.crop:
            if not os.path.isdir(C.crop_out_path):
                os.makedirs(C.crop_out_path)
            aes_box = recover_from_normalization_with_order(w3 - 1, h3 - 1, aes_bbox)  # [int]
            img_crop = img_object.crop(aes_box)
            img_crop.save(os.path.join(C.crop_out_path, img_name))
    if C.log:
        if not os.path.isdir(C.log_path):
            os.mkdir(C.log_path)
        with open(C.log_file, 'w') as f:
            f.write('\n'.join(result))


def main():
    images = sys.argv[1]
    # model = M.EndToEndModel(gamma=C.gamma, theta=C.theta, stage='test', image_file=image_file).BuildModel()
    new_model.load_weights("final_6epochs.h5")
    run(images)


if __name__ == "__main__":
    sys.exit(main())
