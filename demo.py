import warnings

with warnings.catch_warnings():
    import os, sys
    import cv2
    from utils import *
    import numpy as np
    from models import config
    from models import model as M
    from PIL import Image, ImageDraw

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


# Main function to process images
def run(model, images):
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

        if not image_name.endswith(C.pic_extend):
            continue
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
        # Predict bounding boxes using the model
        boxes = model.predict(image, batch_size=1, verbose=0)
        print("boxes:", boxes)
        # Extract offset and saliency_box from the model predictions (boxes)
        offset = boxes[0][0]
        saliency_box = boxes[1][0]
        # print("saliency_box:", saliency_box)
        # Scale the saliency_box to match the image dimensions (multiple of 16)
        saliency_box = saliency_box * 16.0
        # print("saliency_box * 16:", saliency_box)
        saliency_box[2] = saliency_box[0] + saliency_box[2]
        saliency_box[3] = saliency_box[1] + saliency_box[3]
        # Convert the saliency_box coordinates to integers
        saliency_box = [int(y) for y in saliency_box]
        # Rearrange saliency_box coordinates to [x1, x2, y1, y2] format
        saliency_box = [saliency_box[0], saliency_box[2], saliency_box[1], saliency_box[3]]  # x1, x2, y1, y2
        # Convert offset to a NumPy array
        offset = np.array(offset)
        # Normalize face_box coordinates
        saliency_box = normalization(w2 - 1, h2 - 1, saliency_box)
        # Add offset to face_box to get aes_bbox
        aes_bbox = add_offset(w2 - 1, h2 - 1, saliency_box, offset)

        img_name = image_name.split('\\')[-1]
        if C.log:
            to_file = ' '.join([img_name] + [str(u) for u in saliency_box] + [str(y) for y in aes_bbox])
            result.append(to_file)
        if C.draw:
            if not os.path.isdir(C.box_out_path):
                os.makedirs(C.box_out_path)
            aes_box = recover_from_normalization_with_order(w3 - 1, h3 - 1, aes_bbox)  # [int]
            saliency_box = recover_from_normalization_with_order(w3 - 1, h3 - 1, saliency_box)
            img_draw = img_object.copy()
            draw = ImageDraw.Draw(img_draw)
            draw.rectangle(saliency_box, None, C.saliency_box_color)
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


# Main function to run the script
def main(argv=None):
    if len(sys.argv) <= 1:
        images = C.image_path
    else:
        images = sys.argv[1]
    model = M.EndToEndModel(gamma=C.gamma, theta=C.theta, stage='test').BuildModel()
    model.load_weights(C.model)
    run(model, images)


if __name__ == "__main__":
    sys.exit(main())
