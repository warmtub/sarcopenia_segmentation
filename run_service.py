#import flask
import boto3
import cv2
import numpy as np
import os
import torch
from botocore.utils import is_valid_endpoint_url
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError
from Image_Segmentation.evaluation import get_DC
from Image_Segmentation.network import U_Net, AttU_Net
from PIL import Image
from pydicom import dcmread
from sanic import Sanic
from sanic import response
from torchvision import transforms as T


# app = Flask(__name__)
app = Sanic(__name__)

n_classes = 12
print("loading segmentation model")
net = AttU_Net(img_ch=3, output_ch=n_classes).to('cuda')
net.load_state_dict(torch.load('Image_Segmentation/models/trial3(DC_exclude_background)/AttU_Net-f4-490-0.923.pkl'))
net.train(False)
print("done")

value_to_label_name = {}
with open("classes.txt") as f:
    for index, line in enumerate(f):
        value_to_label_name[index] = line[:-1]
print(value_to_label_name)


def download_s3_file(endpoint, access_key, secret_key, bucket, dicom_filepath, dicom_file):
    config = Config(signature_version='s3')
    is_verify = False
    connection = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        config=config,
        verify=is_verify
    )
    docker_dicom_filepath = dicom_file #os.path.join(docker_dicom_dir, dicom_file)

    try:
        connection.download_file(bucket, dicom_filepath, docker_dicom_filepath)
    except ClientError as e:
        pass

    if os.path.isfile(docker_dicom_filepath):
        return True
    else:
        return False

empty_response = {  "version": "4.5.6",
                    "flags": {},
                    "shapes": [],
                    "imagePath": "",
                    "imageData": None,
                    "imageHeight": 0,
                    "imageWidth": 0,
                    "shapes": [],
                    "message": "",
                 }

# curl -X POST http://192.168.211.14:5000/sarcopenia --header "Content-Type: application/json" --data '{"endpoint" : "https://s3.twcc.ai", "access_key" : "5QL09M2O1Y8E4GTOFC9Z", "secret_key" : "9mXMT1kJAYAzOGZusIc5CT856cc3O22FqaYZpeTN", "bucket" : "test-bucket-sarcopenia", "file" : "Sarcopenia_data/SARCOPANIA0010.dcm"}'
@app.route("/sarcopenia", methods=['POST'])
async def sarcopenia_inference(request):
    global n_classes, net, value_to_label_name
    
    """
    file_name = os.path.join("data", request.files["data"][0].name)
    with open(file_name, "wb") as f:
        f.write(request.files["data"][0].body)
        f.close()
    """
        
    request_dict = request.json
    endpoint = request_dict.get('endpoint', None)
    access_key = request_dict.get('access_key', None)
    secret_key = request_dict.get('secret_key', None)
    bucket = request_dict.get('bucket', None)
    dicom_filepath = request_dict.get('file', None)

    # Download file from S3 blob
    if endpoint and access_key and secret_key and bucket and dicom_filepath:
        dicom_file = dicom_filepath.split('/')[-1]
        if not download_s3_file(endpoint, access_key, secret_key, bucket, dicom_filepath, dicom_file):
            empty_response["message"] = "The file does not download from S3 blob"
            return response.json(empty_response)
    else:
        empty_response["message"] = "The API request information is not complete"
        return response.json(empty_response)
    
    ds = dcmread(dicom_file)
    image = ds.pixel_array
    image_h, image_w = image.shape
    image = image.astype('float64')
    intercept = ds.RescaleIntercept
    wc = 50
    ww = 250
    UL = wc + ww/2
    LL = wc - ww/2
    slope = ds.RescaleSlope
    image -= (-intercept+LL)
    image[image<0] = 0
    image[image>(UL-LL)] = UL-LL
    image *= 255.0/image.max()
    image = image.astype('uint8')
    image = Image.fromarray(image)
    image_src = image.convert('RGB')
    
    Transform = []
    Transform.append(T.ToTensor())
    Transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    Transform = T.Compose(Transform)
    image = Transform(image_src)
    image = image.unsqueeze(0).to('cuda')

    pred = torch.sigmoid(net(image))
    pred = pred.squeeze(0).detach().cpu()
    
    #GT_path = image_path.split('.dcm')[0] + '_1.json'
    #GT = Image.open(GT_path)
    #json_data = json.load(open(GT_path))
    #gt_label = utils.shapes_to_label(np.array(image_src).shape, json_data['shapes'], label_name_to_value)
    #gt_img = utils.draw_label(gt_label, np.array(image_src))

    #gt_label = torch.tensor(np.array(gt_label), dtype=torch.int64)
    #gt_label = torch.nn.functional.one_hot(gt_label, n_classes).to(torch.float).permute(2,0,1)
    #print(f'Dice score {get_DC(pred, gt_label)}')
    pred = np.argmax(pred, axis=0)
    
    shapes = []
    for i in range(1, n_classes):
        mask = pred[pred==i]
        layer = np.array(pred==i, np.uint8)*255 
        contours, _ = cv2.findContours(layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour = cv2.approxPolyDP(contour, 3, True).reshape(-1, 2).tolist()
            if len(contour) < 3: continue
            shape_data = {"label": value_to_label_name[i],
                          "points": contour,
                          "group_id": None,
                          "shape_type": "polygon",
                          "flags": {},
                         }
            shapes.append(shape_data)
        #plt.show()
        
    empty_response["version"] = "4.5.6"
    empty_response["imagePath"] = dicom_file.split('.')[0]+".png"
    empty_response["imageHeight"] = image_h
    empty_response["imageWidth"] = image_w
    empty_response["shapes"] = shapes
    
    return response.json(empty_response)



if __name__ == "__main__":
    app.run(host = '192.168.211.14', port = 5000)

