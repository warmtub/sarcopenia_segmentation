#import flask
import asyncio
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
from sanic import Sanic, response
from torchvision import transforms as T

def download_s3_file(endpoint, access_key, secret_key, bucket, dicom_file_path, download_dir_path):
    # Download file from S3 blob
    config = Config(signature_version="s3")
    if endpoint and access_key and secret_key and bucket and dicom_file_path:
        is_verify = False
        connection = boto3.client(
            "s3",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            endpoint_url=endpoint,
            config=config,
            verify=is_verify
        )
        
        dicom_file = dicom_file_path.split("/")[-1]
        download_file_path = os.path.join(download_dir_path, dicom_file) #os.path.join(docker_dicom_dir, dicom_file)

        try:
            connection.download_file(bucket, dicom_file_path, download_file_path)
        except ClientError as e:
            pass

        if os.path.isfile(download_file_path):
            return True, ""
        else:
            error_message = "The file does not download from S3 blob"
            print("The file does not download from S3 blob")
            return False, error_message
    else:
        error_message = "The API request information is not complete"
        print("The API request information is not complete")
        return False, error_message

class Sarcopenia_Inference_APP(object):
    def __init__(self, address='192.168.211.14', port=5000, download_path="download"):
        # Sanic params
        self.address = address
        self.port = port
        self.app = Sanic(__name__)
        self.app.add_route(self.inference_from_client, "/sarcopenia", methods=['POST'])
        
        # Model
        self.n_classes = 12
        self.net = AttU_Net(img_ch=3, output_ch=self.n_classes).to("cuda")
        self.model_path = "Image_Segmentation/models/trial3(DC_exclude_background)/AttU_Net-f4-490-0.923.pkl"
        print("loading segmentation model")
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.train(False)
        print("done")
        
        self.value_to_label_name = {}
        self.classes_path = "classes.txt"
        with open(self.classes_path) as f:
            for index, line in enumerate(f):
                self.value_to_label_name[index] = line[:-1]
        
        # Response
        self.empty_response = {"version": "4.5.6",
                               "flags": {},
                               "shapes": [],
                               "imagePath": "",
                               "imageData": None,
                               "imageHeight": 0,
                               "imageWidth": 0,
                               "shapes": [],
                               "message": "",
                              }
        
        # Download path
        self.download_path = download_path
        if not os.path.exists(download_path):
            os.mkdir(self.download_path)
        
    def inference_from_client(self, request):
        inference_response = self.empty_response.copy()
        
        request_dict = request.json
        endpoint = request_dict.get("endpoint", None)
        access_key = request_dict.get("access_key", None)
        secret_key = request_dict.get("secret_key", None)
        bucket = request_dict.get("bucket", None)
        dicom_file_path = request_dict.get("file", None)
        
        download, error_message = download_s3_file(endpoint, access_key, secret_key, bucket, dicom_file_path, self.download_path)
        if not download:
            inference_response["message"] = error_message
            return response.json(inference_response)
        dicom_file_path = os.path.join(self.download_path, dicom_file_path.split("/")[-1])
        
        return response.json(self.inference(inference_response, dicom_file_path))
    
    def inference_from_list(self, file_list, endpoint=None, access_key=None, secret_key=None, bucket=None):
        inference_response_list = [self.empty_response.copy() for _ in range(len(file_list))]
        download_file_list = ["" for _ in range(len(file_list))]
        #for dicom_file_path, inference_response in zip(file_list, inference_response_list):
        for idx in range(len(file_list)):
            # Download file from S3 blob
            download, error_message = download_s3_file(endpoint, access_key, secret_key, bucket, file_list[idx], self.download_path)
            if not download:
                inference_response_list[idx]["message"] = error_message
                continue
            download_file_list[idx] = os.path.join(self.download_path, file_list[idx].split("/")[-1])
            inference_response_list[idx] = self.inference(inference_response_list[idx], download_file_list[idx])
        
        print(download_file_list)
        return download_file_list, inference_response_list
    
    def inference(self, inference_response, dicom_file):
        ds = dcmread(dicom_file)
        image = ds.pixel_array
        image_h, image_w = image.shape
        image = image.astype("float64")
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
        image = image.astype("uint8")
        image = Image.fromarray(image)
        image_src = image.convert("RGB")

        Transform = []
        Transform.append(T.ToTensor())
        Transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        Transform = T.Compose(Transform)
        image = Transform(image_src)
        image = image.unsqueeze(0).to("cuda")

        pred = torch.sigmoid(self.net(image))
        pred = pred.squeeze(0).detach().cpu()

        #GT_path = image_path.split(".dcm")[0] + "_1.json"
        #GT = Image.open(GT_path)
        #json_data = json.load(open(GT_path))
        #gt_label = utils.shapes_to_label(np.array(image_src).shape, json_data["shapes"], label_name_to_value)
        #gt_img = utils.draw_label(gt_label, np.array(image_src))

        #gt_label = torch.tensor(np.array(gt_label), dtype=torch.int64)
        #gt_label = torch.nn.functional.one_hot(gt_label, n_classes).to(torch.float).permute(2,0,1)
        #print(f"Dice score {get_DC(pred, gt_label)}")
        pred = np.argmax(pred, axis=0)

        shapes = []
        for i in range(1, self.n_classes):
            mask = pred[pred==i]
            layer = np.array(pred==i, np.uint8)*255 
            contours, _ = cv2.findContours(layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour = cv2.approxPolyDP(contour, 3, True).reshape(-1, 2).tolist()
                if len(contour) < 3: continue
                shape_data = {"label": self.value_to_label_name[i],
                              "points": contour,
                              "group_id": None,
                              "shape_type": "polygon",
                              "flags": {},
                             }
                shapes.append(shape_data)

        inference_response["version"] = "4.5.6"
        inference_response["imagePath"] = dicom_file.split(".")[0]+".png"
        inference_response["imageHeight"] = image_h
        inference_response["imageWidth"] = image_w
        inference_response["shapes"] = shapes

        return inference_response
        
    def start(self):
        self.app.run(host=self.address, port=self.port)



# curl -X POST http://192.168.211.14:5000/sarcopenia --header "Content-Type: application/json" --data '{"endpoint" : "https://s3.twcc.ai", "access_key" : "5QL09M2O1Y8E4GTOFC9Z", "secret_key" : "9mXMT1kJAYAzOGZusIc5CT856cc3O22FqaYZpeTN", "bucket" : "test-bucket-sarcopenia", "file" : "Sarcopenia_data/SARCOPANIA0010.dcm"}'
if __name__ == "__main__":
    app = Sarcopenia_Inference_APP()
    app.start()