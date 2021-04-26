import asyncio
import base64
import cv2
import io
import logging
import numpy as np
import os
import torch
from Image_Segmentation.network import AttU_Net
from PIL import Image
from pydicom import dcmread
from sanic import Sanic, response
from sanic.exceptions import SanicException
from torchvision import transforms as T

logging.basicConfig(level=logging.INFO)

class Sarcopenia_Inference_APP(object):
    def __init__(self,
                 activate_server=True,
                 address="192.168.211.14",
                 port=5000,
                 download_path="download",
                 model_path="Image_Segmentation/models/trial3(DC_exclude_background)/AttU_Net-f4-490-0.923.pkl",
                 classes_path="classes.txt",
                ):
        # Sanic params
        self.activate_server = activate_server
        if self.activate_server:
            self.address = address
            self.port = port
            try:
                self.app = Sanic(__name__)
                self.app.add_route(self.inference_from_client, "/sarcopenia", methods=['POST'])
            except SanicException:
                self.app = Sanic.get_app(__name__)
        
        # Model
        self.n_classes = 12
        self.net = AttU_Net(img_ch=3, output_ch=self.n_classes).to('cuda')
        self.model_path = model_path
        logging.info("loading segmentation model")
        self.net.load_state_dict(torch.load(self.model_path))
        self.net.train(False)
        logging.info("done")
        
        self.value_to_label_name = {}
        self.classes_path = classes_path
        with open(self.classes_path) as f:
            for index, line in enumerate(f):
                self.value_to_label_name[index] = line[:-1]
        
        # Response
        self.empty_response = {'version': '4.5.6',
                               'flags': {},
                               'shapes': [],
                               'imagePath': '',
                               'imageData': None,
                               'imageHeight': 0,
                               'imageWidth': 0,
                               'shapes': [],
                               'message': '',
                              }
        
        # Download path
        self.download_path = download_path
        if not os.path.exists(download_path):
            os.mkdir(self.download_path)
        
    #handler for web server and get request dicom file from S3 blob
    def inference_from_client(self, request):
        inference_response = self.empty_response.copy()
        
        request_dict = request.json
        endpoint = request_dict.get('endpoint', None)
        access_key = request_dict.get('access_key', None)
        secret_key = request_dict.get('secret_key', None)
        bucket = request_dict.get('bucket', None)
        dicom_file_path = request_dict.get('file', None)
        
        download, error_message = download_s3_file(endpoint, access_key, secret_key, bucket, dicom_file_path, self.download_path)
        if not download:
            inference_response['message'] = error_message
            return response.json(inference_response)
        dicom_file_path = os.path.join(self.download_path, dicom_file_path.split('/')[-1])
        
        return response.json(self.inference(dicom_file_path, inference_response))
    
    #get list of dicom files from S3 blob
    def inference_from_list(self, file_list, endpoint=None, access_key=None, secret_key=None, bucket=None):
        download_file_list = ["" for _ in range(len(file_list))]
        inference_response_list = [self.empty_response.copy() for _ in range(len(file_list))]
        
        for idx in range(len(file_list)):
            download, error_message = download_s3_file(endpoint, access_key, secret_key, bucket, file_list[idx], self.download_path)
            if not download:
                inference_response_list[idx]['message'] = error_message
                continue
            download_file_list[idx] = os.path.join(self.download_path, file_list[idx].split('/')[-1])
            inference_response_list[idx] = self.inference(download_file_list[idx], inference_response_list[idx])
        
        return download_file_list, inference_response_list
    
    #inference single dicom file and generate json format 
    def inference(self, dicom_file, inference_response=None):
        if inference_response is None:
            inference_response = self.empty_response
        ds = dcmread(dicom_file)
        image = ds.pixel_array
        image_h, image_w = image.shape
        image = image.astype('float64')
        
        #CT window
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

        #predict
        Transform = []
        Transform.append(T.ToTensor())
        Transform.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        Transform = T.Compose(Transform)
        image = Transform(image_src)
        image = image.unsqueeze(0).to('cuda')

        pred = torch.sigmoid(self.net(image))
        pred = pred.squeeze(0).detach().cpu()
        pred = np.argmax(pred, axis=0)

        #generate json format
        shapes = []
        for i in range(1, self.n_classes):
            mask = pred[pred==i]
            layer = np.array(pred==i, np.uint8)*255 
            contours, _ = cv2.findContours(layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                #decrease this param if you need higher accuracy
                epsilon = 3
                contour = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2).tolist()
                if len(contour) < 3: continue
                shape_data = {'label': self.value_to_label_name[i],
                              'points': contour,
                              'group_id': None,
                              'shape_type': 'polygon',
                              'flags': {},
                             }
                shapes.append(shape_data)

        inference_response['version'] = '4.5.6'
        file_name = '.'.join(dicom_file.split('/')[-1].split('.')[:-1])
        inference_response['imagePath'] = file_name
        imageData = io.BytesIO()
        image_src.save(imageData, format='png')
        imageData = imageData.getvalue()
        imageData = base64.b64encode(imageData).decode("utf-8")
        inference_response['imageData'] = imageData
        inference_response['imageHeight'] = image_h
        inference_response['imageWidth'] = image_w
        inference_response['shapes'] = shapes

        logging.info(f"The json format for {dicom_file} is generated")
        
        return inference_response
        
    #start web server
    def start(self):
        if self.activate_server:
            self.app.run(host=self.address, port=self.port)
        else:
            logging.error(f"Sanic server is not activated")



# curl -X POST http://192.168.211.14:5000/sarcopenia --header "Content-Type: application/json" --data '{"endpoint" : "https://s3.twcc.ai", "access_key" : "5QL09M2O1Y8E4GTOFC9Z", "secret_key" : "9mXMT1kJAYAzOGZusIc5CT856cc3O22FqaYZpeTN", "bucket" : "test-bucket-sarcopenia", "file" : "Sarcopenia_data/SARCOPANIA0010.dcm"}'
if __name__ == '__main__':
    app = Sarcopenia_Inference_APP()
    app.start()
