from ultralytics import YOLO
import torch
#code to check if the cuda si available or not 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"you are using the device : {device}")

#load the base model 
model = YOLO('yolov8n.pt')

#path to your data.yaml
data_path = '/home/examroom/Downloads/open_closed.v2i.yolov8/data.yaml'
results = model.train(data= data_path, epochs = 120 , imgsz = 64, device = device)
