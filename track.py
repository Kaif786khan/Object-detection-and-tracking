import ultralytics 
import cv2
from ultralytics import YOLO
import numpy 
from deep_sort.deep_sort  import DeepSort


feature_extractor  = "osnet_x0_25"
deepsort = DeepSort(model_type=feature_extractor)


model = YOLO('yolov5n.pt') #if u want a yolov8 model make the changes in the weight file thats it! .

class_names = model.names
print(type(class_names))

#if you want the code to track from video 
video_path = "walk.mp4"

#insert '0' in place of video_path if you want to track form webcam itself.
cap = cv2.VideoCapture(video_path)

while cap.isOpened:
    ret , frame = cap.read()
    if frame is None :
        print("not reading ")
        break
    result = model.predict(frame)
    for i in result:
        boxes = (i.boxes.xywh).numpy()
        conf = (i.boxes.conf).numpy()
        classes_ = (i.boxes.cls).numpy()

        #we got our inputs for the deepsort algo 
        outputs = deepsort.update(bbox_xywh =  boxes,
                                    confidences = conf,
                                    classes = classes_,
                                    ori_img =  frame,)
                                    
        #you can actually see the outputs if you want to , just uncomment the below commented 		#lines !                            
        # print(outputs)
        # print(type(outputs))
        #here we are getting the output as [bbox(xyxy) , track id , class ]
        
  
        #now we will itirate the outputs and get the desired things to get our visualization 
        for r in outputs:
            x1 , y1 , x2 , y2 = r[:4]
            track_id = r[4]
            class_id = r[-1]
            class_name = class_names[class_id] if class_id in class_names else "Unknown"

            frame = cv2.rectangle(frame , (x1,y1) , (x2 , y2) , (0,255,0) , 1)
            text = f"track id is {track_id} , class id is {class_id},{class_name}"
            text_position  = (x1,y1-10)

            #add text to tghe frame 
            frame = cv2.putText(frame , text , text_position , cv2.FONT_HERSHEY_COMPLEX , 0.5 ,(0,255,0) , 2 )
        cv2.imshow("frame",frame)
        print("error message")
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()




