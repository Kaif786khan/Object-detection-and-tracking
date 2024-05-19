import cv2
from ultralytics import YOLO  # Assuming you have a YOLO class defined

# Load the YOLO model
model = YOLO('yolov8n.pt')

# Open the webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if frame is None:
        print("Frame is not reading")
    else:
        # Predict objects in the frame
        result = model.predict(frame)

        for i in result:
            boxes = i.boxes[0].xyxy
            for box in boxes:
                left, top, right, bottom = map(int, box)
                class_id = int(i.boxes.cls[0].numpy())
                class_name = model.names[class_id]
                confidence = float(i.boxes.conf[0].numpy())

                # Draw bounding box
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Display class name and confidence
                cv2.putText(frame, f"{class_name} ({confidence:.2f})", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
