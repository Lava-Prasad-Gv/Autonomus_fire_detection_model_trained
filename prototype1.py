import cv2 
from ultralytics import YOLO


cam = cv2.VideoCapture('http://192.168.17.114/stream')


model  = YOLO('fire.pt')
while(True):
    _,img = cam.read()
    
    results = model(img, conf=0.5)
    # Draw bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]
            c = box.cls
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = model.names[int(c)]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                img,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
    cv2.imshow('feed',img)

    if _ ==False:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cam.release()


