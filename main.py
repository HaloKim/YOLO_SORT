import cv2
import numpy as np
import time
import pandas as pd
from sort import Sort

#Load YOLO
net = cv2.dnn.readNet("yolov3.weights","yolov3.cfg") # Original yolov3

#net = cv2.dnn.readNet("yolov3-tiny.weights","yolov3-tiny.cfg") #Tiny Yolo

classes = []

with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

# loading image
cap = cv2.VideoCapture(0)  # 0 for 1st webcam

# cap = cv2.VideoCapture("/Users/halo/Documents/sort-master/People\ -\ 6387.mp4")

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
score = 0

# Dataframe
df1 = pd.DataFrame(columns=("time","label","index","confidences","x","y","w","h"))

# coordinate to SORT bbox(x,y,s,r) -> x1,y1,x2,y2
def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  h = x[2] / w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

# SORT Init
mot_tracker = []
mot_tracker.append(Sort())

def add_SORT(obj):
    if len(mot_tracker) < obj:
        for i in range(obj):
            mot_tracker.append(Sort())
    return len(mot_tracker)

# Loop the Frame
while True:
    ret, frame = cap.read()
    frame_id += 1

    height, width, channels = frame.shape

    # detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # reduce 416 to 320

    net.setInput(blob)
    outs = net.forward(outputlayers)
    # print(outs[1])

    # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids = []
    confidences = []
    boxes = []

    # SORT variables
    sortxy = []  # x,y,s,r save
    arg = [] # x,y,s,r save
    xtb = [] # x to bbox
    trackers = [] # SORT return values
    sortboxes = [] # SORT rectangles coordinate

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Check confidence and Check class labels
            if confidence > 0.3 and (str(classes[class_id]) == "bottle" or str(classes[class_id]) == "person"
            or str(classes[class_id]) == "cell phone"):
            #if confidence > 0.3:
                # object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                # rectangle coordinaters
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])  # put all rectangle areas
                confidences.append(float(confidence))

                # how confidence was that object detected and show that percentage
                class_ids.append(class_id)  # name of the object tha was detected
                sortxy.append([center_x, center_y, (w * h), (w / float(h))])
                print(add_SORT(len(boxes)))

    # SORT Class
    for i in range(len(sortxy)):
        arg = sortxy[i][0], sortxy[i][1], sortxy[i][2], sortxy[i][3]
        xtb = convert_x_to_bbox(arg)
        trackers = mot_tracker[i].update(np.array(xtb))
        sortboxes.append(trackers)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    label_lists = [] # Label Lists

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            label_lists.insert(i,label)
            confidence = confidences[i]
            color = colors[class_ids[i]]

            # Draw Rectangles
            for d in sortboxes[i]:
                d = d.astype(np.int32) # x,y,r,s

                # YOLO Rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x + 100, y + 100), font, 1, (0, 0, 0), 2)

                # SORT Rectangle
                cv2.rectangle(frame, (d[0], d[1]), (d[2]+20, d[3]+20), (255, 255, 153), 2)
                cv2.putText(frame, label + str(d[4]), (d[0]+((d[2]-d[0])/2), d[1]+((d[3]-d[1])/2)), font, 1, (255, 0, 255), 2)

                # Time stamp
                # timetmp = time.strftime('%Y-%m-%d', time.localtime(time.time()))
                timetmp = time.strftime('%c', time.localtime(time.time()))

                # Make csv data
                tmp = pd.Series \
                    ([timetmp, label, d[4], confidences[i], x, y, w, h],
                     index=["time", "label", "index", "confidences", "x", "y", "w", "h"])
                print(tmp)
                df1 = df1.append(tmp, ignore_index=True)

    # Write FPS value
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    cv2.putText(frame, "FPS:" + str(round(fps, 2)), (10, 50), font, 2, (255, 255, 153), 3) # FPS

    # Show window
    frame = cv2.resize(frame,(640,480)) # resize frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1)  # wait 1ms the loop will start again and we will process the next frame

    if key == 27:  # esc key stops the process
        df1.to_csv('out.csv',index='true',mode='w')
        break;

cap.release()
cv2.destroyAllWindows()