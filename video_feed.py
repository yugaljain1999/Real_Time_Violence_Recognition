import cv2 
import onnxruntime as runonnx
import numpy as np
import ffmpeg_streaming
import time
def normalize_meanstd(a, axis=None): 
    # axis param denotes axes along which mean & std reductions are to be performed
    mean = np.mean(a, axis=axis, keepdims=True)
    std = np.sqrt(((a - mean)**2).mean(axis=axis, keepdims=True))
    return (a - mean) / std

def video_feeding():
    onnx_path = 'model_wts.onnx'
    font = cv2.FONT_HERSHEY_SIMPLEX 
    start_point = (80, 20) # Ending coordinate, here (220, 220)  
    end_point = (320, 60) # represents the bottom right corner of rectangle
    org = (100, 50)  
    fontScale = 1 # fontScale 
    color = (255, 0, 0) # Blue color in BGR
    thickness = 4 # Line thickness of 2 px 
    color_b = (0, 0, 0) # Blue color in BGR
    thickness_b = 5
    
    vidcap = cv2.VideoCapture(0)
    print(vidcap)
    count = 0
    frames=[]
    success=True
    pred=0.0
    prediction='NO VIOLENCE'
    # Load model #
    sess = runonnx.InferenceSession(onnx_path)
    input_name = sess.get_inputs()[0].name
    while success :
        success,frame= vidcap.read()
        if success==False:
            print('success-false')
        frame = cv2.resize(frame,(224,224), interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.reshape(frame, (224,224,3))
        image = cv2.resize(frame,(424,424), interpolation=cv2.INTER_AREA)
        image = cv2.rectangle(image, start_point, end_point, color_b, thickness_b) 
        image = cv2.putText(image, prediction , org, font, fontScale, color, thickness, cv2.LINE_AA)
        _,image = cv2.imencode('.jpg',image)
        frame_im = image.tobytes()
        #cv2.imshow("video stream",image )
        #cv2.waitKey(25)
        if count%3==0: #each 5 second of video contain 150 frames hence I am taking each 7th frame
            frames.append(frame)
        if len(frames)==20: #once I get 20 frames I giving data to model for prediction
            
            frames=np.array(frames)
            image=normalize_meanstd(frames,axis=(1,2))
            image=np.reshape(image,(1,20,224,224,3))
            start = time.time()
            pred= sess.run(None,{input_name:image.astype(np.float32)})
            end = time.time()
            print('violence_prob',pred)
            
            if pred[0]>0.4:
                prediction='VIOLENCE'
                color = (0, 0,255)
                org = (120, 50) 
            else:
                prediction='NO VIOLENCE'
                color = (255, 0, 0)
                org = (100, 50) 
            

            # total time taken
            print(f"prediction time {end - start}")
            frames=[]
        
        count=count+1
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_im + b'\r\n')
