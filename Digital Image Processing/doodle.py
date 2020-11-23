import cv2

def detect():
    face_cascade = cv2.CascadeClassifier('F:/Applied Informatics/Semester-III/Digital Image Processing/haarcascades/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)
    while (camera.isOpened()):
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

        roi_gray = gray[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 5, 0, (40,40))
        for (ex,ey,ew,eh) in eyes:
                    img = cv2.rectangle(img,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)
        
        cv2.imshow("camera", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
        else:
            continue
    
    camera.release()
    cv2.destroyAllWindows()

detect()
