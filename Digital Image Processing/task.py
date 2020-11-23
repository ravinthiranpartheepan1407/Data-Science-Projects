
import cv2
import timeit

def detect(filename):
    face_cascade = cv2.CascadeClassifier('F:/Applied Informatics/Semester-III/Digital Image Processing/haarcascades/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('F:/Applied Informatics/Semester-III/Digital Image Processing/haarcascades/haarcascades/haarcascade_eye.xml')
    
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image, scaleFactor, minNeighbors, minSize, maxSize
    faces = face_cascade.detectMultiScale(gray, 1.3, 3) # 1.3, 6
    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.45, 6, 0, (30,30))
    for (ex,ey,ew,eh) in eyes:
      for (ex,ey,ew,eh) in eyes:
        img = cv2.rectangle(img,(x+ex,y+ey),(x+ex+ew,y+ey+eh),(0,255,0),2)
    
    while(True):
        cv2.namedWindow('Faces')
        cv2.imshow('Faces', img)
        
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
            break
        elif k & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        else:
            continue
    cv2.imwrite('F:/Applied Informatics/Semester-III/Digital Image Processing/fourth_face.jpg', img)

filename = 'F:/Applied Informatics/Semester-III/Digital Image Processing/ron.jpg'
start = timeit.timeit()
detect(filename)

end = timeit.timeit()
print("%.4f s" % (end - start))


