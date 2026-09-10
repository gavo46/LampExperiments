import cv2 
class Camera: 
    def __init__(self, camera_id=0): 
        self.cap = cv2.VideoCapture(camera_id) 

    def read(self): 
        success, frame = self.cap.read() 
        if not success: 
            return None 
        return frame 
    
    def release(self): 
        self.cap.release()