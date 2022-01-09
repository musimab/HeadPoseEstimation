import numpy as np
import cv2


def ref3DModel():
    modelPoints = [[0.0, 0.0, 0.0],
                   [0.0, -330.0, -65.0],
                   [-225.0, 170.0, -135.0],
                   [225.0, 170.0, -135.0],
                   [-150.0, -150.0, -125.0],
                   [150.0, -150.0, -125.0]]
    return np.array(modelPoints, dtype=np.float32)

def ref2DModel(landmarks_np):
    ref2dFacePoints  = np.array([landmarks_np[30],landmarks_np[8],
                                landmarks_np[36],landmarks_np[45],
                                landmarks_np[48],landmarks_np[54]], dtype=np.float32)
    return ref2dFacePoints


def findEularAngles(rvec):
    RotMatrix = cv2.Rodrigues(rvec)[0]
    
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(RotMatrix)

    #print("X: ", angles[0])
    #print("Y: ", angles[1])
    #print("Z: ", angles[2])
    
    if angles[1] < -15:
        print("Looking: Left")
    elif angles[1] > 15:
        print("Looking: Right")
    else:
        print("Forward")

def displayLandMarks(landmarks_np, frame, display= False):
    if(display):
        # Display the landmarks
        for i, (x, y) in enumerate(landmarks_np):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
           