import math
import cv2
import numpy as np
import dlib
import pyrealsense2 as rs
from matplotlib import pyplot as plt
from utils import ref3DModel, ref2DModel
from utils import findEularAngles,displayLandMarks
import argparse


def initRealsenseSettings(IMG_WIDTH, IMG_HEIGHT):
    # Pointcloud persistency in case of dropped frames
    pc = rs.pointcloud()
    points = rs.points()
    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()

    # This is the minimal recommended resolution for D435
    config.enable_stream(rs.stream.depth,IMG_WIDTH, IMG_HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color,IMG_WIDTH, IMG_HEIGHT, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # Create an align object
    align_to = rs.stream.color

    align = rs.align(align_to)
    return pipeline, align


def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    K = [[intrinsics.fx, 0.0000, intrinsics.ppx],
        [0.0000, intrinsics.fy, intrinsics.ppy],
        [0.0000, 0.0000,        1.00000]]
    distCoef = np.array([0.0,0.0,0.0,0.0])
    np.save("CalibrationMatrix", K)
    #print("Intrinsics:", K)
    return np.array(K)


def find3DprojectionTo2D(points_3D, rvec, tvec, camMatrix):
    RotMatrix, _ = cv2.Rodrigues(rvec)
    #cameraPose = -RotMatrix.T * tvec

    Rt = np.concatenate([RotMatrix, tvec.T], axis = -1)
 
    points_3D = np.hstack((points_3D,np.ones(1).reshape(1,1)))
    ProjectionMatrix = np.matmul(camMatrix, Rt)

    #print("ProjectionM size:", ProjectionMatrix.shape)
    pointProjected2D = (np.matmul(ProjectionMatrix, points_3D.reshape(4,1)))

    p2D = pointProjected2D/pointProjected2D[2]
    points2D = int(p2D[0][0]),int(p2D[1][0])
    return points2D


def main(cameraMatrix, IMG_SIZE):

    IMG_WIDTH, IMG_HEIGHT = IMG_SIZE
    pipeline, align = initRealsenseSettings(IMG_WIDTH, IMG_HEIGHT)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    distCoef = np.zeros((1,4), np.float32())
    #cameraMatrix = np.array([[608.1615600585938, 0.0, 327.25433349609375], 
    #                        [0.0, 608.27685546875, 244.0302734375], 
    #                       [0.0, 0.0, 1.0]])
    frameNumber = 0
    
    while True:

        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_frame or not color_frame:
            continue

        #Save Calibration matrix for realsense camera
        if(frameNumber == 0):
            cameraMatrix = get_intrinsic_matrix(color_frame)
        frameNumber+=1
        
        frame = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = detector(gray)
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            landmarks = predictor(gray, face)
            landmarks_np = np.zeros((68, 2), dtype="int")
            
            #find 68 landmarks points in a face
            for i in range(0, 68):
                landmarks_np[i] = (landmarks.part(i).x,landmarks.part(i).y)
            
            #it will show all detected landmarks in a face
            displayLandMarks(landmarks_np,frame,False)
            #Corrosponding 2d->3d face points
            ref2dFacePts = ref2DModel(landmarks_np)
            ref3dFacePts = ref3DModel()
            
            # calculate rotation and translation vector using solvePnP
            success, rotationVector, translationVector = cv2.solvePnP(ref3dFacePts, ref2dFacePts, 
                                                                        cameraMatrix, distCoef)
        
            noseEndPoints3D = np.array([[0, 0, 1000.0]], dtype=np.float64)
            noseEndPoint2D, jacobian = cv2.projectPoints(noseEndPoints3D, rotationVector, 
                                                    translationVector, cameraMatrix, distCoef)
            #print("cv_projection:", noseEndPoint2D)
          
            NosePoints2D = find3DprojectionTo2D(noseEndPoints3D, rotationVector.reshape(1,3), translationVector.reshape(1,3), cameraMatrix)
            #print("my_projection:", NosePoints2D)
            
            #calculate yaw, pitch and roll angles
            findEularAngles(rotationVector)
            cv2.circle(frame, NosePoints2D, 5, (255,0,255), 5)
            
            p1 = (int(ref2dFacePts[0, 0]), int(ref2dFacePts[0, 1]))
            p2 = (int(noseEndPoint2D[0, 0, 0]), int(noseEndPoint2D[0, 0, 1]))
            cv2.line(frame, p1, p2, (110, 220, 0), thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-K_Matrix", "--IntrinsicMatrix",
                        required=True,
                        help="Path to calibration matrix (numpy file)")
    parser.add_argument("-ImgSize", "--img_width_height", type=int, default=(640,480),
        help="Image shape info for auto calibration")
    args = vars(parser.parse_args())

    calibration_matrix_path = args["IntrinsicMatrix"]
    IMG_SIZE = args["img_width_height"]
    args = vars(parser.parse_args())
    
    CamMatrix = np.load(calibration_matrix_path)
    
    main(CamMatrix, IMG_SIZE)


