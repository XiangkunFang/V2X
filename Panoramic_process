import cv2
import math
import os
import numpy as np
import torch
#Image undistort to one direction
def pano_undistortion(img, theta, phi, res_x, res_y, fov):

    img_x = img.shape[0]
    img_y = img.shape[1]

    theta = theta / 180 * math.pi
    phi = phi / 180 * math.pi

    fov_x = fov
    aspect_ratio = res_y * 1.0 / res_x
    half_len_x = math.tan(fov_x / 180 * math.pi / 2)
    half_len_y = aspect_ratio * half_len_x

    pixel_len_x = 2 * half_len_x / res_x
    pixel_len_y = 2 * half_len_y / res_y

    map_x = np.zeros((res_x, res_y), dtype=np.float32)
    map_y = np.zeros((res_x, res_y), dtype=np.float32)

    axis_y = math.cos(theta)
    axis_z = math.sin(theta)
    axis_x = 0

    # theta rotation matrix
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    theta_rot_mat = np.array([[1, 0, 0], \
            [0, cos_theta, -sin_theta], \
            [0, sin_theta, cos_theta]], dtype=np.float32)

    # phi rotation matrix
    cos_phi = math.cos(phi)
    sin_phi = -math.sin(phi)
    phi_rot_mat = np.array([[cos_phi + axis_x**2 * (1 - cos_phi), \
            axis_x * axis_y * (1 - cos_phi) - axis_z * sin_phi, \
            axis_x * axis_z * (1 - cos_phi) + axis_y * sin_phi], \
            [axis_y * axis_x * (1 - cos_phi) + axis_z * sin_phi, \
            cos_phi + axis_y**2 * (1 - cos_phi), \
            axis_y * axis_z * (1 - cos_phi) - axis_x * sin_phi], \
            [axis_z * axis_x * (1 - cos_phi) - axis_y * sin_phi, \
            axis_z * axis_y * (1 - cos_phi) + axis_x * sin_phi, \
            cos_phi + axis_z**2 * (1 - cos_phi)]], dtype=np.float32)

    map_x = np.tile(np.array(np.arange(res_x), dtype=np.float32), (res_y, 1)).T
    map_y = np.tile(np.array(np.arange(res_y), dtype=np.float32), (res_x, 1))

    map_x = map_x * pixel_len_x + pixel_len_x / 2 - half_len_x
    map_y = map_y * pixel_len_y + pixel_len_y / 2 - half_len_y
    map_z = np.ones((res_x, res_y)).astype(np.float32) * -1

    ind = np.reshape(np.concatenate((np.expand_dims(map_x, 2), np.expand_dims(map_y, 2), \
            np.expand_dims(map_z, 2)), axis=2), [-1, 3]).T

    ind = theta_rot_mat.dot(ind)
    ind = phi_rot_mat.dot(ind)

    vec_len = np.sqrt(np.sum(ind**2, axis=0))
    ind /= np.tile(vec_len, (3, 1))

    cur_phi = np.arcsin(ind[0, :])
    cur_theta = np.arctan2(ind[1, :], -ind[2, :])

    map_x = (cur_phi + math.pi/2) / math.pi * img_x
    map_y = cur_theta % (2 * math.pi) / (2 * math.pi) * img_y

    map_x = np.reshape(map_x, [res_x, res_y])
    map_y = np.reshape(map_y, [res_x, res_y])

    return cv2.remap(img, map_y, map_x, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

#All direction convert
def pano_process(frame, horizontal_angle=[-60,0,60,-120,-180,120], vertical_angle=0, res_x=1024, res_y=2048, fov=60.0): 
    back_left = pano_undistortion(frame, theta=horizontal_angle[0], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    back = pano_undistortion(frame, theta=horizontal_angle[1], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    back_right = pano_undistortion(frame,theta=horizontal_angle[2], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    front_left = pano_undistortion(frame, theta=horizontal_angle[3], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    front = pano_undistortion(frame, theta=horizontal_angle[4], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    front_right = pano_undistortion(frame,theta=horizontal_angle[5], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    img_concat = cv2.vconcat([cv2.hconcat([front_right, front,front_left]),cv2.hconcat([back_right, back,back_left])])
    return(img_concat)

#All direction convert with label
def pano_process_detction(model, frame, horizontal_angle=[-60,0,60,-120,-180,120], vertical_angle=0, res_x=1024, res_y=2048, fov=60.0): 
    back_left = pano_undistortion(image_detection(model,frame), theta=horizontal_angle[0], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    back = pano_undistortion(image_detection(model,frame), theta=horizontal_angle[1], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    back_right = pano_undistortion(image_detection(model,frame),theta=horizontal_angle[2], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    front_left = pano_undistortion(image_detection(model,frame), theta=horizontal_angle[3], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    front = pano_undistortion(image_detection(model,frame), theta=horizontal_angle[4], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    front_right = pano_undistortion(image_detection(model,frame),theta=horizontal_angle[5], phi=vertical_angle, res_x=res_x, res_y=res_y, fov=fov)
    img_concat = cv2.vconcat([cv2.hconcat([front_right, front,front_left]),cv2.hconcat([back_right, back,back_left])])
    return(img_concat)



def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

#Detection
def image_detection(model,frame):
    result = model(frame)
    label_result = result.xyxyn[0][:, -1], result.xyxyn[0][:, :-1]
    frame_with_label = plot_boxes(label_result,frame,model)
    return frame_with_label

#Plot boxes
def plot_boxes(results, frame, model):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        n = len(labels)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            if row[4] >= 0.3:
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                bgr = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
                cv2.putText(frame,model.names[int(labels[i])], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)
        return frame

#Video process
def video_pano_process(video_path,save_path,start_time_,end_time_,horizontal_angle=[-60,0,60,-120,-180,120],vertical_angle=0,res_x=1024, res_y=2048, fov=60.0, detection=False):
    if detection == True:
        model = load_model()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
            print('video is not opened')
    else:
        success, frame = cap.read()
        frame_shape = pano_process(frame, horizontal_angle=horizontal_angle, vertical_angle=vertical_angle, res_x=res_x,  res_y=res_y, fov=fov).shape
        frame_height = frame_shape[0]
        frame_width = frame_shape[1]
        fps = round(cap.get(cv2.CAP_PROP_FPS)) #为了后续计算余数方便
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration =  frame_count / fps    
        print('Total duration is %s seconds' % str(duration))
        start = start_time_
        start_time = fps * float(start)
        end = end_time_
        end_time = fps * float(end)
        # mp4 format output
        four_cc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(save_path, four_cc, fps, (int(frame_width),int(frame_height)))
        num = round(start_time)
        while True:
            success, frame = cap.read()
            if int(start_time) <= int(num) <= int(end_time):
                if success:
                    if detection == True and num%fps%1==0:     
                        video_writer.write(pano_process_detction(model, frame, horizontal_angle=horizontal_angle, vertical_angle=vertical_angle, res_x=res_x,  res_y=res_y, fov=fov))
                    else:
                        video_writer.write(pano_process(frame, horizontal_angle=horizontal_angle, vertical_angle=vertical_angle, res_x=res_x,  res_y=res_y, fov=fov))
                    print('Successfully writing {} frame, {}s in video.         '.format(int(num%fps),int(num/fps)),end="\r")
                else:
                    break
            num += 1
            if num > end_time:
                break
        cap.release()
