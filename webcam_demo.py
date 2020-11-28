import torch
import cv2
import time
import argparse

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=50) # (50, 75, 100, 101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=432)
parser.add_argument('--cam_height', type=int, default=368)
# parser.add_argument('--scale_factor', type=float, default=0.7125)
parser.add_argument('--scale_factor', type=float, default=0.4)
args = parser.parse_args()


def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    cap = cv2.VideoCapture('zhastay.mp4')
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0

    prev_frame_time = 0
    new_frame_time = 0

    no_people = 0
    many_people = 0
    turn_head = 0
    while True:
        cap.read()
        cap.read()
        cap.read()
        cap.read()
        cap.read()
        cap.read()
        cap.read()
        cap.read()
        cap.read()
        cap.read()

        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)
        
        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        new_frame_time = time.time()
        
        try:
            if keypoint_coords.sum() == 0 and no_people == 0:
                print('No people')
                no_people = 1
            elif keypoint_coords.sum() > 0 and ((keypoint_scores[0][3] > 0.8 and keypoint_scores[0][4] > 0.8) or (keypoint_scores[1][3] > 0.8 and keypoint_scores[1][4] > 0.8)):
                no_people = 0
            
            if keypoint_scores[1].sum() > 0 and many_people == 0:
                print('Many people')
                many_people = 1
            elif keypoint_scores[1].sum() == 0:
                many_people = 0
            
            if ((keypoint_scores[0][3] < 0.1 or keypoint_scores[0][4] < 0.1) or (keypoint_scores[0][3] < 0.1 or keypoint_scores[0][4] < 0.1)) and turn_head == 0 and no_people == 0:
                print('Turn Head', keypoint_scores[0][3], keypoint_scores[0][4])
                turn_head = 1
            elif (keypoint_scores[0][3] > 0.8 and keypoint_scores[0][4] > 0.8):
                turn_head = 0
        except:
            pass

        if frame_count%10 == 0:
            fps = round(1/(new_frame_time-prev_frame_time), 4)
        prev_frame_time = new_frame_time 
        cv2.putText(overlay_image, str(fps), (7, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('posenet', overlay_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except:
        pass
    print(time.time() - start_time)