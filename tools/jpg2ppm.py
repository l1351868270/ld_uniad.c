
import argparse
import os
import shutil
import cv2

if __name__ == '__main__':
    # python jpg2ppm.py --jpg_path=/home/ubuntu/vtv/Autopilot-TensorFlow/driving_dataset/data --ppm_path=/home/ubuntu/vtv/Autopilot-TensorFlow/driving_dataset/data_ppm
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpg_root', type=str, default='./data/nuscenes/samples/')
    parser.add_argument('--ppm_root', type=str, default='./data/ld_nuscenes/samples/')
    args = parser.parse_args()
    jpg_root = args.jpg_root
    ppm_root = args.ppm_root

    if not os.path.exists(jpg_root):
        raise Exception(f'jpg_root: {jpg_root} is not exists')
    
    if os.path.exists(ppm_root):
        shutil.rmtree(ppm_root)

    if not os.path.exists(ppm_root):
        os.makedirs(ppm_root)

    for d in os.listdir(jpg_root):
        if not d.startswith('CAM'):
            continue
        jpg_path = os.path.join(jpg_root, d)
        ppm_path = os.path.join(ppm_root, d)
        os.makedirs(ppm_path)
        
        for f in os.listdir(jpg_path):
            d_jpg = os.path.join(jpg_path, f)
            print(f'd_jpg: {d_jpg} ')
            img = cv2.imread(d_jpg)
            f_ppm = os.path.splitext(f)[0] + '.ppm'
            d_ppm = os.path.join(ppm_path, f_ppm)
            cv2.imwrite(d_ppm, img)