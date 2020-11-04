from __future__ import division

import argparse
import configparser
import glob
import json
import os
import sys
import time
os.chdir(os.getcwd())
from darknet import Darknet
from preprocess import letterbox_image
from util import *
from PAC.player_audience_classfication import Player_Audience_Classfication as SFM
from PAC.bbm import RemoveAudienceWithBBM as RBBM

class DetectPlayers(SFM):
    '''
    Detect Players
    Using Yolo-v3 with network size of 1024 with 3 splits suggested.
    Provides text file with all detection in the format specified,
    frame,player_id,x,y,w,h,1,1,1,1
    '''

    def __init__(self,arg_parse):
        super().__init__()
        self.args = arg_parse
        self.conf_settings()
        self.load_model()
        assert os.path.exists(self.args['video_path']), 'Video path error'

        path,name=os.path.split(self.args['video_path'])

        self.new_dir=os.path.join(path,name.split('.')[0])
        self.img_dir=os.path.join(self.new_dir,'image')
        # self.txt_path=os.path.join(self.new_dir,'%s_dt.txt'%name.split('.')[0])
        self.txt_path=os.path.join(self.new_dir,'players.txt')
        if not os.path.exists(self.new_dir):
            os.mkdir(self.new_dir)
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        print('\n-------Detection (SFM)--------')
        # Do detection of person class objects ( also use sfm to detect players)
        self.get_bbox_video_split()

        # use bbm
        if self.args['bbm']:
            print('\n-------BBM--------')
            RBBM(self.txt_path, self.img_dir, self.FRAME_HEIGHT, self.FRAME_WIDTH, self.thersh_w, self.thersh_h, remove_small_large_bbox = True,
            format = 'jpg', split_method = True, split_rate = 9000,)


        # Create Json and save params
        #print('-------Saving  Params--------\n')
        data={'vid_file_name':name,
              'f_height':int(self.FRAME_HEIGHT),
              'f_width':int(self.FRAME_WIDTH),
              'base_dir':path
              }
        jsonfile = os.path.join(self.new_dir, 'params.json')
        with open(jsonfile,'w') as outfile:
            json.dump(data,outfile)

        print('\n------- Done --------\n')






    def conf_settings(self):
        conf = configparser.ConfigParser()
        conf_file = self.args['settings_path']
        conf.read(conf_file)
        # Model params
        self.cfg=conf.get('model','cfg')
        self.weights=conf.get('model','weights')
        self.confidence=float(conf.get('model','condifence'))
        self.network_sz=int(conf.get('model','network_sz'))
        self.nms_thresh=float(conf.get('model','nms_thresh'))
        # Video params
        if self.args['video_path'] is None: self.args['video_path']=conf.get('video','video_path')
        if self.args['save_video'] is None: self.args['save_video']=conf.getboolean('video','save_video')
        if self.args['split'] is None: self.args['split']=conf.getint('video','split')
        if self.args['ce'] is None: self.args['ce']=conf.getboolean('video','ce')
        if self.args['alpha'] is None: self.args['alpha']=conf.getfloat('video','alpha')
        if self.args['beta'] is None: self.args['beta']=conf.getfloat('video','beta')
        if self.args['show_live'] is None: self.args['show_live']=conf.getboolean('video','show_live')
        # BBM params
        self.thersh_w=[conf.getint('bbm','width_min'),conf.getint('bbm','width_max')]
        self.thersh_h=[conf.getint('bbm','height_min'),conf.getint('bbm','height_max')]




    def prep_image(self, orig_im, inp_dim):
        """
        Prepare image for inputting to the neural network.

        Returns a Variable
        """
        dim = orig_im.shape[1], orig_im.shape[0]
        img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
        img_ = img[:, :, ::-1].transpose((2, 0, 1)).copy()
        img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
        return img_, orig_im, dim


    def load_model(self):

        self.num_classes = 80

        self.CUDA = torch.cuda.is_available()

        bbox_attrs = 5 + self.num_classes

        print("Loading network.....")
        self.model = Darknet(self.cfg)
        self.model.load_weights(self.weights)
        print("Player Network successfully loaded")

        self.model.net_info["height"] = self.network_sz
        self.inp_dim = int(self.model.net_info["height"])
        assert self.inp_dim % 32 == 0
        assert self.inp_dim > 32

        if self.CUDA:
            self.model.cuda()
            print('using GPU ')
        else:
            print('no GPU')

        self.model.eval()

    def contrast_enhance(self, img, alpha, beta):
        # alpha Contrast control (1.0-3.0)
        # beta Brightness control (0-100)
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    def read_video(self, video_path, video_save=False):

        cap = cv2.VideoCapture(video_path)
        self.FRAME_WIDTH = cap.get(3)
        self.FRAME_HEIGHT = cap.get(4)
        FRAME_FPS = cap.get(5)

        if video_save:

            output_file = os.path.join(self.new_dir,
                                       'DT_' + os.path.split(video_path)[-1].split('.')[0] + '.avi')
            out_video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FRAME_FPS,
                                        (int(self.FRAME_WIDTH), int(self.FRAME_HEIGHT)))
            return cap, out_video
        else:
            return cap, None

    def draw_bbox(self, xyxy, img):
        cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 0, 255), 2)
        return img



    def get_bbox_video_split(self):

        cap, out_video = self.read_video(self.args['video_path'], self.args['save_video'])
        assert cap.isOpened(), 'Cannot Open video'
        txt_ = open(self.txt_path, 'w')
        frames = 0
        fps = []

        while cap.isOpened():
            t1 = time.time()
            ret, frame_ = cap.read()

            if ret:
                # Variables
                draw_frame_ = frame_.copy()
                if self.args['ce']:
                    frame_ = self.contrast_enhance(frame_, self.args['alpha'], self.args['beta'])

                cv2.imwrite('%s/%d.jpg' % (self.img_dir, frames), frame_)
                bbox = []

                # Start
                ys, xs, ch = frame_.shape
                idxs = int(xs / self.args['split'])
                end = 0
                start = 0
                adjust_start = -50
                adjust_end = 50
                for i_ in range(self.args['split']):
                    start = idxs * i_
                    end = (i_ + 1) * idxs
                    if start > 0:
                        start = start + adjust_start
                    if end < (idxs - 1) * self.args['split']:
                        end = end + adjust_end

                    frame = frame_[:, start:end]

                    # use split image
                    img, orig_im, dim = self.prep_image(frame, self.inp_dim)

                    im_dim = torch.FloatTensor(dim).repeat(1, 2)

                    if self.CUDA:
                        im_dim = im_dim.cuda()
                        img = img.cuda()

                    with torch.no_grad():
                        output = self.model(Variable(img), self.CUDA)
                    output = write_results(output, self.confidence, self.num_classes, nms=True, nms_conf=self.nms_thresh)

                    im_dim = im_dim.repeat(output.size(0), 1)
                    scaling_factor = torch.min(self.inp_dim / im_dim, 1)[0].view(-1, 1)

                    output[:, [1, 3]] -= (self.inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
                    output[:, [2, 4]] -= (self.inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

                    output[:, 1:5] /= scaling_factor

                    for i in range(output.shape[0]):
                        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
                        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])
                    for x in output:
                        x1 = int(x[1]) + start
                        y1 = int(x[2])
                        x2 = int(x[3]) + start
                        y2 = int(x[4])
                        cls = int(x[-1])
                        score = float(x[-2])
                        if cls == 0 and score >= self.confidence:
                            bbox.append([x1, y1, x2, y2, round(score)])

                # NMS
                bboxes = non_max_suppression_fast(np.asarray(bbox), 0.5)
                del bbox

                t2 = time.time()

                sys.stdout.write('\rFPS: %2.2f   Frame %d' % ((1 / (t2 - t1)), frames))
                sys.stdout.flush()
                fps.append((1 / (t2 - t1)))
                for bb in bboxes:
                    # x1 = bb[0]
                    # y1 = bb[1]
                    # x2 = bb[2]
                    # y2 = bb[3]

                    if self.args['sfm']:

                        # Classify images into player and non-player
                        pac_=self.predict_image(frame_[bb[1]:bb[3],bb[0]:bb[2]])
                        if pac_=='player':
                            draw_frame_ = self.draw_bbox([bb[0], bb[1], bb[2], bb[3]], draw_frame_)
                            txt_.write(
                                '%d,-1,%.2f,%.2f,%.2f,%.2f,1.0,1,1,1\n' % (frames, bb[0], bb[1], bb[2] -bb[0], bb[3] - bb[1]))
                    else:
                        draw_frame_ = self.draw_bbox([bb[0], bb[1], bb[2], bb[3]], draw_frame_)
                        txt_.write(
                            '%d,-1,%.2f,%.2f,%.2f,%.2f,1.0,1,1,1\n' % (
                            frames, bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]))

                frames += 1
                if self.args['show_live']:
                    cv2.imshow('test', cv2.resize(draw_frame_, (1200, 800)))
                    cv2.waitKey(1)
                if self.args['save_video']:
                    out_video.write(draw_frame_)
            else:
                break


        print('\nAverage FPS: %2.2f' % (sum(fps) / len(fps)))


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # print(overlap)
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


if __name__ == '__main__':


    """ Parse arguments to the detect module """
    parser = argparse.ArgumentParser(description='YOLO v3 Player Detection and Classification Script')
    # model
    parser.add_argument("--settings_path", dest="settings_path",
                        help="Settings file path",required=True)
    parser.add_argument("--sfm", dest="sfm",
                        help="Spatial filter method", default=False,action='store_true')
    parser.add_argument("--bbm", dest="bbm",
                        help="Bounding box method", default=False,action='store_true')

    # video
    # parser.add_argument("--video_path", dest='video_path', help="Location of video file",
    #                     required=True, type=str)
    # Video additional params
    # parser.add_argument("--save_video", dest='save_video', help="Save resulting video with bbox",
    #                     default=False, type=bool)
    # parser.add_argument("--split", dest='split', help="Frame Splits",
    #                     default=3, type=int)
    # parser.add_argument("--CE", dest='CE', help="Contrast Enhancement",
    #                     default=True, type=bool)
    # parser.add_argument("--Alpha", dest='alpha', help="Contrast param (1.0 - 3.0)",
    #                     default=1.95, type=float)
    # parser.add_argument("--Beta", dest='beta', help="Brightness param (1-100)",
    #                     default=1, type=float)
    # parser.add_argument("--show_live", dest='show_live', help="Watch live detections",
    #                     default=False, type=bool)

    ### Load params from settings file

    parser.add_argument("--video_path", dest='video_path', help="Location of video file")
    # Video additional params
    parser.add_argument("--save_video", dest='save_video', help="Save resulting video with bbox")
    parser.add_argument("--split", dest='split', help="Frame Splits")
    parser.add_argument("--ce", dest='ce', help="Contrast Enhancement")
    parser.add_argument("--alpha", dest='alpha', help="Contrast param (1.0 - 3.0)")
    parser.add_argument("--beta", dest='beta', help="Brightness param (1-100)")
    parser.add_argument("--show_live", dest='show_live', help="Watch live detections")



    args = vars(parser.parse_args())
    # print(args)
    DetectPlayers(args)



