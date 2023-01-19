from __future__ import absolute_import

import sys
import torch
import argparse
import os
from trajnetplusplustools.data import TrackRow


sys.path.insert(1,"/data2/mcleav/conformalRNNs/icra_2022/carlaLSTMTraining/Carla/trajnetplusplusbaselines")
from trajnetbaselines.lstm import LSTMPredictor
import trajnetplusplustools

path = os.path.abspath(trajnetplusplustools.__file__)
print(path)

# sys.path.append("/data2/mcleav/conformalRNNs/icra_2022/carlaLSTMTraining/Carla/trajnetplusplusbaselines/trajnetbaselines/lstm")
# from lstm import LSTMPredictor



class trajnetLSTM():
    def __init__(self, model_name,device="cpu"):

        self.predictor = LSTMPredictor.load(model_name)
        self.predictor.model.to(torch.device(device))

        pass

    def predict(self,x,y,args):
        paths = []
        
        for i,_ in enumerate(x):
            paths.append(TrackRow(x=x[i],y=y[i],frame=i,pedestrian=0))
        paths = [paths]
        # print("paths: " + str(paths))

        scene_goal = [[0,0]]

        # print("Running lstm")
        # print("paths: " + str(paths))
        # print("paths type: " + str(type(paths)))
        # print("paths type: " + str(type(paths[0])))
        # print("paths type: " + str(type(paths[0][0])))
        # print("scene_goal: " + str(scene_goal))
        # print("predict_length: " + str(args.pred_length))
        # print("obs_length: " + str(args.obs_length))
        # print("modes: " + str(args.modes))
        # print("other args: " + str(args))

        predictions = self.predictor(paths, scene_goal, n_predict=args.pred_length, obs_length=args.obs_length, modes=args.modes, args=args)

        return predictions

    def predict_batch(self,xs,ys,args,ids=None):

        scene_goal = []

        for i in range(len(xs)):
            scene_goal.append([0,0])

        paths = []
        print("len xs: " + str(len(xs)))
        for i in range(len(xs)):
            temp_path = []
            for j in range(len(xs[i])):
                if ids is None:
                    temp_path.append(TrackRow(x=xs[i][j],y=ys[i][j],frame=j,pedestrian=i))
                else:
                    temp_path.append(TrackRow(x=xs[i][j],y=ys[i][j],frame=j,pedestrian=ids[i]))
            paths.append(temp_path)
            
        # print(paths)
        print("len paths " + str(len(paths)))
        predictions = self.predictor(paths, scene_goal, n_predict=args.pred_length, obs_length=args.obs_length, modes=args.modes, args=args)
        # print("len predictions " + str(len(predictions)))
        return predictions

if __name__ == "__main__":
    print("Testing LSTM class")

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print(args)
    print(type(args))

    model_name = "/data2/mcleav/conformalRNNs/icra_2022/carlaLSTMTraining/Carla/trajnetplusplusbaselines/OUTPUT_BLOCK/synth_data/lstm_goals_carla_social_None.pkl.epoch10"

    a = trajnetLSTM(model_name)

    x = [i for i in range(10)]
    y = [5 for i in range(10)]
    args.pred_length = 20
    args.obs_length = 20
    args.modes = 1
    args.normalize_scene = False
    
    predictions = a.predict(x,y,args)
    print(predictions)