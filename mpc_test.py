import glob
import os
import sys
import time
import subprocess
import queue
# from tkinter import _ImageSpec
# from agents.navigation.controller import VehiclePIDController


sys.path.append('../carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')


import numpy as np
import argparse
from time import ctime
import logging
import random
from casadi import *
from test import *
import math
import carla
import matplotlib.pyplot as plt
import pickle

from trajnetLSTM import trajnetLSTM
from spawn_peds_lstm_func import spawn_peds, reset_peds

def contheta(thet):
    if thet < 0:
        thet = 360 - abs(thet)
    return thet


def get_max_d_vals(d_dict):

    max_d_vals = []
    first_iter = True
    for key in d_dict:
        if first_iter:
            max_d_vals = d_dict[key]
            first_iter = False
        else:
            for i in range(len(d_dict[key])):
                if d_dict[key][i] > max_d_vals[i]:
                    max_d_vals[i] = d_dict[key][i]
    
    return max_d_vals


def check_dynamics(x,y,V,theta,uv,delta,delta_t,L):

    for i in range(len(x)):
        print("Step " + str(i))
        if i==0:
            continue

        x_diff = x[i] - x[i-1] - delta_t * (V[i-1]*math.cos(theta[i-1]))
        y_diff = y[i] - y[i-1] - delta_t * (V[i-1]*math.sin(theta[i-1]))

        theta_diff = theta[i] - theta[i-1] - delta_t * (V[i-1]/L*math.tan(delta[i-1]))
        v_diff = V[i] - V[i-1] - delta_t * uv[i-1]

        print("x diff: " + str(x_diff))
        print("x diff: " + str(y_diff))
        print("V diff: " + str(v_diff))
        print("theta diff: " + str(theta_diff))

    return


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-n', '--number-of-scenes',
        metavar='N',
        default=10,
        type=int,)
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='vehicles filter (default: "vehicle.*")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='pedestrians filter (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Enanble')
    # argparser.add_argument(
    #     '-m', '--map',
    #     help='load a new map, use --list to see available maps')

    args = argparser.parse_args()
    
    #actor_list = []
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    client = carla.Client(args.host, args.port)
    client.set_timeout(2.0)
    IM_WIDTH = 1280
    IM_HEIGHT = 720
    
    
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(5.0)
    if args.hybrid:
        traffic_manager.set_hybrid_physics_mode(True)

    print("sync: " + str(args.sync))
    if args.sync:
        CARLA_HZ = 20
        settings = world.get_settings()
        traffic_manager.set_synchronous_mode(True)
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(1/CARLA_HZ) #0.05
        world.apply_settings(settings)

    print("Settings: " + str(world.get_settings()))

    world.tick()

    
    ## setup lstm
    # model_name = "/data2/mcleav/conformalRNNs/icra_2022/carlaLSTMTraining/Carla/trajnetplusplusbaselines/OUTPUT_BLOCK/synth_data/lstm_goals_carla_social_None.pkl.epoch10" ## 20Hz
    model_name ="/data2/mcleav/carla/carla_0_9_9/PythonAPI/examples/lstm/2Hz/lstm_carla_70_social_None.pkl" ## 2Hz

    numTrialLow = 0
    numTrial = 100
    for trial in range(numTrialLow,numTrialLow+numTrial):

        print("Trial " + str(trial))

        lstm_model = trajnetLSTM(model_name)

        save_dir_name = "100TrialsPred4_noTurns_60sec_bugFixes_14/only_near_peds_" + str(trial) + "_actual_d_values/"
        imageFolder = "images/runs_manyTrials/" + save_dir_name + "mpcTest/"
        lstmImageFolder = "images/runs_manyTrials/" + save_dir_name + "lstmPredictions/"
        obstacleDataFolder = "images/runs_manyTrials/" + save_dir_name + "obstPreds/"
        carDataFolder = "images/runs_manyTrials/" + save_dir_name + "/"
        print("Image folder: " + str(imageFolder))
        print("LSTM Pred Folder: " + str(lstmImageFolder))
        os.makedirs(imageFolder, exist_ok=True)
        os.makedirs(lstmImageFolder, exist_ok=True)
        os.makedirs(obstacleDataFolder, exist_ok=True)


        try:

            vehicles_list = []
            camera_list = []

            blueprint_library = world.get_blueprint_library()
            vehicle_bp = blueprint_library.filter('vehicle.toyota.prius*')[0]

            initpoint = carla.Transform(carla.Location(x=-48.81999023, y=-18.79507507, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=89.83876))
            # initpoint = carla.Transform(carla.Location(x=-48.81999023, y=42.55527832, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=-90))
            print("Init point: " + str(initpoint))

            vehicle = world.spawn_actor(vehicle_bp, initpoint)

            print("spawning camera")

            world.tick()

            camera_bp = blueprint_library.find('sensor.camera.rgb')
            camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
            camera_bp.set_attribute('image_size_x', str(IM_WIDTH))
            camera_bp.set_attribute('image_size_y', str(IM_HEIGHT))
            camera_bp.set_attribute('fov', '110')
            camera_bp.set_attribute('sensor_tick', '0.1')
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            camera_list.append(camera)
            image_queue = queue.Queue()
            camera.listen(lambda image: image_queue.put(image))
            imageCount = 0
            print("spawned camera")

            world.tick()


            wplist = world.get_map().get_topology()
            wps = wplist[81][1].next_until_lane_end(1.0)
            for w in wps:
                world.debug.draw_string(w.transform.location, 'O', draw_shadow=False,
                                        color=carla.Color(r=0, g=255, b=0), life_time=20.0,
                                        persistent_lines=True)

            world.tick()
            control = carla.VehicleControl()
            vehicles_list.append(vehicle)


            ## spawning pedestrians
            print("Number of pedestrians " + str(world.get_actors().filter('walker.pedestrian.*')))
            if len(world.get_actors().filter('walker.pedestrian.*')) == 0:
                spawn_peds(client,args)
            else:
                reset_peds(client,args)


            target_waypoint = carla.Transform(carla.Location(x=-48.68674805, y=42.55527832, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=89.83876))
            # target_waypoint = carla.Transform(carla.Location(x=-48.68674805, y=-4.79507507, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=89.83876))
            client.get_world().debug.draw_string(target_waypoint.location, 'O', draw_shadow=False,
                                                color=carla.Color(r=255, g=0, b=0), life_time=20,
                                                persistent_lines=True)

            actor_list = world.get_actors()

            ped_dict = {}
            veh_dict = {}
            prediction_dict_ped = {}
            prediction_dict_veh = {}
            ped_dict_at_pred_step = {}
            num_tick_before_MPC = int(3*CARLA_HZ)
            for _ in range(num_tick_before_MPC):
                for ped in actor_list.filter('walker.pedestrian.*'):
                    if ped.id in ped_dict:
                        ped_loc = ped.get_location()
                        ped_dict[ped.id].append([ped_loc.x, ped_loc.y])
                    else:
                        ped_loc = ped.get_location()
                        ped_dict[ped.id] = [[ped_loc.x, ped_loc.y]]
                for veh in actor_list.filter('vehicle.*'):
                    if veh.id == vehicle.id:
                        continue
                    if veh.id in veh_dict:
                        veh_loc = veh.get_location()
                        veh_dict[veh.id].append([veh_loc.x, veh_loc.y])
                    else:
                        veh_loc = veh.get_location()
                        veh_dict[veh.id] = [[veh_loc.x,veh_loc.y]]


                world.tick()

            # -------------------MPC--------------------

            opti = casadi.Opti()
            
            # frequencies
            MPC_HZ = 2
            LSTM_HZ = 2#20

            # LSTM parameters
            t_obs = 3
            t_pred = 5 ## TODO: increase this (nevermind lol)
            args.pred_length = int(t_pred*LSTM_HZ)
            args.obs_length = int(t_obs*LSTM_HZ)
            args.modes = 1
            args.normalize_scene = False
            # d_values = [0.25, 1.99, 4.64, 6.94, 8.99, 10.63, 12.47, 14.83, 17.1, 19.49] ## 20Hz
            # d_values = [0.2657, 0.4754, 0.6629, 0.7398, 0.9302, 1.1502, 1.3524, 1.5646, 1.7781, 2.0121] # 2Hz
            use_d_values = True
            obst_avoidance_thresh = 7 # FIXME: was 5

            with open("d_values_with_H_30.pkl","rb") as f:
                d_dict = pickle.load(f)
            d_values = get_max_d_vals(d_dict)

            # MPC parameters
            T_1 = 60*MPC_HZ # was 30*MPC_HZ
            T = t_pred*MPC_HZ ## TODO: increase this in accordance with args.pred_length
            H = 60 ## TODO: make this a function of MPC_HZ once obstacle constraints are added
            L = 3
            eps = 0
            Delta = 1/MPC_HZ
            goal_pos_thresh = 3
            tau_base = 5
            tau_inc = 0.25
            obst_slack_mult = 10000000
            slack_decay = 0.9
            plotMPCSolns = False
            


            print("Starting location: " + str([initpoint.location.x,initpoint.location.y]))
            print("Goal location: " + str([target_waypoint.location.x,target_waypoint.location.y]))
            
            x_state = [initpoint.location.x]*T
            y_state = [initpoint.location.y]*T
            v_state = [0]*T
            theta_state = [0]*T
            uv_state = [0]*T
            delta_state = [0]*T

            vehicle_x = []
            vehicle_y = []
            vehicle_v = []
            vehicle_theta = []

            xref = target_waypoint.location.x
            yref = target_waypoint.location.y

            yref_fake = yref + 50

            uv_max = 10
            V_max = 15

            p_opts = {"expand":True, "verbose":False, "ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
            s_opts = {"max_iter": 5000}
            opti.solver('ipopt', p_opts, s_opts)
            sol = None
            prev_sol = None

            mpc_run_times = []
            mpc_solve_times = []
            initFrame = 0

            mpc_solns_x = []
            mpc_solns_y = []
            mpc_solns_v = []
            mpc_solns_theta = []
            mpc_solns_uv = []
            mpc_solns_delta = []
            mpc_solns_slacks = []

            vehicle_xs = []
            vehicle_ys = []
            vehicle_vels = []
            vehicle_thetas = []

            feasible = []

            # T_1 = 10 ## TODO: For debugging only, remove once code is finished

            for t in range(T_1):

                print("Step " + str(t))

                # opti = casadi.Opti()
                # p_opts = {"expand":True, "verbose":False, "ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
                # s_opts = {"max_iter": 5000}
                # opti.solver('ipopt', p_opts, s_opts)
                # sol = None


                ## add in time.time() to get the timespans of different parts of code
                mpc_start_time = time.time()

                x = opti.variable(T)
                y = opti.variable(T)
                V = opti.variable(T)
                theta = opti.variable(T)

                uv = opti.variable(T)
                delta = opti.variable(T)

                obst_slack = opti.variable(T)

                vehicle_location = [vehicle.get_location().x,vehicle.get_location().y]
                vehicle_xs.append(vehicle_location[0])
                vehicle_ys.append(vehicle_location[1])
                vehicle_vels.append(math.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2))
                vehicle_thetas.append(math.radians(vehicle.get_transform().rotation.yaw))

                print("Vehicle location: " + str(vehicle_location))
                x_current = vehicle_location[0]
                y_current = vehicle_location[1]
                v_current = math.sqrt(vehicle.get_velocity().x**2 + vehicle.get_velocity().y**2)
                theta_current = math.radians(vehicle.get_transform().rotation.yaw)            

                vehicle_x.append(x_current)
                vehicle_y.append(y_current)
                vehicle_v.append(v_current)
                vehicle_theta.append(theta_current)

                opti.subject_to(x[0] == x_current)
                opti.subject_to(y[0] == y_current)
                opti.subject_to(V[0] == v_current)
                opti.subject_to(theta[0] == theta_current)
                # opti.subject_to(obst_slack >= 0)

                opti.set_initial(x[0], x_current)
                opti.set_initial(y[0], y_current)
                opti.set_initial(V[0], v_current)
                opti.set_initial(theta[0], theta_current)
                opti.set_initial(obst_slack[0], 0)

                if sol is not None:
                    ## warm start the solver
                    opti.set_initial(sol.value_variables())

                while not image_queue.empty():
                    image = image_queue.get(timeout=1.0)
                    if initFrame == 0:
                        initFrame = image.frame
                    # print("Saving image frame " + str(image.frame))
                    image.save_to_disk(imageFolder + '%08d.jpg' % (image.frame))
                    # image.save_to_disk(imageFolder + '%08d.jpg' % (image.frame - initFrame))
                    # image.save_to_disk(imageFolder + '%08d.jpg' % imageCount)
                    imageCount += 1

                print("Carla vehicle state: " + str(vehicle.get_transform()))
                print("Carla vehicle vel: " + str(vehicle.get_velocity()))
                print("MPC vehicle state: " + str([x_current,y_current,v_current,theta_current]))

                if math.sqrt((vehicle_location[0] - xref)**2 + (vehicle_location[1] - yref)**2) <= goal_pos_thresh:
                    print("Reached goal!")
                    break
                

                obj = 0
                # opti.subject_to(opti.bounded(-pi/6, delta[0], pi/6))
                opti.subject_to(opti.bounded(math.tan(-pi/6), delta[0], math.tan(pi/6)))
                opti.subject_to(opti.bounded(-uv_max, uv[0], uv_max))

                for i in range(1, T):
                    # opti.subject_to(opti.bounded(-pi/6, delta[i], pi/6))
                    opti.subject_to(opti.bounded(math.tan(-pi/6), delta[i], math.tan(pi/6)))
                    opti.subject_to(opti.bounded(-uv_max, uv[i], uv_max))
                    # opti.subject_to(opti.bounded(-49.61999023, x[i], -48.11999023))
                    # opti.subject_to(opti.bounded(-51, x[i], -46))
                    opti.subject_to(opti.bounded(-50, x[i], -47))

                    opti.subject_to(opti.bounded(0, V[i], V_max))
                    # opti.subject_to(opti.bounded(-V_max, V[i], V_max))

                    opti.subject_to(x[i] == x[i-1] + Delta * (V[i-1]*cos(theta[i-1])))
                    opti.subject_to(y[i] == y[i-1] + Delta * (V[i-1]*sin(theta[i-1])))
                    # opti.subject_to(theta[i] == theta[i-1] + Delta * (V[i-1]/L*tan(delta[i-1])))
                    opti.subject_to(theta[i] == theta[i-1]) # + Delta * (V[i-1]/L*delta[i-1])) ## FIXME: disabling turns for now
                    opti.subject_to(V[i] == V[i-1] + Delta*(uv[i-1]))
            
                    # obj += V[i]**2 + uv[i]**2 + 10*T * (x[T-t-1]-xref)**2 + (y[T-t-1]-yref)**2
                    # obj += 10*T * (x[i]-xref)**2 + (y[i]-yref)**2
                    obj += 10*T * (x[i]-xref)**2 + (y[i]-yref_fake)**2
                    # obj += T * (x[T-1]-xref)**2 + (y[T-1]-yref)**2
                

                all_obs_x = []
                all_obs_y = []
                pred_ids = []
                obs_downsample_rate = int(CARLA_HZ/LSTM_HZ)
                for ped in actor_list.filter('walker.pedestrian.*'):
                    if ped.id in ped_dict:
                        ped_obs = ped_dict[ped.id][-(args.obs_length)*obs_downsample_rate+obs_downsample_rate-1::obs_downsample_rate]
                        
                        print("All ped obs " + str(ped_dict[ped.id][-(args.obs_length)*obs_downsample_rate:]))
                        print("LSTM input obs " + str(ped_obs))
                        # print("Ped loc: " + str(ped_obs))
                        ped_obs_x = [x-vehicle_location[0] for [x,y] in ped_obs]
                        ped_obs_y = [y-vehicle_location[1] for [x,y] in ped_obs]

                        all_obs_x.append(ped_obs_x)
                        all_obs_y.append(ped_obs_y)
                        pred_ids.append(ped.id)

                        if ped.id in ped_dict_at_pred_step:
                            ped_dict_at_pred_step[ped.id].append([ped_obs[-1][0],ped_obs[-1][1]])
                        else:
                            ped_dict_at_pred_step[ped.id] = [[ped_obs[-1][0],ped_obs[-1][1]]]
                    else:
                        # print("New pedestrian appeared. Ignoring")
                        pass

                for veh in actor_list.filter('vehicle.*'):
                    if veh.id in veh_dict:
                        veh_obs = veh_dict[veh.id][-(args.obs_length)*obs_downsample_rate::obs_downsample_rate]

                        # print("Veh loc: " + str(veh_obs))
                        veh_obs_x = [x-vehicle_location[0] for [x,y] in veh_obs]
                        veh_obs_y = [y-vehicle_location[1] for [x,y] in veh_obs]

                        all_obs_x.append(veh_obs_x)
                        all_obs_y.append(veh_obs_y)
                        pred_ids.append(-veh.id)
                    else:
                        # print("New vehicle appeared. Ignoring")
                        pass


                lstm_start_time = time.time()
                predictions = lstm_model.predict_batch(all_obs_x,all_obs_y,args,pred_ids)
                lstm_end_time = time.time()
                print("lstm run time: " + str(lstm_end_time-lstm_start_time))


                ped_preds = []
                veh_preds = []
                lstm_pred_downsample_rate = int(LSTM_HZ/MPC_HZ)
                for i,ped in enumerate(actor_list.filter('walker.pedestrian.*')):
                    # print("Ped " + str(ped.id))
                    if i==0:
                        predictions_downsample = predictions[0][0][lstm_pred_downsample_rate-1::lstm_pred_downsample_rate]
                    else:
                        this_preds = [predictions[0][1][dummy_t][i-1] for dummy_t in range(len(predictions[0][1]))]
                        # print(this_preds)
                        predictions_downsample = this_preds[lstm_pred_downsample_rate-1::lstm_pred_downsample_rate]

                    predictions_downsample_global = []
                    for k in range(len(predictions_downsample)):
                        predictions_downsample_global.append([predictions_downsample[k][0]+vehicle_location[0], predictions_downsample[k][1]+vehicle_location[1]])
                    
                    ped_preds.append(predictions_downsample_global)

                    if ped.id in prediction_dict_ped:
                        prediction_dict_ped[ped.id].append(predictions_downsample_global)
                    else:
                        prediction_dict_ped[ped.id] = [predictions_downsample_global]

                num_ped = len(actor_list.filter('walker.pedestrian.*'))
                print("Num peds: " + str(num_ped))
                # print("Num vehs: " + str(len(actor_list.filter('vehicle.*'))))
                ego_veh_offset = 0
                for i,veh in enumerate(actor_list.filter('vehicle.*')):
                    # print("Veh " + str(veh.id))
                    if veh.id == vehicle.id:
                        ego_veh_offset = 1
                        continue

                    this_preds = [predictions[0][1][dummy_t][i+num_ped-1-ego_veh_offset] for dummy_t in range(len(predictions[0][1]))]
                    # print(this_preds)
                    predictions_downsample = this_preds[lstm_pred_downsample_rate-1::lstm_pred_downsample_rate]
                    # print("Preds downsample")
                    # veh_preds.append(predictions_downsample)

                    # print("preds downsample global")
                    predictions_downsample_global = []
                    for k in range(len(predictions_downsample)):
                        predictions_downsample_global.append([predictions_downsample[k][0]+vehicle_location[0], predictions_downsample[k][1]+vehicle_location[1]])

                    veh_preds.append(predictions_downsample_global)
                    # print("Adding to veh dict")
                    # print(prediction_dict_veh)
                    if veh.id in prediction_dict_veh:
                        prediction_dict_veh[veh.id].append(predictions_downsample_global)
                    else:
                        prediction_dict_veh[veh.id] = [predictions_downsample_global]

                print("Making pedestrian constriants")
                tau_vals = [tau_base+tau*tau_inc for tau in range(1,T)]
                # print(tau_vals)

                for k in range(len(ped_preds)):
                    predictions_x = [x for [x,y] in ped_preds[k]]
                    predictions_y = [y for [x,y] in ped_preds[k]]

                    for tau in range(1,T):
                        # print("tau: " + str(tau) + ", " + str(2+tau/2))
                        # print("Obstacle loc at tau: " + str([predictions_x[tau-1],predictions_y[tau-1]]))

                        if use_d_values:
                            obst_dist_thresh = d_values[min(len(d_values)-1,tau-1)] + obst_avoidance_thresh
                        else:
                            obst_dist_thresh = tau_vals[tau-1] + obst_avoidance_thresh
                        # print("dist thresh: " + str(obst_dist_thresh))

                        ## actual constraint
                        # opti.subject_to(sqrt((x[tau]-predictions_x[tau-1])**2+(y[tau]-predictions_y[tau-1])**2) >= obst_dist_thresh)

                        ## barriers on constraint
                        # obj += 50/log((sqrt((x[tau]-predictions_x[tau-1])**2+(y[tau]-predictions_y[tau-1])**2) - obst_dist_thresh))
                        # obj += -100*sqrt((x[tau]-predictions_x[tau-1])**2+(y[tau]-predictions_y[tau-1])**2)

                        ## slack for soft constraint
                        opti.subject_to(obst_slack[tau] >= 0)
                        opti.subject_to(obst_slack[tau] >= obst_dist_thresh - sqrt((x[tau]-predictions_x[tau-1])**2+(y[tau]-predictions_y[tau-1])**2))
                        obj += obst_slack_mult*(obst_slack[tau]**2)*(slack_decay**tau)

                        

                print("Making vehicle constraints")
                for k in range(len(veh_preds)):
                    predictions_x = [x for [x,y] in veh_preds[k]]
                    predictions_y = [y for [x,y] in veh_preds[k]]

                    for tau in range(1,T):

                        if use_d_values:
                            obst_dist_thresh = d_values[min(len(d_values)-1,tau-1)] + obst_avoidance_thresh
                        else:
                            obst_dist_thresh = tau_vals[tau-1] + obst_avoidance_thresh

                        ## actual constraint
                        # opti.subject_to(sqrt((x[tau]-predictions_x[tau-1])**2+(y[tau]-predictions_y[tau-1])**2) >= obst_dist_thresh)

                        ## slack for soft constraint
                        opti.subject_to(obst_slack[tau] >= obst_dist_thresh - sqrt((x[tau]-predictions_x[tau-1])**2+(y[tau]-predictions_y[tau-1])**2))

                opti.minimize(obj)
                
                mpc_solve_start_time = time.time()
                try:
                    sol = opti.solve()
                    prev_sol = sol
                    mpc_solve_end_time = time.time()
                    feasible.append(1)

                    x_current = sol.value(x)[1]
                    y_current = sol.value(y)[1]
                    v_current = sol.value(V)[1]
                    theta_current = sol.value(theta)[1]
                    uv_current = sol.value(uv)[0]
                    # delta_current = sol.value(delta)[0]
                    delta_current = math.atan(sol.value(delta)[0])
            
                    print("Full mpc solution")
                    print("x: " + str(sol.value(x)))
                    print("y: " + str(sol.value(y)))
                    print("V: " + str(sol.value(V)))
                    print("theta: " + str(sol.value(theta)))
                    print("uv: " + str(sol.value(uv)))
                    print("delta: " + str(sol.value(delta)))
                    print("slack: " + str(sol.value(obst_slack)))

                    mpc_solns_x.append(sol.value(x))
                    mpc_solns_y.append(sol.value(y))
                    mpc_solns_v.append(sol.value(V))
                    mpc_solns_theta.append(sol.value(theta))
                    mpc_solns_uv.append(sol.value(uv))
                    mpc_solns_delta.append(sol.value(delta))
                    mpc_solns_slacks.append(sol.value(obst_slack))

                    for k in range(len(ped_preds)):
                        
                        predictions_x = [x for [x,y] in ped_preds[k]]
                        predictions_y = [y for [x,y] in ped_preds[k]]

                        # print("Printing dists to ped")
                        for tau in range(1,T):
                            # print(math.sqrt((sol.value(x)[tau]-predictions_x[tau-1])**2+(sol.value(y)[tau]-predictions_y[tau-1])**2))
                            if use_d_values:
                                obst_dist_thresh = d_values[min(len(d_values)-1,tau-1)]
                            else:
                                obst_dist_thresh = tau_vals[tau-1]

                            # print(obst_dist_thresh)


                except Exception as e1:
                    ## get debug
                    mpc_solve_end_time = time.time()
                    feasible.append(0)

                    print("MPC failed")
                    print(e1)

                    print("x: " + str(opti.debug.value(x)))
                    print("y: " + str(opti.debug.value(y)))
                    print("V: " + str(opti.debug.value(V)))
                    print("theta: " + str(opti.debug.value(theta)))
                    print("uv: " + str(opti.debug.value(uv)))
                    print("delta: " + str(opti.debug.value(delta)))
                    print("slack: " + str(opti.debug.value(obst_slack)))

                    x_current = opti.debug.value(x)[1]
                    y_current = opti.debug.value(y)[1]
                    v_current = opti.debug.value(V)[1]
                    theta_current = opti.debug.value(theta)[1]
                    uv_current = opti.debug.value(uv)[0]
                    # delta_current = opti.debug.value(delta)[0]
                    delta_current = math.atan(opti.debug.value(delta)[0])

                    mpc_solns_x.append(opti.debug.value(x))
                    mpc_solns_y.append(opti.debug.value(y))
                    mpc_solns_v.append(opti.debug.value(V))
                    mpc_solns_theta.append(opti.debug.value(theta))
                    mpc_solns_uv.append(opti.debug.value(uv))
                    mpc_solns_delta.append(opti.debug.value(delta))
                    mpc_solns_slacks.append(opti.debug.value(obst_slack))

                    
                    # if prev_sol is not None:
                    #     delta_current = prev_sol.value(delta)[1] #0
                    #     uv_current = prev_sol.value(uv)[1] #-uv_max
                    # else:
                    #     delta_current = 0
                    #     uv_current = -uv_max

                    debug_x = opti.debug.value(x)
                    debug_y = opti.debug.value(y)
                    debug_V = opti.debug.value(V)
                    debug_theta = opti.debug.value(theta)
                    debug_uv = opti.debug.value(uv)
                    debug_delta = opti.debug.value(delta)


                    check_dynamics(debug_x,debug_y,debug_V,debug_theta,debug_uv,debug_delta,Delta,L)

                    for k in range(len(ped_preds)):
                        
                        predictions_x = [x for [x,y] in ped_preds[k]]
                        predictions_y = [y for [x,y] in ped_preds[k]]

                        # print("Printing dists to ped")
                        # for tau in range(1,T):
                        #     print(math.sqrt((debug_x[tau]-predictions_x[tau-1])**2+(debug_y[tau]-predictions_y[tau-1])**2))
                        #     if use_d_values:
                        #         obst_dist_thresh = d_values[min(len(d_values)-1,tau-1)]
                        #     else:
                        #         obst_dist_thresh = tau_vals[tau-1]

                        #     print(obst_dist_thresh)



                print("mpc total run time: " + str(mpc_solve_end_time - mpc_start_time))
                print("mpc solve run time: " + str(mpc_solve_end_time - mpc_solve_start_time))

                mpc_run_times.append(mpc_solve_end_time - mpc_start_time)
                mpc_solve_times.append(mpc_solve_end_time - mpc_solve_start_time)


                x_state.append(x_current)
                y_state.append(y_current)
                v_state.append(v_current)
                theta_state.append(theta_current)
                uv_state.append(uv_current)
                delta_state.append(delta_current)

                control = carla.VehicleControl()
                delta_carla_control = delta_current*(6/pi) * (1/5)
                uv_carla_control = uv_current/5
                if uv_current >= 0:   
                    control.throttle = uv_carla_control
                    control.steer= delta_carla_control
                    control.brake = 0
                else:
                    control.throttle = 0
                    control.steer= delta_carla_control
                    control.brake = -uv_carla_control/2

                    # control.throttle = uv_carla_control
                    # control.steer= delta_carla_control
                    # control.brake = 0
                    # control.reverse = True
                
                if vehicle.get_control().gear == 0:
                    control.gear = 1
                    control.manual_gear_shift = True
                vehicle.apply_control(control)
                print("Control: " + str(control))
                # print(v_current)
                # print(delta_current)

                time.sleep(0.5)
                
                if not args.sync:
                    print("Async. Waiting for tick.")
                    world.wait_for_tick()
                else:
                    print("Sync. Ticking world.")
                    num_tick = int(CARLA_HZ/MPC_HZ)
                    for _ in range(num_tick):

                        for ped in actor_list.filter('walker.pedestrian.*'):
                            if ped.id in ped_dict:
                                ped_loc = ped.get_location()
                                ped_dict[ped.id].append([ped_loc.x, ped_loc.y])
                            else:
                                ped_loc = ped.get_location()
                                ped_dict[ped.id] = [[ped_loc.x, ped_loc.y]]

                        for veh in actor_list.filter('vehicle.*'):
                            if veh.id == vehicle.id:
                                continue
                            if veh.id in veh_dict:
                                veh_loc = veh.get_location()
                                veh_dict[veh.id].append([veh_loc.x, veh_loc.y])
                            else:
                                veh_loc = veh.get_location()
                                veh_dict[veh.id] = [[veh_loc.x, veh_loc.y]]


                        world.tick()
                        time.sleep(0.1)
                        print("Prev vehicle control: " + str(vehicle.get_control()))
                        print("Carla vehicle state: " + str(vehicle.get_transform()))
                        print("Carla vehicle vel: " + str(vehicle.get_velocity()))
                    
                print("Desired MPC state: " + str([x_current, y_current, v_current, theta_current]))

            time.sleep(5)
        except Exception as e:
            print(str(e))        
        finally:

            print("All mpc run times: " + str(mpc_run_times))
            print("All mpc solve times: " + str(mpc_solve_times))

            print("x states: " + str(x_state[T:]))
            print("y states: " + str(y_state[T:]))
            print("theta states: " + str(theta_state[T:]))
            print("v states: " + str(v_state[T:]))
            print("uv states: " + str(uv_state[T:]))
            print("delta states: " + str(delta_state[T:]))
            # print("slacks: " + str(mpc_solns_slacks))
            # print("max slack: " + str(max(mpc_solns_slacks)))

            
            print("Saving obstacle positions and predictions")
            with open(obstacleDataFolder + "pedDict.pkl", 'wb') as f:
                pickle.dump(ped_dict,f)
            with open(obstacleDataFolder + "pedPredDict.pkl", 'wb') as f:
                pickle.dump(prediction_dict_ped,f)
            with open(obstacleDataFolder + "vehDict.pkl", 'wb') as f:
                pickle.dump(veh_dict,f)
            with open(obstacleDataFolder + "vehPredDict.pkl", 'wb') as f:
                pickle.dump(prediction_dict_veh,f)
            with open(obstacleDataFolder + "dvalues.pkl", 'wb') as f:
                pickle.dump(d_values,f)
            with open(obstacleDataFolder + "pedDictAtPredStep.pkl",'wb') as f:
                pickle.dump(ped_dict_at_pred_step,f)

            print("Saving car states")
            car_states_mpc = {}
            car_states_mpc["x"] = x_state[T:]
            car_states_mpc["y"] = y_state[T:]
            car_states_mpc["theta"] = theta_state[T:]
            car_states_mpc["v"] = v_state[T:]
            car_states_mpc["uv"] = uv_state[T:]
            car_states_mpc["delta"] = delta_state[T:]
            with open(carDataFolder + "carStatesMPC.pkl", 'wb') as f:
                pickle.dump(car_states_mpc,f)

            print("Saving car states")
            car_states = {}
            car_states["x"] = vehicle_x
            car_states["y"] = vehicle_y
            car_states["theta"] = vehicle_theta
            car_states["v"] = vehicle_v
            with open(carDataFolder + "carStates.pkl", 'wb') as f:
                pickle.dump(car_states,f)


            print("Plotting ped and predictions")


            for ped in actor_list.filter('walker.pedestrian.*'):
                if ped.id in prediction_dict_ped:
                    times_to_plot = len(prediction_dict_ped[ped.id])
                    break
            
            for t in range(times_to_plot):
                plt.clf()
                ax = plt.gca()

                tau_vals = [tau_base+tau*tau_inc for tau in range(1,T)]
                for ped in actor_list.filter('walker.pedestrian.*'):
                    ped_x = [x for [x,y] in ped_dict[ped.id]]
                    ped_y = [y for [x,y] in ped_dict[ped.id]]

                    pred_x = [x for [x,y] in prediction_dict_ped[ped.id][t]]
                    pred_y = [y for [x,y] in prediction_dict_ped[ped.id][t]]

                    ped_downsample_rate = int(CARLA_HZ/MPC_HZ)
                    # plt.plot(ped_x[::ped_downsample_rate],ped_y[::ped_downsample_rate], 'r*')
                    plt.plot(ped_x[num_tick_before_MPC+ped_downsample_rate*t],ped_y[num_tick_before_MPC+ped_downsample_rate*t], 'r*')
                    plt.plot(pred_x,pred_y, 'g.')

                    for k in range(len(pred_x)):
                        if use_d_values:
                            obst_dist_thresh = d_values[min(len(d_values)-1,k)]
                        else:
                            obst_dist_thresh = tau_vals[k]

                        circle = plt.Circle( (pred_x[k], pred_y[k]), obst_dist_thresh, color='g',fill=False)
                        ax.add_patch(circle)

                
                for veh in actor_list.filter('vehicle.*'):
                    if veh.id == vehicle.id:
                        continue
                    veh_x = [x for [x,y] in veh_dict[veh.id]]
                    veh_y = [y for [x,y] in veh_dict[veh.id]]

                    pred_x = [x for [x,y] in prediction_dict_veh[veh.id][t]]
                    pred_y = [y for [x,y] in prediction_dict_veh[veh.id][t]]

                    plt.plot(veh_x,veh_y, 'b*')
                    plt.plot(pred_x,pred_y, 'y.')

                    for k in range(len(pred_x)):
                        if use_d_values:
                            obst_dist_thresh = d_values[min(len(d_values)-1,k)]
                        else:
                            obst_dist_thresh = tau_vals[k]


                        circle = plt.Circle( (pred_x[k], pred_y[k]), obst_dist_thresh, color='y',fill=False)
                        ax.add_patch(circle)


                # print(mpc_solns_x)
                if t < len(mpc_solns_x):
                    
                    if not feasible[t]:
                        print("MPC invalid. Printing solvers state at last iteration")
                    print("Step " + str(t) + " MPC x: " + str(mpc_solns_x[t]))
                    print("Step " + str(t) + " MPC y: " + str(mpc_solns_y[t]))
                    print("Step " + str(t) + " MPC v: " + str(mpc_solns_v[t]))
                    
                    if feasible[t]:
                        plt.plot(mpc_solns_x[t],mpc_solns_y[t], 'b-')
                    else:
                        plt.plot(mpc_solns_x[t],mpc_solns_y[t], 'm-')

                    if t>=1 and plotMPCSolns:
                        plt.plot(mpc_solns_x[t-1],mpc_solns_y[t-1], 'y.',markersize=6)


                # print("Plotting actual car states")
                if t < len(vehicle_xs):
                    plt.plot(vehicle_xs[t],vehicle_ys[t], 'k.',markersize=16)
                # plt.title("LSTM Predictions MPC Step " + str(t) + ", slack=" + str(mpc_solns_slacks[t]))
                plt.title("LSTM Predictions MPC Step " + str(t))
                plt.xlim([-70,-20])
                plt.ylim([-20, 50])
                plt.savefig(lstmImageFolder + "mpcStep%04d.png" % t)

                if t>=1:
                    print("Step " + str(t))
                    print("Prev MPC step desired state (x,y,v,theta): " + str([mpc_solns_x[t-1][1],mpc_solns_y[t-1][1],mpc_solns_v[t-1][1],mpc_solns_theta[t-1][1]]))
                    print("Current vehicle state (x,y,v,theta): " + str([vehicle_xs[t],vehicle_ys[t],vehicle_vels[t],vehicle_thetas[t]]))



            # if args.sync and synchronous_master:
            #     settings = world.get_settings()
            #     settings.synchronous_mode = False
            #     settings.fixed_delta_seconds = None
            #     world.apply_settings(settings)

            actor_list = world.get_actors()

            print("Actor list")
            print(actor_list)

            print('\ndestroying %d cameras' % len(camera_list))
            # client.apply_batch([carla.command.DestroyActor(x) for x in camera_list])

            for c in camera_list:
                c.destroy()
            
            # client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])


            ## destroy all cars
            for v in actor_list.filter('vehicle.*'):
                v.destroy()

            # client.apply_batch([carla.command.DestroyActor(x) for x in actor_list.filter('vehicle.*')])

            ## destory all pedestrians
            for p in actor_list.filter('pedestrian.*'):
                p.destroy()

            # client.apply_batch([carla.command.DestroyActor(x) for x in actor_list.filter('pedestrian.*')])

            ## destroy all cameras 
            # client.apply_batch([carla.command.DestroyActor(x) for x in actor_list.filter('sensor.camera.*')])

            world.tick()
            time.sleep(0.5)

            print("Actor list")
            print(actor_list)


if __name__ == '__main__':
	main()





