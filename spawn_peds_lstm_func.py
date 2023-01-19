import glob
import os
import sys
import time
import subprocess

sys.path.append('../carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg')

import carla
import argparse
import logging
import random
from time import ctime
import math
import numpy as np
    



def spawn_peds(client,args):

    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)

    settings = world.get_settings()
    print("sync: " + str(settings.synchronous_mode))
    assert settings.synchronous_mode
    
    # Pedestrians
    blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")

    # right-top
    spawn_point1 = carla.Transform(carla.Location(x=-26.72213135+round(random.uniform(-3, 3), 3), y=5.19141479+round(random.uniform(-2, 2), 3), z=0.2), carla.Rotation(roll=0, pitch=-90.0, yaw=0))
    # left-top
    spawn_point2 = carla.Transform(carla.Location(x=-67.90557129+round(random.uniform(-2, 2), 3), y=5.18553772+round(random.uniform(-1, 1), 3), z=0.2), carla.Rotation(roll=274.444824, pitch=-90.0, yaw=630.0))
    # left-bottm
    spawn_point3 = carla.Transform(carla.Location(x=-67.63589355+round(random.uniform(-2.5, 2.5), 3), y=34.29434082+round(random.uniform(-0.6, 0.6), 3), z=0.2), carla.Rotation(roll=87.13102, pitch=-90.0, yaw=-360))
    # right-bottm
    spawn_point4 = carla.Transform(carla.Location(x=-24.58656738+round(random.uniform(-2, 2), 3), y=35.17018555+round(random.uniform(-1.7, 1.7), 3), z=0.2), carla.Rotation(roll=-348.64447, pitch=-90.0, yaw=350.030304))
    number_of_spawn_points = 4

    # @todo cannot import these directly.
    SpawnActor = carla.command.SpawnActor
    #SetAutopilot = carla.command.SetAutopilot
    #FutureActor = carla.command.FutureActor

    walkers_list = []
    all_id = []

    # -------------
    # Spawn Walkers
    # -------------
    # some settings
    percentagePedestriansRunning = 4.0      # how many pedestrians will run
    percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
    # 1. take all the random locations to spawn
    spawn_points = []

    spawn_points.append(spawn_point1)
    spawn_points.append(spawn_point2)
    spawn_points.append(spawn_point3)
    spawn_points.append(spawn_point4)
    
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for i in range(4):
        # if i >= 2:
        #     continue
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_points[i]))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, spawn_points[i], walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put altogether the walkers and controllers id to get the objects from their id

    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # wait for a tick to ensure client receives the last transform of the walkers we have just created
    world.tick()

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    ped_points = []
    # right-top (done)
    ped_destination1 = carla.Location(x=-121.37990234+round(random.uniform(-1.1, 1.1), 3), y=2.65673645+round(random.uniform(-1, 1), 3), z=0.2)
    # left-top
    ped_destination2 = carla.Location(x=84.22938477+round(random.uniform(-3, 3), 3), y=5.21222656+round(random.uniform(-2, 2), 3), z=0.2)
    # left-bottom (done)
    ped_destination3 = carla.Location(x=84.22938477+round(random.uniform(-2.5, 2.5), 3), y=36.1528833+round(random.uniform(-2, 2), 3), z=0.2)
    # right-bottom (done)
    ped_destination4 = carla.Location(x=-122.34274414+round(random.uniform(-1.7, 1.7), 3), y=39.35759277+round(random.uniform(-1.6, 1.6), 3), z=0.2)

    ped_points.append(ped_destination1)
    ped_points.append(ped_destination2)
    ped_points.append(ped_destination3)
    ped_points.append(ped_destination4)
    
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(ped_points[int(i/2)])
        # max speed
        all_actors[i].set_max_speed(2+round(random.uniform(-1, 0.5), 3))

    print("Ped spawns")
    for p in spawn_points:
        print(p.location)




    return



def reset_peds(client,args):
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)
    traffic_manager.set_global_distance_to_leading_vehicle(1.0)

    settings = world.get_settings()
    print("sync: " + str(settings.synchronous_mode))
    assert settings.synchronous_mode

    actor_list = world.get_actors()
    print(actor_list.filter('walker.pedestrian.*'))
    assert len(actor_list.filter('walker.pedestrian.*')) <= 4


    # right-top
    spawn_point1 = carla.Transform(carla.Location(x=-26.72213135+round(random.uniform(-3, 3), 3), y=5.19141479+round(random.uniform(-2, 2), 3), z=0.2), carla.Rotation(roll=0, pitch=-90.0, yaw=0))
    # left-top
    spawn_point2 = carla.Transform(carla.Location(x=-67.90557129+round(random.uniform(-2, 2), 3), y=5.18553772+round(random.uniform(-1, 1), 3), z=0.2), carla.Rotation(roll=274.444824, pitch=-90.0, yaw=630.0))
    # left-bottm
    spawn_point3 = carla.Transform(carla.Location(x=-67.63589355+round(random.uniform(-2.5, 2.5), 3), y=34.29434082+round(random.uniform(-0.6, 0.6), 3), z=0.2), carla.Rotation(roll=87.13102, pitch=-90.0, yaw=-360))
    # right-bottm
    spawn_point4 = carla.Transform(carla.Location(x=-24.58656738+round(random.uniform(-2, 2), 3), y=35.17018555+round(random.uniform(-1.7, 1.7), 3), z=0.2), carla.Rotation(roll=-348.64447, pitch=-90.0, yaw=350.030304))
    spawn_points = [spawn_point1,spawn_point2,spawn_point3,spawn_point4]

    # -------------
    # Spawn Walkers
    # -------------
    # some settings
    percentagePedestriansRunning = 4.0      # how many pedestrians will run
    percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
    # 1. take all the random locations to spawn

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    ped_points = []
    # right-top (done)
    ped_destination1 = carla.Location(x=-121.37990234+round(random.uniform(-1.1, 1.1), 3), y=2.65673645+round(random.uniform(-1, 1), 3), z=0.2)
    # left-top
    ped_destination2 = carla.Location(x=84.22938477+round(random.uniform(-3, 3), 3), y=5.21222656+round(random.uniform(-2, 2), 3), z=0.2)
    # left-bottom (done)
    ped_destination3 = carla.Location(x=84.22938477+round(random.uniform(-2.5, 2.5), 3), y=36.1528833+round(random.uniform(-2, 2), 3), z=0.2)
    # right-bottom (done)
    ped_destination4 = carla.Location(x=-122.34274414+round(random.uniform(-1.7, 1.7), 3), y=39.35759277+round(random.uniform(-1.6, 1.6), 3), z=0.2)

    ped_points.append(ped_destination1)
    ped_points.append(ped_destination2)
    ped_points.append(ped_destination3)
    ped_points.append(ped_destination4)

    
    ## Stop and destroy all ped controllers
    for con in actor_list.filter('controller.ai.walker'):
        con.stop()
        if con.destroy():
            print("Destroyed walker controller " + str(con.id))
        else:
            print("Failed to destroy walker controller " + str(con.id))

 
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    blueprintsWalkers = world.get_blueprint_library().filter("walker.pedestrian.*")
    # for i,ped in enumerate(actor_list.filter('pedestrian.*')):
    #     ped.set_transform(spawn_points[i])
    #     ped_con = world.spawn_actor(walker_controller_bp,spawn_points[i],ped)
    #     ped_con.start()
    #     ped_con.go_to_location(ped_points[i])
    #     ped_con.set_max_speed(2+round(random.uniform(-1, 0.5), 3))

    all_peds = actor_list.filter('walker.pedestrian.*')
    for i in range(len(spawn_points)):
        if i < len(all_peds):
            ped = all_peds[i]
            ped.set_transform(spawn_points[i])
        else:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            ped = world.spawn_actor(walker_bp, spawn_points[i])
        world.tick()
        ped_con = world.spawn_actor(walker_controller_bp,spawn_points[i],ped)
        ped_con.start()
        ped_con.go_to_location(ped_points[i])
        ped_con.set_max_speed(2+round(random.uniform(-1, 0.5), 3))

    print("Ped spawns")
    for p in spawn_points:
        print(p.location)


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
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    vehicles_list = []
    walkers_list = []
    all_id = []
    frame = 0
    client = carla.Client(args.host, args.port)
    client.set_timeout(45.0)
    synchronous_master = False

    spawn_vehicles = False
    spawn_walkers = True
    
    try:
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)

        if args.sync:
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            if not settings.synchronous_mode:
                synchronous_master = True
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = 0.05
                world.apply_settings(settings)
            else:
                synchronous_master = False

        if spawn_vehicles:
            # Vehicles
            blueprint_library = world.get_blueprint_library()
            tesla_model3_1 = blueprint_library.filter('model3')[0]
            tesla_model3_2 = blueprint_library.filter('model3')[0]
            tesla_model3_3 = blueprint_library.filter('model3')[0]
            tesla_model3_4 = blueprint_library.filter('model3')[0]

            # Random Initial Position of Vehicles
            spawn_1 = []
            spawn_1_rnd1 = carla.Transform(carla.Location(x=-88.67606445, y=24.43763428, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=0.159198))
            spawn_1_rnd2 = carla.Transform(carla.Location(x=-78.0661084, y=24.46711426, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=0.159198))
            spawn_1_rnd3 = carla.Transform(carla.Location(x=-66.04484863, y=24.46711426, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=0.159198))
            spawn_1_rnd4 = carla.Transform(carla.Location(x=-88.68579102, y=27.93761963, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=0.159198))
            spawn_1_rnd5 = carla.Transform(carla.Location(x=-78.07583496, y=27.96709961, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=0.159198))
            spawn_1_rnd6 = carla.Transform(carla.Location(x=-66.0545752, y=27.96709961, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=0.159198))
            spawn_1.append(spawn_1_rnd1)
            spawn_1.append(spawn_1_rnd2)
            spawn_1.append(spawn_1_rnd3)
            spawn_1.append(spawn_1_rnd4)
            spawn_1.append(spawn_1_rnd5)
            spawn_1.append(spawn_1_rnd6)

            spawn_2 = []
            spawn_2_rnd1 = carla.Transform(carla.Location(x=-52.31998047, y=-4.78522583, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=89.83876))
            spawn_2_rnd2 = carla.Transform(carla.Location(x=-48.83995117, y=-17.21319946, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=90.432327))
            spawn_2_rnd3 = carla.Transform(carla.Location(x=-48.83995117, y=-32.03491943, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=90.432327))
            spawn_2_rnd4 = carla.Transform(carla.Location(x=-54.31998047, y=-4.78522583, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=89.83876))
            spawn_2_rnd5 = carla.Transform(carla.Location(x=-52.33985352, y=-17.23960938, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=90.432327))
            spawn_2_rnd6 = carla.Transform(carla.Location(x=-52.33985352, y=-32.06133057, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=90.432327))
            spawn_2.append(spawn_2_rnd1)
            spawn_2.append(spawn_2_rnd2)
            spawn_2.append(spawn_2_rnd3)
            spawn_2.append(spawn_2_rnd4)
            spawn_2.append(spawn_2_rnd5)
            spawn_2.append(spawn_2_rnd6)

            spawn_3 = []
            spawn_3_rnd1 = carla.Transform(carla.Location(x=-17.11512695, y=16.75744385, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=-179.84079))
            spawn_3_rnd2 = carla.Transform(carla.Location(x=-17.10540283, y=13.25745728, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=-179.84079))
            spawn_3.append(spawn_3_rnd1)
            spawn_3.append(spawn_3_rnd2)

            spawn_4 = []
            spawn_4_rnd1 = carla.Transform(carla.Location(x=-45.16095703, y=51.71539062, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=-90.161217))
            spawn_4_rnd2 = carla.Transform(carla.Location(x=-41.6609668, y=51.70554199, z=0.6), carla.Rotation(roll=0, pitch=0, yaw=-90.161217))
            spawn_4.append(spawn_4_rnd1)
            spawn_4.append(spawn_4_rnd2)
            
            vehicle1 = world.spawn_actor(tesla_model3_1, spawn_1[random.randint(0, 5)])
            vehicle2 = world.spawn_actor(tesla_model3_2, spawn_2[random.randint(0, 5)])
            vehicle3 = world.spawn_actor(tesla_model3_3, spawn_3[random.randint(0, 1)])
            vehicle4 = world.spawn_actor(tesla_model3_4, spawn_4[random.randint(0, 1)])
            
            vehicle1.set_autopilot(True, traffic_manager.get_port())
            vehicle2.set_autopilot(True, traffic_manager.get_port())
            vehicle3.set_autopilot(True, traffic_manager.get_port())
            vehicle4.set_autopilot(True, traffic_manager.get_port())
            vehicles_list.append(vehicle1)
            vehicles_list.append(vehicle2)
            vehicles_list.append(vehicle3)
            vehicles_list.append(vehicle4)

        # # Ego Vehicle
        # ego_vehicle = blueprint_library.filter("model3")[0]
        # spawn_point = world.get_map().get_spawn_points()[46]
        # world.debug.draw_string(spawn_point.location, 'O', draw_shadow=False,
        #                         color=carla.Color(r=255, g=0, b=0),
        #                         persistent_lines = True)

        # ego_v = world.spawn_actor(ego_vehicle, spawn_point)

        if spawn_walkers:
            # Pedestrians
            blueprints = world.get_blueprint_library().filter(args.filterv)
            blueprintsWalkers = world.get_blueprint_library().filter(args.filterw)

            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]

            # right-top
            spawn_point1 = carla.Transform(carla.Location(x=-29.41560547+round(random.uniform(-2.4, 2.4), 3), y=5.29434082+round(random.uniform(-0.5, 0.5), 3), z=0.2), carla.Rotation(roll=0, pitch=-90.0, yaw=0))
            # left-top
            spawn_point2 = carla.Transform(carla.Location(x=-61.21560547+round(random.uniform(-1.7, 1.7), 3), y=2.79434082+round(random.uniform(-0.5, 0.5), 3), z=0.2), carla.Rotation(roll=274.444824, pitch=-90.0, yaw=630.0))
            # left-bottm
            spawn_point3 = carla.Transform(carla.Location(x=-64.71560547+round(random.uniform(-2.5, 2.5), 3), y=34.29434082+round(random.uniform(-0.5, 0.5), 3), z=0.2), carla.Rotation(roll=87.13102, pitch=-90.0, yaw=-360))
            # right-bottm
            spawn_point4 = carla.Transform(carla.Location(x=-32.91560547+round(random.uniform(-0.5, 0.5), 3), y=42.69434082+round(random.uniform(-2.9, 2.9), 3), z=0.2), carla.Rotation(roll=-348.64447, pitch=-90.0, yaw=350.030304))
            number_of_spawn_points = 4

            # @todo cannot import these directly.
            SpawnActor = carla.command.SpawnActor
            #SetAutopilot = carla.command.SetAutopilot
            #FutureActor = carla.command.FutureActor

            # -------------
            # Spawn Walkers
            # -------------
            # some settings
            percentagePedestriansRunning = 4.0      # how many pedestrians will run
            percentagePedestriansCrossing = 0.0     # how many pedestrians will walk through the road
            # 1. take all the random locations to spawn
            spawn_points = []

            spawn_points.append(spawn_point1)
            spawn_points.append(spawn_point2)
            spawn_points.append(spawn_point3)
            spawn_points.append(spawn_point4)
            
            # 2. we spawn the walker object
            batch = []
            walker_speed = []
            for i in range(4):
                # if i >= 2:
                #     continue
                walker_bp = random.choice(blueprintsWalkers)
                # set as not invincible
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
                # set the max speed
                if walker_bp.has_attribute('speed'):
                    if (random.random() > percentagePedestriansRunning):
                        # walking
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                    else:
                        # running
                        walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
                else:
                    print("Walker has no speed")
                    walker_speed.append(0.0)
                batch.append(SpawnActor(walker_bp, spawn_points[i]))
            results = client.apply_batch_sync(batch, True)
            walker_speed2 = []
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list.append({"id": results[i].actor_id})
                    walker_speed2.append(walker_speed[i])
            walker_speed = walker_speed2
            # 3. we spawn the walker controller
            batch = []
            walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            for i in range(len(walkers_list)):
                batch.append(SpawnActor(walker_controller_bp, spawn_points[i], walkers_list[i]["id"]))
            results = client.apply_batch_sync(batch, True)
            for i in range(len(results)):
                if results[i].error:
                    logging.error(results[i].error)
                else:
                    walkers_list[i]["con"] = results[i].actor_id
            # 4. we put altogether the walkers and controllers id to get the objects from their id

            for i in range(len(walkers_list)):
                all_id.append(walkers_list[i]["con"])
                all_id.append(walkers_list[i]["id"])
            all_actors = world.get_actors(all_id)

            # wait for a tick to ensure client receives the last transform of the walkers we have just created
            if not args.sync or not synchronous_master:
                world.wait_for_tick()
            else:
                world.tick()

            # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
            # set how many pedestrians can cross the road
            world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
            ped_points = []
            # right-top (done)
            ped_destination1 = carla.Location(x=-90.97136719+round(random.uniform(-2.5, 2.5), 3), y=4.8528833+round(random.uniform(-2.5, 2.5), 3), z=0.2)
            # left-top
            ped_destination2 = carla.Location(x=9.02863281+round(random.uniform(-3, 3), 3), y=5.21222656+round(random.uniform(-2, 2), 3), z=0.2)
            # left-bottom (done)
            ped_destination3 = carla.Location(x=7.62863281+round(random.uniform(-2.5, 2.5), 3), y=36.1528833+round(random.uniform(-2, 2), 3), z=0.2)
            # right-bottom (done)
            ped_destination4 = carla.Location(x=-90.47136719+round(random.uniform(-2, 2), 3), y=34.44422852+round(random.uniform(-1.2, 1.2), 3), z=0.2)

            ped_points.append(ped_destination1)
            ped_points.append(ped_destination2)
            ped_points.append(ped_destination3)
            ped_points.append(ped_destination4)
            
            for i in range(0, len(all_id), 2):
                # start walker
                all_actors[i].start()
                # set walk to random point
                all_actors[i].go_to_location(ped_points[int(i/2)])
                # max speed
                all_actors[i].set_max_speed(2+round(random.uniform(-1, 0.5), 3))

        if spawn_vehicles:
            print("Vehicle spawns")
            for veh in vehicles_list:
                print(veh.get_location())

        if spawn_walkers:
            print("Ped spawns")
            for p in spawn_points:
                print(p.location)

        while True:
            if args.sync and synchronous_master:
                world.tick()
            else:
                world.wait_for_tick()


    finally:
        if args.sync and synchronous_master:
            settings = world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.apply_settings(settings)
            
        if spawn_vehicles:
            print('\ndestroying %d vehicles' % len(vehicles_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
        
        if spawn_walkers:
            # stop walker controllers (list is [controller, actor, controller, actor ...])
            print('\ndestroying %d walkers' % len(walkers_list))
            client.apply_batch([carla.command.DestroyActor(x) for x in all_id])
            
        time.sleep(0.5)

if __name__ == '__main__':
	main()
