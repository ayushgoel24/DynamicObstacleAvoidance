import carla
import argparse

"""
Parses the input arguments of the file.
"""
def parseArguments():
    argparser.add_argument( '--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)' )
    argparser.add_argument( '-n', '--number-of-scenes', metavar='N', default=10, type=int, )
    argparser.add_argument( '-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)' )
    argparser.add_argument( '--filterv', metavar='PATTERN', default='vehicle.*', help='vehicles filter (default: "vehicle.*")' )
    argparser.add_argument( '--filterw', metavar='PATTERN', default='walker.pedestrian.*', help='pedestrians filter (default: "walker.pedestrian.*")' )
    argparser.add_argument( '--tm-port', metavar='P', default=8000, type=int, help='port to communicate with TM (default: 8000)' )
    argparser.add_argument( '--sync', action='store_true', help='Synchronous mode execution' )
    argparser.add_argument( '--hybrid', action='store_true', help='Enanble' )
    argparser.add_argument( '--timeout', default=45, type=int, help='carla timeout (default: 45)' )

if __name__ == "__main__":

    argparser = argparse.ArgumentParser( description=__doc__ )
    args = argparser.parse_args()

    carla_client = carla.Client( args.host, args.port )
    carla_client.set_timeout( args.timeout )

    try:
        # Returns the world object currently active in the simulation
        # https://carla.readthedocs.io/en/latest/python_api/#carla.World
        carla_world = carla_client.get_world()

        # Returns an instance of the traffic manager related to the specified port
        # https://carla.readthedocs.io/en/latest/python_api/#carla.TrafficManager
        carla_traffic_manager = carla_client.get_trafficmanager( args.tm_port )
        carla_traffic_manager.set_global_distance_to_leading_vehicle( 1.0 )
        
        if args.hybrid:
            carla_traffic_manager.set_hybrid_physics_mode( True )

        if args.sync:
            # Returns an object containing some data about the simulation
            # https://carla.readthedocs.io/en/latest/python_api/#carla.WorldSettings
            carla_world_settings = carla_world.get_settings()

            carla_traffic_manager.set_synchronous_mode( True )
            if not carla_world_settings.synchronous_mode:
                synchronous_master = True
                carla_world_settings.synchronous_mode = True
                carla_world_settings.fixed_delta_seconds = 0.05
                carla_world.apply_settings(carla_world_settings)
            else:
                synchronous_master = False

    finally:
        pass