# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla
from shapely.geometry import Polygon

from local_planner import LocalPlanner, RoadOption
from global_route_planner import GlobalRoutePlanner
from misc import (get_speed, is_within_distance,
                               get_trafficlight_trigger_location,
                               compute_distance,custom_distance)

class BasicAgent(object):
    """
    BasicAgent implements an agent that navigates the scene.
    This agent respects traffic lights and other vehicles, but ignores stop signs.
    It has several functions available to specify the route that the agent must follow,
    as well as to change its parameters in case a different driving mode is desired.
    """

    def __init__(self, vehicle, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Initialization the agent paramters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.

        """
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        if map_inst:
            if isinstance(map_inst, carla.Map):
                self._map = map_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._map = self._world.get_map()
        else:
            self._map = self._world.get_map()
        self._last_traffic_light = None

        # Base parameters
        self._ignore_traffic_lights = False
        self._ignore_stop_signs = False
        self._ignore_vehicles = False
        self._use_bbs_detection = False
        self._vehicle_l2 = []               # veicoli filtrati all'occorrenza dalla lista di tutti i veicoli presenti nel mondo
        self._target_speed = 5.0
        self._sampling_resolution = 2.0
        self._base_tlight_threshold = 5.0   # metri
        self._base_vehicle_threshold = 7.0  # metri
        self._base_stop_threshold = 5.0     # metri
        self._signal_stop_distance = 17     # distanza visibilità stop
        self._speed_ratio = 1
        self._max_brake = 1
        self._offset = 0
        
        self._speed_limit = self._vehicle.get_speed_limit()
        # Change parameters according to the dictionary
        if 'target_speed' in opt_dict:
            self._target_speed = opt_dict['target_speed']
        if 'ignore_traffic_lights' in opt_dict:
            self._ignore_traffic_lights = opt_dict['ignore_traffic_lights']
        if 'ignore_stop_signs' in opt_dict:
            self._ignore_stop_signs = opt_dict['ignore_stop_signs']
        if 'ignore_vehicles' in opt_dict:
            self._ignore_vehicles = opt_dict['ignore_vehicles']
        if 'use_bbs_detection' in opt_dict:
            self._use_bbs_detection = opt_dict['use_bbs_detection']
        if 'sampling_resolution' in opt_dict:
            self._sampling_resolution = opt_dict['sampling_resolution']
        if 'base_tlight_threshold' in opt_dict:
            self._base_tlight_threshold = opt_dict['base_tlight_threshold']
        if 'base_vehicle_threshold' in opt_dict:
            self._base_vehicle_threshold = opt_dict['base_vehicle_threshold']
        if 'detection_speed_ratio' in opt_dict:
            self._speed_ratio = opt_dict['detection_speed_ratio']
        if 'max_brake' in opt_dict:
            self._max_brake = opt_dict['max_brake']
        if 'offset' in opt_dict:
            self._offset = opt_dict['offset']
        
        # Initialize the planners
        self._local_planner = LocalPlanner(self._vehicle, opt_dict=opt_dict, map_inst=self._map)
        if grp_inst:
            if isinstance(grp_inst, GlobalRoutePlanner):
                self._global_planner = grp_inst
            else:
                print("Warning: Ignoring the given map as it is not a 'carla.Map'")
                self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)
        else:
            self._global_planner = GlobalRoutePlanner(self._map, self._sampling_resolution)

        # Get the static elements of the scene
        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map = {}  # Dictionary mapping a traffic light to a wp corrspoing to its trigger volume location
        self._stop_map = {} # Dizionario che mappa gli stop

    def add_emergency_stop(self, control):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def set_target_speed(self, speed):
        """
        Changes the target speed of the agent
            :param speed (float): target speed in Km/h
        """
        self._target_speed = speed
        self._local_planner.set_speed(speed)

    def follow_speed_limits(self, value=True):
        """
        If active, the agent will dynamically change the target speed according to the speed limits

            :param value (bool): whether or not to activate this behavior
        """
        self._local_planner.follow_speed_limits(value)

    def get_local_planner(self):
        """Get method for protected member local planner"""
        return self._local_planner

    def get_global_planner(self):
        """Get method for protected member local planner"""
        return self._global_planner

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def run_step(self):
        """Execute one step of navigation."""
        print("basic_agent")
        hazard_detected = False

        #####
        #  Retrieve all relevant actors
        #####
        # Basic Agent :
        vehicle_list = self._world.get_actors().filter("*vehicle*")
        ### 

        vehicle_speed = get_speed(self._vehicle) / 3.6

        # Check for possible vehicle obstacles
        max_vehicle_distance = self._base_vehicle_threshold + self._speed_ratio * vehicle_speed
        affected_by_vehicle, _, _ = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:
            hazard_detected = True

        # Check if the vehicle is affected by a red traffic light
        max_tlight_distance = self._base_tlight_threshold + self._speed_ratio * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(self._lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_detected = True

        control = self._local_planner.run_step()
        if hazard_detected:
            control = self.add_emergency_stop(control)

        return control
    
    def reset(self):
        pass

    def done(self):
        """Check whether the agent has reached its destination."""
        return self._local_planner.done()

    def ignore_traffic_lights(self, active=True):
        """(De)activates the checks for traffic lights"""
        self._ignore_traffic_lights = active

    def ignore_stop_signs(self, active=True):
        """(De)activates the checks for stop signs"""
        self._ignore_stop_signs = active

    def ignore_vehicles(self, active=True):
        """(De)activates the checks for vehicles"""
        self._ignore_vehicles = active

    def lane_change(self, direction, same_lane_time=0, other_lane_time=0, lane_change_time=2):
        """
        Changes the path so that the vehicle performs a lane change.
        Use 'direction' to specify either a 'left' or 'right' lane change,
        and the other 3 fine tune the maneuver
        """
        speed = self._vehicle.get_velocity().length()
        path = self._generate_lane_change_path(
            self._map.get_waypoint(self._vehicle.get_location()),
            direction,
            same_lane_time * speed,
            other_lane_time * speed,
            lane_change_time * speed,
            False,
            1,
            self._sampling_resolution
        )
        if not path:
            print("WARNING: Ignoring the lane change as no path was found")

        self.set_global_plan(path)

    def _affected_by_traffic_light(self, lights_list=None, max_distance=None):
        """
        Method to check if there is a red light affecting the vehicle.

            :param lights_list (list of carla.TrafficLight): list containing TrafficLight objects.
                If None, all traffic lights in the scene are used
            :param max_distance (float): max distance for traffic lights to be considered relevant.
                If None, the base threshold value is used
        """
        if self._ignore_traffic_lights:
            return (False, None)

        if not lights_list:
            lights_list = self._world.get_actors().filter("*traffic_light*")

        if not max_distance:
            max_distance = self._base_tlight_threshold

        if self._last_traffic_light:
            if self._last_traffic_light.state != carla.TrafficLightState.Red:
                self._last_traffic_light = None
            else:
                return (True, self._last_traffic_light)

        ego_vehicle_location = self._vehicle.get_location()
        ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)

        for traffic_light in lights_list:
            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_location = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = self._map.get_waypoint(trigger_location)
                self._lights_map[traffic_light.id] = trigger_wp

            if trigger_wp.transform.location.distance(ego_vehicle_location) > max_distance:
                continue

            if trigger_wp.road_id != ego_vehicle_waypoint.road_id:
                continue

            ve_dir = ego_vehicle_waypoint.transform.get_forward_vector()
            wp_dir = trigger_wp.transform.get_forward_vector()
            dot_ve_wp = ve_dir.x * wp_dir.x + ve_dir.y * wp_dir.y + ve_dir.z * wp_dir.z

            if dot_ve_wp < 0:
                continue

            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            if is_within_distance(trigger_wp.transform, self._vehicle.get_transform(), max_distance, [0, 90]):
                self._last_traffic_light = traffic_light
                return (True, traffic_light)

        return (False, None)

    def _vehicle_obstacle_detected(self, vehicle_list=None, max_distance=None, lane_offset=0, type_obstacle = None, opposite_direction = None):
        """
        Method to check if there is a vehicle in front of the agent blocking its path.

            :param vehicle_list (list of carla.Vehicle): list contatining vehicle objects.
                If None, all vehicle in the scene are used
            :param max_distance: max freespace to check for obstacles.
                If None, the base threshold value is used
        """

        target_list=[] # Initialize empty list for potential obstacles

        if self._ignore_vehicles:
            return (False, None, -1)
        
        ego_transform = self._vehicle.get_transform()
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )
        
        if not vehicle_list:
            target_list=self.obstacles_list(type_obstacle,opposite_direction, ego_front_transform, ego_wpt)
            if type_obstacle == "walker" and  len(target_list)>0:
                return (True, target_list[0], 0, 0) # Return if a pedestrian is detected
                            
        if not max_distance:
            max_distance = self._base_vehicle_threshold

        # Get the right offset for lane
        if ego_wpt.lane_id < 0 and lane_offset != 0:
            lane_offset *= -1

    
        if len(target_list) > 0:
            for target in target_list: # Iterate over all detected obstacles
                target_transform = target.get_transform()
                target_wpt = self._map.get_waypoint(target_transform.location, lane_type=carla.LaneType.Any)

               # CASE I: Simplified version for outside junctions
                if not ego_wpt.is_junction:
                    # Consider vehicles on same or adjacent lane
                    if target_wpt.lane_id*ego_wpt.lane_id >= 0  and target_wpt.lane_id in [-2, -1, 0, 1, 2]:
                        target_forward_vector = target_transform.get_forward_vector() # Get forward vector of obstacle
                        target_extent = target.bounding_box.extent.x  # Get half-length of obstacle
                        target_rear_transform = target_transform # Create rear reference
                        target_rear_transform.location -= carla.Location(
                        x=target_extent * target_forward_vector.x, # Move location backward in x
                        y=target_extent * target_forward_vector.y, # Move location backward in y
                        )
                        # Return obstacle info if it is close enough
                        if custom_distance(self._vehicle,target) < max_distance:
                            return (True, target, compute_distance(target_transform.location, ego_transform.location), target_list)
                
                # Case II: both ego and obstacle are inside a junction       
                elif ego_wpt.is_junction and ego_wpt.next(1)[0].is_junction and target_wpt.is_junction:
                    target_forward_vector = target_transform.get_forward_vector()
                    target_extent = target.bounding_box.extent.x
                    target_front_transform = target_transform
                    target_front_transform.location += carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                    )
                    # Return if vehicle in junction is close and moving
                    if custom_distance(self._vehicle, target) < max_distance and get_speed(target) > 0.5:
                        return (True, target, compute_distance(target_transform.location, ego_transform.location), target_list)
                
                # Case III: ego is exiting a junction
                elif ego_wpt.is_junction and not ego_wpt.next(6)[0].is_junction:
                    target_forward_vector = target_transform.get_forward_vector()
                    target_extent = target.bounding_box.extent.x
                    target_front_transform = target_transform
                    target_front_transform.location += carla.Location(
                    x=target_extent * target_forward_vector.x,
                    y=target_extent * target_forward_vector.y,
                    )
                    # Obstacle is in same outgoing lane and close
                    if custom_distance(self._vehicle, target) < max_distance and target_wpt.next(8)[0].lane_id == ego_wpt.next(6)[0].lane_id:
                            self._local_planner.set_speed(17) # Set target speed to 17
                            self._control= self._local_planner.run_step() # Apply local planner control
                            return (True, target, compute_distance(target_transform.location, ego_transform.location), target_list)

                # Waypoints aren't reliable, check the proximity of the vehicle to the route
                else:
                    route_bb = []
                    ego_location = ego_transform.location
                    extent_y = self._vehicle.bounding_box.extent.y
                    r_vec = ego_transform.get_right_vector()
                    p1 = ego_location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                    p2 = ego_location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                    route_bb.append([p1.x, p1.y, p1.z])
                    route_bb.append([p2.x, p2.y, p2.z])

                    for wp, _ in self._local_planner.get_plan():
                        
                        if ego_location.distance(wp.transform.location) > max_distance:
                            break

                        r_vec = wp.transform.get_right_vector()
                        p1 = wp.transform.location + carla.Location(extent_y * r_vec.x, extent_y * r_vec.y)
                        p2 = wp.transform.location + carla.Location(-extent_y * r_vec.x, -extent_y * r_vec.y)
                        route_bb.append([p1.x, p1.y, p1.z])
                        route_bb.append([p2.x, p2.y, p2.z])

                    if len(route_bb) < 3:
                        # 2 points don't create a polygon, nothing to check
                        return (False, None, -1, target_list)
                    ego_polygon = Polygon(route_bb)

                    # Compare the two polygons
                    for target in target_list:
                        if "walker" in str(target.type_id):
                            target_extent = target.bounding_box.extent.x
                            if target.id == self._vehicle.id:
                                continue
                            if ego_location.distance(target.get_location()) > max_distance:
                                continue

                            target_bb = target.bounding_box
                            target_vertices = target_bb.get_world_vertices(target.get_transform())
                            target_list = [[v.x, v.y, v.z] for v in target_vertices]
                            target_polygon = Polygon(target_list)

                            if ego_polygon.intersects(target_polygon):

                                return (True, target, compute_distance(target.get_location(), ego_location), target_list)

                            return (False, None, -1, target_list)

        return (False, None, -1, target_list)

    def obstacles_list(self, type_obstacle, opposite_direction, ego_front_transform, ego_wpt):
        """
        Method to load a list of nearby obstacles (vehicles or pedestrians) relative to the ego vehicle.
        The returned list is sorted by distance from the ego vehicle in ascending order.

            :param type_obstacle (str): the type of obstacle to consider. Accepted values: "vehicle", "static.prop", "walker".
            :param opposite_direction (str): the direction of travel used to determine if a target is ahead or behind.
            :param ego_front_transform (carla.Transform): the transform of the front of the ego vehicle.
            :param ego_wpt (carla.Waypoint): the waypoint associated with the ego vehicle's location.
            
            :return target_list (list of carla.Actor): list of actors (vehicles or walkers) that are potential obstacles.
        """
        obstacles_list = []
        coord_map = {
            '-y': ('y', 1),
            'y': ('y', -1),
            'x': ('x', -1),
            '-x': ('x', 1)
        }

        if type_obstacle in ["vehicle", "static.prop"]:
            self._vehicle_l2 = [] if not ego_wpt.is_junction else self._vehicle_l2
            dist_value = 40 if get_speed(self._vehicle) > 50 else 30
            axis, multiplier = coord_map.get(opposite_direction, ('x', 1))

            for v in self._world.get_actors().filter(f"*{type_obstacle}*"):
                if v.id == self._vehicle.id: continue
                
                v_loc = getattr(v.get_transform().location, axis)
                ego_loc = getattr(ego_front_transform.location, axis)
                is_ahead = (v_loc * multiplier) > (ego_loc * multiplier)

                if custom_distance(self._vehicle, v) < dist_value and is_ahead:
                    v_wpt = self._map.get_waypoint(v.get_location())
                    lane_check = (
                        (-2 <= v_wpt.lane_id < 0) if ego_wpt.lane_id < 0 else
                        (0 < v_wpt.lane_id <= 2)
                    ) if not ego_wpt.is_junction else True

                    if lane_check or ego_wpt.is_junction:
                        distance = min(
                            custom_distance(self._vehicle, v, "front"),
                            custom_distance(self._vehicle, v)
                        ) if ego_wpt.is_junction else custom_distance(self._vehicle, v)

                        if not any(car.id == v.id for car, _ in self._vehicle_l2):
                            self._vehicle_l2.append((v, distance))

            obstacles_list = [v[0] for v in sorted(self._vehicle_l2, key=lambda x: x[1])]

        elif type_obstacle == "walker":
            obstacles_list = [
                w[0] for w in sorted(
                    ((w, custom_distance(self._vehicle, w)) 
                    for w in self._world.get_actors().filter("*walker.pedestrian*") 
                    if custom_distance(self._vehicle, w) < 20),
                    key=lambda x: x[1]
                )
            ]

        return obstacles_list
    
    def _generate_lane_change_path(self, waypoint, direction='left', distance_same_lane=10, distance_other_lane=25, lane_change_distance=25, lane_changes=1, step_distance=2):
        """
        Generates the path for a lane change maneuver.
        """

        plan = [(waypoint, RoadOption.CHANGELANELEFT)]

        option = RoadOption.LANEFOLLOW

        # Stessa lane
        distance = 0
        while distance < distance_same_lane:
            next_wps = plan[-1][0].next(
                step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        if direction == 'left':
            option = RoadOption.CHANGELANELEFT
        elif direction == 'right':
            option = RoadOption.CHANGELANERIGHT
        else:
            return []

        lane_changes_done = 0
        lane_change_distance = lane_change_distance // lane_changes

        # Cambio lane
        while lane_changes_done < lane_changes:
            next_wps = plan[-1][0].next(lane_change_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]

            # Prende il waypoint e la lane di sinistra
            if direction == 'left':
                side_wp = next_wp.get_left_lane()
                
            else:
                side_wp = next_wp.get_right_lane()

            if not side_wp or side_wp.lane_type != carla.LaneType.Driving:
                return []

            # Aggiorna il plan
            plan.append((side_wp, option))
            lane_changes_done += 1

        # Andare dritto nella nuova lane
        distance = 0
        pivot = plan[-1][0].lane_id
        while distance < distance_other_lane:
            if waypoint.lane_id * pivot > 0 and concorde:
                next_wps = plan[-1][0].next(step_distance)
            else:
                next_wps = plan[-1][0].previous(step_distance)
            if not next_wps:
                return []
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))

        return plan

    def _distance_hypo(self, waypoint, hypo_short, hypo_long, lane_changes=1):
        # Inizializza il percorso con il waypoint di partenza
        trajectory = [(waypoint, RoadOption.LANEFOLLOW)]
        
        # Prima fase: avanzamento con distanza corta
        completed_changes = 0
        while completed_changes < lane_changes:
            current_wp = trajectory[-1][0]
            forward_waypoints = current_wp.next(hypo_short)
            
            if len(forward_waypoints) == 0:
                return []
                
            selected_wp = forward_waypoints[0]
            left_waypoint = selected_wp.get_left_lane()
            completed_changes += 1
        
        # Reset del contatore per la seconda fase
        completed_changes = 0
        
        # Seconda fase: avanzamento con distanza lunga
        while completed_changes < lane_changes:
            current_wp = trajectory[-1][0]
            extended_waypoints = current_wp.next(hypo_long)
            
            if len(extended_waypoints) == 0:
                return []
                
            chosen_wp = extended_waypoints[0]
            adjacent_wp = chosen_wp.get_left_lane()
            completed_changes += 1
        
        # Calcolo della distanza tra le due posizioni finali
        distance_between_positions = left_waypoint.transform.location.distance(adjacent_wp.transform.location)
        
        return distance_between_positions
            