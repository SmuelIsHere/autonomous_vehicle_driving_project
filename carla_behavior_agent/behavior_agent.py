# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import numpy as np
import carla 
import math
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
from shapely.geometry import Polygon

from misc import get_speed, positive, get_trafficlight_trigger_location, custom_distance, custom_distance_not_norm


class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._control= carla.VehicleControl()
        self._sampling_resolution = 4.5
        self._vehicle_front_distance = 0        # Distance to the front of the vehicle to overtake (single vehicle case)
        self._can_overtake = False              # True indicates that you can overtake
        self._cross_line = False                # False indicates that we have returned to the right after overtaking
        self._stop = False                      # True when encountering a stop sign
        self._target_car = None                 # Opposite vehicle to wait for before overtaking
        self._ego_wpt_pre_overtake = None       # waypoint of our vehicle before overtaking
        self._opposite_direction = None         # Direction of vehicles in opposite lanes
        self._calculate_distance = True         # True if you need to save the distance between us and the vehicle to overtake (single vehicle case)
        self._overtake_custom = False           # True when performing custom overtake (single vehicle case)
        self._can_overtake_bicycle = False      # True if you can overtake bicycles
        
        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
        self._local_planner.set_speed(target_speed - target_speed/2.5)
        
        self._direction = self._local_planner.target_road_option
        if self._direction is None and not self._can_overtake:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None and not self._can_overtake:
            self._incoming_direction = RoadOption.LANEFOLLOW

    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def collision_and_car_avoid_manager(self, waypoint, dis = 50):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance, target_list = self._vehicle_obstacle_detected( None, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), lane_offset=-1, type_obstacle = 'vehicle', opposite_direction=self._opposite_direction)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance, target_list= self._vehicle_obstacle_detected(
                None, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), lane_offset=1, type_obstacle = 'vehicle', opposite_direction=self._opposite_direction)
        else:
            vehicle_state, vehicle, distance, target_list = self._vehicle_obstacle_detected(
                None, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 4), type_obstacle = 'vehicle', opposite_direction=self._opposite_direction)

        return vehicle_state, vehicle, distance, target_list
        
    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return walker_state: True if there is a walker nearby, False if not
            :return walker: nearby walker
            :return distance: distance to nearby walker
        """
  
        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance, _ = self._vehicle_obstacle_detected(None, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), lane_offset=-1, type_obstacle = 'walker')
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance, _ = self._vehicle_obstacle_detected(None, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), lane_offset=1, type_obstacle = 'walker')
        else:
            walker_state, walker, distance, _ = self._vehicle_obstacle_detected(None, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), type_obstacle = 'walker')

        return walker_state, walker, distance

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return self._control: carla.Vehiclecontrol
        """
        if str(vehicle.type_id) in ["vehicle.diamondback.century", "vehicle.bh.crossbike", "vehicle.gazelle.omafiets"] :
            threshold = 1.8
        else:
            threshold = 2 
            
        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)
        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            self._control= self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif threshold * self._behavior.safety_time > ttc >= self._behavior.safety_time:
            
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            self._control= self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            self._control= self._local_planner.run_step(debug=debug)

        return self._control

    def run_step(self, debug=False):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return self._control: carla.Vehiclecontrol
        """
        # ----------------------------FASE PREPARATORIA ----------------------------------

        static_object_position_ahead = 0    # distanza dell'ostacolo davanti al veicolo ego
        static_object_position_behind = 0   # distanza dell'ostacolo dietro il veicolo ego
        c_distance = 0                      # distanza tra veicolo ego e l'ostacolo
        next_ego_wp_dir = RoadOption.STRAIGHT 

        # Ottieni posizione e waypoint attuale del veicolo ego
        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        # Calcola i prossimi waypoint per determinare la direzione futura
        next_ego_wp = ego_vehicle_wp.next(6)[0]
        n_ego_wp = ego_vehicle_wp.next(4)[0]
        ego_yaw = ego_vehicle_wp.transform.rotation.yaw

        # Determina la direzione del prossimo waypoint rispetto all'orientamento attuale
        if ego_yaw - 90 < n_ego_wp.transform.rotation.yaw < ego_yaw - 20:
            next_ego_wp_dir = RoadOption.LEFT
        elif ego_yaw + 20 < n_ego_wp.transform.rotation.yaw < ego_yaw + 90:
            next_ego_wp_dir = RoadOption.RIGHT
        else:
            next_ego_wp_dir = RoadOption.STRAIGHT

        # Ottengo le coordinate attuali e prossime a partire dai waypoint
        next_ego_location_x = round(next_ego_wp.transform.location.x)
        next_ego_location_y = round(next_ego_wp.transform.location.y)
        ego_location_x = round(ego_vehicle_wp.transform.location.x)
        ego_location_y = round(ego_vehicle_wp.transform.location.y)

        # Determina la direzione opposta a quella del veicolo ego
        dx = next_ego_location_x - ego_location_x
        dy = next_ego_location_y - ego_location_y

        if dx != 0:
            self._opposite_direction = "x" if dx < 0 else "-x"
        elif dy != 0:
            self._opposite_direction = "y" if dy < 0 else "-y"

        # Calcola la posizione frontale del veicolo ego
        ego_transform = self._vehicle.get_transform()
        ego_forward_vector = ego_transform.get_forward_vector()
        ego_extent = self._vehicle.bounding_box.extent.x
        ego_front_transform = ego_transform
        ego_front_transform.location += carla.Location(
            x=ego_extent * ego_forward_vector.x,
            y=ego_extent * ego_forward_vector.y,
        )

        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())
        ego_yaw = round(ego_transform.rotation.yaw)

        # Aggiorna informazioni dinamiche (velocità, limiti, ecc.)
        self._update_information()
        
        # ----------------------------FINE FASE PREPARATORIA ----------------------------------

        # ----------------------------- INIZIO GESTIONE TRAFFICO -----------------------------
        
        # --- Gestione semafori rossi e stop ---
        if self.traffic_light_manager():
            print("Semaforo rosso rilevato: Frenata di emergenza in corso")
            return self.emergency_stop()
        
        # --- Gestione pedoni: se troppo vicini, fermati ---
        walker_state, walker, _ = self.pedestrian_avoid_manager(ego_vehicle_wp)
        if walker_state:
            self._can_overtake = False
            if custom_distance(self._vehicle, walker) < 8:
                return self.emergency_stop()
        
        # --- Gestione ostacoli statici (props) ---
        prop_state, prop_other_lane_state, prop_list, cone_list_other_lane = self._static_obstacles_manager(static_element='static.prop.*',
                                                                                                          static_max_distance=30)
        prop_l2 = []

        # Se ci sono props statici nella corsia del veicolo ego 
        if prop_state:
            # first warning contiene il primo ostacolo rilevato 
            first_warning = prop_list[0][0]
            first_warning_transform = first_warning.get_transform()
            
            # Determina se l'ostacolo statico si trova davanti al veicolo
            if self._opposite_direction == '-y':
                static_object_position_ahead = first_warning_transform.location.y
                static_object_position_behind = ego_front_transform.location.y
            elif self._opposite_direction == "y":
                static_object_position_behind = first_warning_transform.location.y 
                static_object_position_ahead = ego_front_transform.location.y
            elif self._opposite_direction == "x":
                static_object_position_behind = first_warning_transform.location.x
                static_object_position_ahead = ego_front_transform.location.x
            elif self._opposite_direction == "-x":
                static_object_position_behind = ego_front_transform.location.x
                static_object_position_ahead = first_warning_transform.location.x 

            # Se l'ostacolo è davanti e non possiamo sorpassare, fermati
            # la update information scrive su can_overtake
            if not self._can_overtake and static_object_position_ahead > static_object_position_behind:
                if (get_speed(self._vehicle) / 3.6) > 0 and custom_distance(self._vehicle, first_warning) < 10 and not "static.prop.dirtdebris" in str(first_warning.type_id):
                    return self.emergency_stop()

                # Se siamo fermi e non in una giunzione, inizia il sorpasso dell'ostacolo statico
                if ((get_speed(self._vehicle) / 3.6) == 0) and not n_ego_wp.is_junction and not "static.prop.dirtdebris" in str(first_warning.type_id):
                    self._ego_wpt_pre_overtake = self._map.get_waypoint(self._vehicle.get_location()) 
                    for prop in prop_list:
                        prop_l2.append(prop[0]) 
                    self.new_overtake(ego_vehicle_wp, first_warning, target_list=prop_l2)
                    return self._control
        
        # --- Gestione coni nella corsia opposta: spostamento a destra se necessario ---
        if prop_other_lane_state:
            offset = 0.02
            for i in cone_list_other_lane:
                current_location = self._vehicle.get_location()
                if self._opposite_direction == "-x":
                    new_position = carla.Location(x=current_location.x, y=current_location.y + offset, z=current_location.z)
                elif self._opposite_direction == "x":
                    new_position = carla.Location(x=current_location.x, y=current_location.y - offset, z=current_location.z)
                elif self._opposite_direction == "y":
                    new_position = carla.Location(x=current_location.x + offset, y=current_location.y, z=current_location.z)
                elif self._opposite_direction == "-y":
                    new_position = carla.Location(x=current_location.x - offset, y=current_location.y, z=current_location.z)
                self._vehicle.set_location(new_position)
        

        # --- Gestione veicoli: collisioni, sorpassi, frenate ---

        vehicle_state, vehicle, distance, target_list = self.collision_and_car_avoid_manager(ego_vehicle_wp)
        self._incoming_waypoint = ego_vehicle_wp.next(3)[0] 
        stop_state, _ = self.signal_avoid_manager()

        # Se c'è un veicolo davanti e non siamo in una giunzione
        if vehicle is not None and not n_ego_wp.is_junction:
            vehicle_transform = vehicle.get_transform()
            vehicle_wpt = self._map.get_waypoint(vehicle_transform.location, lane_type=carla.LaneType.Any)

            # Soglie per considerare i veicoli sulla nostra corsia
            if self._incoming_waypoint.lane_id < 0:
                left_thresh = -2
                right_thresh = 0
            else:
                left_thresh = 0
                right_thresh = 2
            
            # Determina se il veicolo è davanti o dietro rispetto all'ego
            if self._opposite_direction == '-y':
                static_object_position_ahead = vehicle_transform.location.y
                static_object_position_behind = ego_front_transform.location.y

            elif self._opposite_direction == "y":
                static_object_position_behind = vehicle_transform.location.y 
                static_object_position_ahead = ego_front_transform.location.y

            elif self._opposite_direction == "x":
                static_object_position_behind = vehicle_transform.location.x
                static_object_position_ahead = ego_front_transform.location.x

            elif self._opposite_direction == "-x":
                static_object_position_behind = ego_front_transform.location.x
                static_object_position_ahead = vehicle_transform.location.x 

            # Gestione biciclette: sorpasso o stop
            if str(vehicle.type_id) in ["vehicle.diamondback.century", "vehicle.bh.crossbike", "vehicle.gazelle.omafiets"]:
                target_transform = vehicle.get_transform()
                target_yaw = round(target_transform.rotation.yaw)
                # Se la bici è davanti e nella stessa direzione, sorpassa
                if (ego_yaw - 50 <= target_yaw <= ego_yaw + 50) and static_object_position_ahead > static_object_position_behind:
                    self.new_overtake_bicycle(ego_wpt, vehicle)
                    return self._control
                # Se la bici attraversa, fermati
                elif custom_distance(self._vehicle, vehicle) < 4.25:
                    self.emergency_stop()

            # Gestione veicoli sulla nostra corsia che si trovano davanti
            # se il veicolo è davanti e nella stessa direzione, rallenta o fermati
            # se il veicolo è dietro e nella stessa direzione, sorpassa
            elif vehicle_state and (ego_wpt.is_junction or (vehicle_wpt.lane_id >= left_thresh and vehicle_wpt.lane_id <= right_thresh and static_object_position_ahead > static_object_position_behind)):
                target_transform = vehicle.get_transform()
                target_yaw = round(target_transform.rotation.yaw)
                ego_yaw = round(self._vehicle.get_transform().rotation.yaw)
                c_distance = custom_distance(self._vehicle, vehicle)
                # Determina la distanza di frenata in base alla velocità del veicolo davanti
                if get_speed(vehicle) > 0.3:
                    dist_brake = 10
                else:
                    dist_brake = 14 
                
                # Se troppo vicino, rallenta o fermati
                if not ego_wpt.is_junction and (c_distance <= dist_brake and c_distance >= 6 and (get_speed(self._vehicle) > 0.2) and not self._can_overtake): 
                    self._control = self.stop(c_distance, dist_brake)
                elif (ego_wpt.is_junction and c_distance < 10) or c_distance < 6 and (get_speed(self._vehicle) > 0 and not self._can_overtake):
                    self.emergency_stop()
                
                # Sorpasso auto ferme
                if not self._can_overtake and ego_vehicle_wp.lane_id * vehicle_wpt.lane_id > 1 and ((get_speed(self._vehicle) / 3.6) < 0.01) and not vehicle_wpt.is_junction and not n_ego_wp.is_junction:   
                    self._ego_wpt_pre_overtake = self._map.get_waypoint(self._vehicle.get_location())
                    self.new_overtake(ego_vehicle_wp, vehicle, target_list=target_list)
                    self._control = self._local_planner.run_step(debug=debug)
                    # Correzione sterzata anomala dopo sorpasso
                    if self._control.steer > 0:
                        self._control.steer *= -1 
                    return self._control
                
                # Rientro a destra dopo sorpasso (caso auto singola)
                elif self._can_overtake and self._overtake_custom:
                    c_distance_nn = custom_distance_not_norm(vehicle, self._vehicle, self._opposite_direction, "front")
                    # Rientra se il front ha superato la macchina a destra 
                    if self._calculate_distance:
                        self._vehicle_front_distance = c_distance_nn
                        self._calculate_distance = False
                    if c_distance_nn * self._vehicle_front_distance < 0:
                        self.end_overtake()
                        self._calculate_distance = True    
            
        # --- Gestione delle giunzioni (incroci) ---
        elif next_ego_wp.is_junction:
            distance = 0
            if vehicle is not None:
                custom_front = custom_distance(self._vehicle, vehicle, "front")
                custom_rear = custom_distance(self._vehicle, vehicle)
                distance = min(custom_front, custom_rear)
            if vehicle is None or distance > 20:
                self._local_planner.set_speed(17)
                self._control = self._local_planner.run_step(debug=debug)            
           
            # Gestione STOP nelle giunzioni                                           
            stop_state, _ = self.signal_avoid_manager()
            if stop_state and not self._stop:
                self._speed = get_speed(self._vehicle)
                if self._speed > 0:
                    return self.emergency_stop()
                else:
                    self._stop = True
           
            # Rallenta se non c'è segnale di stop
            if self._speed > 0.02 and not self._stop:
                self.emergency_stop()
            else:
                self._stop = True

            if self._stop == True:
                vehicle_list_junction = self.ordered_vehicles(self._vehicle, 35)
                for vehicle in vehicle_list_junction:
                    ego_transform = self._vehicle.get_transform()
                    ego_wpt = self._map.get_waypoint(self._vehicle.get_location())
                    vehicle_transform = vehicle.get_transform()
                    vehicle_wpt = self._map.get_waypoint(vehicle.get_location())
                    # Mi fermo se l'incrocio è impegnato da un'altra auto
                    if self._control.brake > 0 and 0.3 < get_speed(vehicle) < 1 and (vehicle.get_light_state() == carla.libcarla.VehicleLightState.LeftBlinker or vehicle.get_light_state() == carla.libcarla.VehicleLightState.RightBlinker):
                        return self.emergency_stop()
                    else:
                        vehicle_is_obstacle, vehicle_obstacle, _, _ = self.collision_and_car_avoid_manager(ego_vehicle_wp, 15)
                        if vehicle_is_obstacle:
                            vehicle_obstacle_transform = vehicle_obstacle.get_transform()
                            if self._opposite_direction == '-y':
                                static_object_position_ahead = vehicle_obstacle_transform.location.y
                                static_object_position_behind = ego_front_transform.location.y
                            elif self._opposite_direction == "y":
                                static_object_position_behind = vehicle_obstacle_transform.location.y 
                                static_object_position_ahead = ego_front_transform.location.y
                            elif self._opposite_direction == "x":
                                static_object_position_behind = vehicle_obstacle_transform.location.x
                                static_object_position_ahead = ego_front_transform.location.x
                            elif self._opposite_direction == "-x":
                                static_object_position_behind = ego_front_transform.location.x
                                static_object_position_ahead = vehicle_obstacle_transform.location.x 

                            # Se il veicolo opposto si sta muovendo mi fermo                           
                            if get_speed(vehicle_obstacle) > 0.3:
                                return self.emergency_stop()
                # Se nessuna condizione è verificata, riparto
                return self._local_planner.run_step(debug=debug) 
                
        # --- Comportamento normale: segui la strada e rispetta i limiti ---
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed - target_speed / 2.5)
            self._control = self._local_planner.run_step(debug=debug)

        # --- Reset flag dopo il sorpasso ---
        if self._can_overtake == True:
            ego_wpt_overtake = self._map.get_waypoint(self._vehicle.get_location())
            if ego_wpt_overtake.lane_id != self._ego_wpt_pre_overtake.lane_id and not self._cross_line:
                self._cross_line = True
            if self._cross_line == True and ego_wpt_overtake.lane_id == self._ego_wpt_pre_overtake.lane_id:
                self._can_overtake = False
                self._cross_line = False

        # Reset flag di stato
        self._stop = False
        self._can_overtake_bicycle = False

        return self._control

    def bb_to_polygon(self,bounding_box):
        polygon = Polygon([(bounding_box.location.x - bounding_box.extent.x, bounding_box.location.y - bounding_box.extent.y),
                        (bounding_box.location.x + bounding_box.extent.x, bounding_box.location.y - bounding_box.extent.y),
                        (bounding_box.location.x + bounding_box.extent.x, bounding_box.location.y + bounding_box.extent.y),
                        (bounding_box.location.x - bounding_box.extent.x, bounding_box.location.y + bounding_box.extent.y)])
        return polygon

    # Gestione della frenata leggera
    def stop(self, distance, dist_brake):

        if (distance < dist_brake) and (distance > 8):
            self._control.throttle      = 0.0
            self._control.brake         = 0.15
            self._control.hand_brake    = False

        elif (distance <= 8) and (distance > 6):
            self._control.throttle      = 0.0
            self._control.brake         = 0.3
            self._control.hand_brake    = False

        return self._control

    def unif_accel_motion(self, v_0, total_distance, a):
        return (-2 * v_0 + math.sqrt(4 * v_0 ** 2 + 8 * a * total_distance)) / (2 * a)

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a self._controlto perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.Vehiclecontrol): self._controlto be modified
        """
        self._control.throttle = 0.0
        self._control.brake =  self._max_brake
        self._control.hand_brake = False
        return self._control

    def new_overtake(self, ego_wpt, vehicle_to_overtake, target_list=None):
    
        # Costanti per i cambi di corsia
        lane_change_dist_1 = 2 
        lane_change_dist_2 = 2

        # Distanza dal primo veicolo da sorpassare
        initial_vehicle_distance = custom_distance(self._vehicle, vehicle_to_overtake)

        # Distanza totale da percorrere nella corsia di sorpasso
        overtaking_lane_distance = self.overtake_distance(target_list)

        # Calcolo delle ipotenuse per determinare le distanze di manovra
        first_maneuver_hypotenuse = math.hypot(initial_vehicle_distance, ego_wpt.lane_width)
        second_maneuver_hypotenuse = math.hypot(lane_change_dist_2, ego_wpt.lane_width)

        # Ipotenusa per il cambio di corsia iniziale
        lane_change_hypotenuse = math.hypot(lane_change_dist_1, ego_wpt.lane_width)
        
        # Calcolo della differenza tra ipotenuse
        hypotenuse_difference = self._distance_hypo(ego_wpt, lane_change_hypotenuse, first_maneuver_hypotenuse)
        return_distance = initial_vehicle_distance + overtaking_lane_distance + 2 
        adjusted_overtaking_distance = overtaking_lane_distance + hypotenuse_difference
        
        # Distanza totale della manovra di sorpasso
        complete_overtake_distance = first_maneuver_hypotenuse + adjusted_overtaking_distance + second_maneuver_hypotenuse
        
        # Calcolo del tempo necessario usando le formule del moto uniformemente accelerato
        initial_velocity = (get_speed(self._vehicle))/3.6
        vehicle_acceleration = 5.15 * math.cos(math.radians(3 * self._vehicle.get_transform().rotation.pitch))
        estimated_overtake_time = self.unif_accel_motion(initial_velocity, complete_overtake_distance, vehicle_acceleration)
        
        # Distanza percorsa dai veicoli opposti durante il sorpasso
        oncoming_vehicle_travel_distance = estimated_overtake_time * ((self._speed_limit)/3.6)
        
        # Distanza minima di sicurezza richiesta
        required_safety_distance = complete_overtake_distance + oncoming_vehicle_travel_distance

        # Lista dei veicoli che procedono in direzione opposta
        oncoming_vehicles_list = self.ordered_vehicles_opposite(self._vehicle)
        gap_distances_list = [] 
        
        # Calcolo delle distanze tra veicoli consecutivi opposti
        for idx in range(0, len(oncoming_vehicles_list)-1):
            gap_distances_list.append(abs(oncoming_vehicles_list[idx][1] - oncoming_vehicles_list[idx+1][1]))
        
        # SCENARIO: Nessun veicolo opposto, più veicoli da sorpassare
        if len(oncoming_vehicles_list) == 0 and len(target_list) > 1: 
            generated_overtake_path = self._generate_lane_change_path(ego_wpt, direction='left', distance_same_lane=0,
                                                            distance_other_lane=adjusted_overtaking_distance, 
                                                            lane_change_distance=lane_change_hypotenuse, step_distance=2.0)

            if generated_overtake_path:
                updated_plan = self._local_planner.set_overtake_plan(generated_overtake_path, return_distance)
                self.set_target_speed(self._speed_limit - self._speed_limit/3)
                self.set_global_plan(updated_plan, clean_queue=False)
                self._can_overtake = True

        # SCENARIO: Nessun veicolo opposto, un solo veicolo da sorpassare
        elif len(oncoming_vehicles_list) == 0 and len(target_list) == 1:  
            self._can_overtake = True
            lane_change_offset = 0.35
            self.overtake_custom_loc(lane_change_offset, lane_change_hypotenuse)

        # SCENARIO: Un solo veicolo opposto, più veicoli da sorpassare 
        elif len(gap_distances_list) == 0 and len(oncoming_vehicles_list) > 0 and len(target_list) > 1: 
            current_distance_to_oncoming = custom_distance(self._vehicle, oncoming_vehicles_list[0][0], "front")
            oncoming_vehicle_speed = get_speed(oncoming_vehicles_list[0][0])
            available_time = (current_distance_to_oncoming - complete_overtake_distance - 3) / (oncoming_vehicle_speed/3.6)

            if current_distance_to_oncoming >= required_safety_distance and available_time > estimated_overtake_time:
                generated_overtake_path = self._generate_lane_change_path(ego_wpt, direction='left', distance_same_lane=0,
                                                                distance_other_lane=adjusted_overtaking_distance,
                                                                lane_change_distance=lane_change_hypotenuse, step_distance=2.0)

                if generated_overtake_path:
                    updated_plan = self._local_planner.set_overtake_plan(generated_overtake_path, return_distance)
                    self.set_target_speed(self._speed_limit - self._speed_limit/3)
                    self.set_global_plan(updated_plan, clean_queue=False)
                    self._can_overtake = True 

        # SCENARIO: Un solo veicolo opposto, un veicolo da sorpassare 
        elif len(gap_distances_list) == 0 and len(oncoming_vehicles_list) > 0 and len(target_list) == 1:
            lane_change_offset = 0.35
            oncoming_distance = oncoming_vehicles_list[0][1]
            oncoming_vehicle_speed = get_speed(oncoming_vehicles_list[0][0])
            available_time = (oncoming_distance - complete_overtake_distance - 3) / (oncoming_vehicle_speed/3.6)

            if oncoming_distance >= required_safety_distance + 5 and available_time > estimated_overtake_time + 5:
                self._can_overtake = True
                self.overtake_custom_loc(lane_change_offset, first_maneuver_hypotenuse)
        
        # SCENARIO: Più veicoli opposti - ricerca della finestra di sorpasso
        else:   
            if len(target_list) == 1:
                required_safety_distance += 5
                
            for gap_distance in gap_distances_list:
                gap_index = 0
                self._target_car = None
                
                if gap_distance >= required_safety_distance: 
                    gap_index = gap_distances_list.index(gap_distance)
                    self._target_car = oncoming_vehicles_list[gap_index][0]  

                if self._target_car is not None:
                    normalized_distance = custom_distance_not_norm(self._vehicle, self._target_car, self._opposite_direction)

                    if self._opposite_direction == "x" or self._opposite_direction == "y":
                        position_left = normalized_distance
                        position_right = 0
                    elif self._opposite_direction == "-x" or self._opposite_direction == "-y":
                        position_left = 0
                        position_right = normalized_distance
                
                    # CASO: Target car superata, più veicoli da sorpassare
                    if position_left <= position_right and len(target_list) > 1: 
                        next_oncoming_vehicle = oncoming_vehicles_list[gap_index+1][0]
                        distance_to_next = custom_distance(self._vehicle, next_oncoming_vehicle, "front")
                        next_vehicle_speed = get_speed(oncoming_vehicles_list[gap_index+1][0])
                        time_until_collision = (distance_to_next - complete_overtake_distance - 3) / (next_vehicle_speed/3.6)

                        if time_until_collision > estimated_overtake_time:
                            generated_overtake_path = self._generate_lane_change_path(ego_wpt, direction='left', distance_same_lane=0,
                                                                            distance_other_lane=adjusted_overtaking_distance,
                                                                            lane_change_distance=lane_change_hypotenuse, step_distance=2.0)

                            if generated_overtake_path:
                                updated_plan = self._local_planner.set_overtake_plan(generated_overtake_path, return_distance)
                                self.set_target_speed(self._speed_limit - self._speed_limit/3)
                                self.set_global_plan(updated_plan, clean_queue=False)
                                self._can_overtake = True
                            break

                    # CASO: Target car superata, un solo veicolo da sorpassare 
                    elif position_left <= position_right and len(target_list) == 1:
                        lane_change_offset = 0.35
                        next_oncoming_vehicle = oncoming_vehicles_list[gap_index+1][0]
                        distance_to_next = custom_distance(self._vehicle, next_oncoming_vehicle, "front")
                        next_vehicle_speed = get_speed(oncoming_vehicles_list[gap_index+1][0])
                        time_until_collision = (distance_to_next - complete_overtake_distance - 3) / (next_vehicle_speed/3.6)

                        if time_until_collision > estimated_overtake_time + 3:
                            self._can_overtake = True
                            self.overtake_custom_loc(lane_change_offset, first_maneuver_hypotenuse)
                            break

            
    def overtake_custom_loc(self, offset_change_lane, hyp_d_change):
        # Ci spostiamo sulla corsia di sinistra per effettuare il sorpasso di una sola macchina
        self._overtake_custom = True
        for i in range(0,int(hyp_d_change)):
            current_location = self._vehicle.get_location()
            if self._opposite_direction == "-x":
                new_position = carla.Location(x=current_location.x, y=current_location.y - offset_change_lane, z=current_location.z)
            elif self._opposite_direction == "x":
                new_position = carla.Location(x=current_location.x, y=current_location.y + offset_change_lane, z=current_location.z)
            elif self._opposite_direction == "y":
                new_position = carla.Location(x=current_location.x - offset_change_lane, y=current_location.y, z=current_location.z)
            elif self._opposite_direction == "-y":
                new_position = carla.Location(x=current_location.x + offset_change_lane, y=current_location.y, z=current_location.z)
            self._vehicle.set_location(new_position)

    # Rientriamo sulla corsia di destra al termine del sorpasso di una sola macchina
    def end_overtake(self, offset_change_lane = 0.5, hyp_d_change = 2):
        self._overtake_custom = False
        for i in range(0,int(hyp_d_change)):
            current_location = self._vehicle.get_location()
            if self._opposite_direction == "-x":
                new_position = carla.Location(x=current_location.x, y=current_location.y + offset_change_lane, z=current_location.z)
            elif self._opposite_direction == "x":
                new_position = carla.Location(x=current_location.x, y=current_location.y - offset_change_lane, z=current_location.z)
            elif self._opposite_direction == "y":
                new_position = carla.Location(x=current_location.x + offset_change_lane, y=current_location.y, z=current_location.z)
            elif self._opposite_direction == "-y":
                new_position = carla.Location(x=current_location.x - offset_change_lane, y=current_location.y, z=current_location.z)
            self._vehicle.set_location(new_position)
            
    # Gestione del sorpasso dei veicolo di tipo bicycle
    def new_overtake_bicycle(self, ego_wpt, vehicle):
        
        # Se la larghezza della corsia è inferiore a 3.5 metri e la distanza dalla bici è tra 8 e 12 metri e non è già in corso un sorpasso, segui la bici mantenendo una distanza di sicurezza
        if ego_wpt.lane_width < 3.5 and custom_distance(self._vehicle,vehicle) > 8 and custom_distance(self._vehicle,vehicle) < 12 and not self._can_overtake_bicycle:
            self._control = self.car_following_manager(vehicle,5)
        # Se la corsia è stretta e la distanza dalla bici è tra 12 e 18 metri, imposta una velocità target più alta per prepararsi al sorpasso
        elif ego_wpt.lane_width < 3.5 and custom_distance(self._vehicle,vehicle) > 12 and custom_distance(self._vehicle,vehicle) < 18:
            target_speed = 30
            self._local_planner.set_speed(target_speed)
            self._control= self._local_planner.run_step()

        # Effettua il sorpasso della bici se sei abbastanza vicino e la strada è sufficientemente larga,
        # oppure se sei vicino, la corsia è stretta e la bici è praticamente ferma
        elif (custom_distance(self._vehicle,vehicle) < 8 and ego_wpt.lane_width >=3.5) or (custom_distance(self._vehicle,vehicle) < 8 and ego_wpt.lane_width < 3.5 and get_speed(vehicle)<0.02):
            offset=0.02
            self._can_overtake_bicycle = True
            for i in range(0, 10):
                current_location = self._vehicle.get_location()
                if self._opposite_direction == "-x":
                    new_position = carla.Location(x=current_location.x, y=current_location.y - offset, z=current_location.z)
                elif self._opposite_direction == "x":
                    new_position = carla.Location(x=current_location.x, y=current_location.y + offset, z=current_location.z)
                elif self._opposite_direction == "y":
                    new_position = carla.Location(x=current_location.x - offset, y=current_location.y, z=current_location.z)
                elif self._opposite_direction == "-y":
                    new_position = carla.Location(x=current_location.x + offset, y=current_location.y, z=current_location.z)
                self._vehicle.set_location(new_position)

            # Dopo lo spostamento laterale, imposta una velocità target elevata per completare il sorpasso
            target_speed = 30
            self._local_planner.set_speed(target_speed)
            self._control= self._local_planner.run_step()
                        
    # Prende tutti i veicoli fino a una certa distanza e li ordina in ordine crescente di distanza dall'ego vehicle
    def ordered_vehicles(self, reference, max_distance):
        """
        Returns a list of vehicles in the world ordered by their distance to a reference vehicle, 
        within a specified maximum distance.
        Args:
            reference: The reference vehicle actor to measure distances from.
            max_distance (float): The maximum distance to consider for nearby vehicles.
        Returns:
            List[carla.Actor]: A list of vehicle actors sorted by their distance to the reference vehicle,
            excluding the reference vehicle itself and any vehicles farther than max_distance.
        Note:
            - The function uses a custom_distance function to compute the distance between vehicles.
            - Only vehicles with a distance less than max_distance are included in the result.
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        
        vehicle_list = [(v, custom_distance(v, reference)) for v in vehicle_list if
                            custom_distance(v, reference) < max_distance and v.id != reference.id]

        vehicle_list.sort(key=lambda v: v[1])
        return [v[0] for v in vehicle_list]

    # Prendiamo i veicoli opposti a noi ma nella stessa strada
    def ordered_vehicles_opposite(self, reference):
        """
        Returns a list of vehicles that are in the opposite direction relative to a reference vehicle,
        ordered by their custom distance to the reference.
        Args:
            reference (carla.Actor): The reference vehicle actor.
        Returns:
            list of tuple: A list of tuples, each containing a vehicle actor and its custom distance to the reference,
            sorted in ascending order of distance.
        The function determines the opposite vehicles based on lane and coordinate direction, considering the
        current ego vehicle's lane and a specified opposite direction (self._opposite_direction).
        Only vehicles on the same road and in the opposite direction are included.
        """

        vehicle_list = self._world.get_actors().filter("*vehicle*")
        reference_transform = reference.get_transform()
        reference_wp = self._map.get_waypoint(reference_transform.location, lane_type=carla.LaneType.Any)
        opposite_list =[] 

        for v in vehicle_list:
            v_transform = v.get_transform()
            v_wp = self._map.get_waypoint(v_transform.location, lane_type=carla.LaneType.Any)
            ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

            if ego_wpt.lane_id > 0: 
                left_value_lane = 0
                right_value_lane = v_wp.lane_id
            else:
                left_value_lane = v_wp.lane_id
                right_value_lane = 0

            if self._opposite_direction == "x":
                left_value_coord= v_transform.location.x
                right_value_coord= reference_transform.location.x
            elif self._opposite_direction == "-x":
                right_value_coord= v_transform.location.x
                left_value_coord= reference_transform.location.x
            elif self._opposite_direction == "-y":
                right_value_coord= v_transform.location.y
                left_value_coord= reference_transform.location.y
            elif self._opposite_direction == "y":
                left_value_coord= v_transform.location.y
                right_value_coord= reference_transform.location.y

            if  v.id != reference.id and (  left_value_lane >= right_value_lane and reference_wp.road_id == v_wp.road_id) and left_value_coord < right_value_coord:
                opposite_list.append((v, custom_distance(v, reference)))

        
        opposite_list.sort(key=lambda v: v[1])
        return opposite_list

    # Calcolo distanza da effettuare nella corsia opposta
    def overtake_distance(self, target_list):
        overtake_dist = 0
        if target_list is not None:
            overtake_dist = custom_distance(target_list[-1], target_list[0])
        return overtake_dist

    def signal_avoid_manager(self, max_distance=None):
        # Recupera tutti gli attori e filtra i segnali di stop
        world_actors = self._world.get_actors()
        stop_signals = world_actors.filter("traffic.stop")

        # Controllo anticipato per segnali ignorati
        if self._ignore_stop_signs:
            return (False, None)

        # Imposta la distanza massima di default se non specificata
        detection_range = max_distance if max_distance else self._signal_stop_distance

        # Ottiene posizione e waypoint del veicolo ego
        current_vehicle_position = self._vehicle.get_location()
        vehicle_waypoint = self._map.get_waypoint(current_vehicle_position)

        # Itera attraverso tutti i segnali di stop
        for stop_signal in stop_signals:
            
            # Gestione del waypoint di trigger (con cache)
            if stop_signal.id in self._stop_map:
                signal_trigger_waypoint = self._stop_map[stop_signal.id]
            else:
                signal_trigger_position = get_trafficlight_trigger_location(stop_signal)
                signal_trigger_waypoint = self._map.get_waypoint(signal_trigger_position)
                self._stop_map[stop_signal.id] = signal_trigger_waypoint

            # Verifica se il segnale è troppo distante
            if custom_distance(self._vehicle, stop_signal) > detection_range:
                continue
                
            # Verifica se il segnale è sulla stessa strada
            if signal_trigger_waypoint.road_id != vehicle_waypoint.road_id:
                continue
            
            # Calcola i vettori di direzione
            vehicle_direction = vehicle_waypoint.transform.get_forward_vector()
            signal_direction = signal_trigger_waypoint.transform.get_forward_vector()
            
            # Calcola il prodotto scalare per determinare l'orientamento
            direction_alignment = (vehicle_direction.x * signal_direction.x + 
                                vehicle_direction.y * signal_direction.y + 
                                vehicle_direction.z * signal_direction.z)
            
            # Ignora segnali non rivolti verso il veicolo
            if direction_alignment < 0:
                continue

            # Verifica finale della distanza per attivazione
            if custom_distance(self._vehicle, stop_signal) < detection_range:
                return (True, stop_signal)

        return (False, None)

    def _static_obstacles_manager(self, static_element: str = 'static.prop.*', static_max_distance=20):
        # La funzione restituisce 2 liste, 
        # una contenente gli static prop e una contenente i coni che sono a sinistra della carreggiata

        static_props = self._world.get_actors().filter(static_element)
        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())
        static_props_list=[]
        cone_list_other_lane=[] 
        

        for w in static_props:
                prop_transform = w.get_transform()
                prop_wpt = self._map.get_waypoint(prop_transform.location, lane_type=carla.LaneType.Any)

                # Calcolo della posizione del waypoint 
                diff = prop_wpt.transform.location - ego_wpt.transform.location
                ego_rv = ego_wpt.transform.rotation.get_right_vector()
                # Vediamo se gli static prop sono sulla nostra corsia
                dot_prod=ego_rv.x*diff.x + ego_rv.y*diff.y
                
                if custom_distance(self._vehicle,w) < static_max_distance and prop_wpt.lane_id==ego_wpt.lane_id:
                    static_props_list.append((w, custom_distance(self._vehicle, w)))
                
                # Prende i coni nella lane di sinistra
                elif custom_distance(self._vehicle,w) < 30 and str(w.type_id)=="static.prop.constructioncone" and dot_prod < 0:
                    cone_list_other_lane.append((w, custom_distance(self._vehicle, w)))
        static_props_list.sort(key=lambda v: v[1])
        cone_list_other_lane.sort(key=lambda v: v[1])
        return len(static_props_list) > 0, len(cone_list_other_lane) > 0,static_props_list, cone_list_other_lane
