# enhanced_v2x_simulation_app.py
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit
import time
import random
import math
import numpy as np
from enum import Enum
import threading
import json
from scipy.ndimage import gaussian_filter
from datetime import datetime
from dataclasses import dataclass

# ========== Simulation Models ==========


class VehicleType(Enum):
    CAR = 1
    TRUCK = 2
    BUS = 3
    EMERGENCY = 4


class MessageType(Enum):
    SAFETY_WARNING = 1
    TRAFFIC_INFO = 2
    ROAD_CONDITION = 3
    EMERGENCY_VEHICLE = 4
    V2V_POSITION = 5
    V2I_REQUEST = 6
    I2V_SIGNAL = 7


class WeatherCondition(Enum):
    CLEAR = 1
    RAIN = 2
    SNOW = 3
    FOG = 4


@dataclass
class SimulationScenario:
    name: str
    description: str
    vehicle_density: float
    weather: WeatherCondition
    time_of_day: str  # "DAY" or "NIGHT"
    map_type: str


class Vehicle:
    def __init__(self, vehicle_id, vehicle_type, position, direction, speed):
        self.id = vehicle_id
        self.type = vehicle_type
        self.position = position  # (lat, lng)
        self.direction = direction  # angle in degrees
        self.speed = speed  # km/h
        self.messages_sent = []
        self.messages_received = []
        self.route = []

        # Enhanced properties
        self.fuel_level = 100.0
        self.fuel_efficiency = 0.05  # fuel units per km
        self.battery_level = (
            100.0 if vehicle_type == VehicleType.CAR and random.random() < 0.3 else 0
        )  # for electric vehicles
        self.platooning = False
        self.platoon_id = None
        self.automation_level = random.randint(0, 5)  # SAE automation levels
        self.route_history = []
        self.base_speed = speed

    def update_position(self, road_network, time_step):
        # Move based on speed and direction
        speed_ms = self.speed * 0.27778  # Convert km/h to m/s
        distance = speed_ms * time_step

        # Convert direction to radians
        direction_rad = math.radians(self.direction)

        # Calculate movement in both directions (simplified)
        # In real system, would use proper geospatial calculations
        lat_change = (
            distance * math.cos(direction_rad) * 0.00001
        )  # Simplified conversion to degrees
        lng_change = distance * math.sin(direction_rad) * 0.00001

        # Update position
        self.position = (self.position[0] + lat_change, self.position[1] + lng_change)

        # Simple boundary check (keep vehicles in the visible area)
        self.position = (
            max(min(self.position[0], 37.79), 37.76),
            max(min(self.position[1], -122.39), -122.43),
        )

        # Random small changes to direction for realism
        self.direction += random.uniform(-2, 2)
        if self.direction > 360:
            self.direction -= 360
        elif self.direction < 0:
            self.direction += 360

        # Random speed variations if not in platoon
        if not self.platooning:
            self.speed += random.uniform(-2, 2)
            self.speed = max(5, min(120, self.speed))  # Speed limits

        # Track route history
        self.route_history.append(self.position)
        if len(self.route_history) > 100:  # Keep only recent history
            self.route_history = self.route_history[-100:]

        # Update fuel/battery
        distance_km = distance / 1000
        self.update_fuel(distance_km)

    def update_fuel(self, distance_km):
        """Update fuel or battery level based on distance traveled"""
        if self.battery_level > 0:  # Electric vehicle
            self.battery_level -= distance_km * 0.1
            self.battery_level = max(0, self.battery_level)
        else:  # Conventional vehicle
            self.fuel_level -= distance_km * self.fuel_efficiency
            self.fuel_level = max(0, self.fuel_level)

    def join_platoon(self, platoon_id):
        """Join a vehicle platoon"""
        self.platooning = True
        self.platoon_id = platoon_id
        # Vehicles in a platoon follow closely and maintain consistent speed

    def leave_platoon(self):
        """Leave the current platoon"""
        self.platooning = False
        self.platoon_id = None

    def to_dict(self):
        """Convert vehicle to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "type": self.type.name,
            "position": self.position,
            "direction": self.direction,
            "speed": self.speed,
            "fuel_level": self.fuel_level,
            "battery_level": self.battery_level,
            "platooning": self.platooning,
            "platoon_id": self.platoon_id,
            "automation_level": self.automation_level,
        }


class Infrastructure:
    def __init__(self, infra_id, infra_type, position):
        self.id = infra_id
        self.type = infra_type  # Traffic light, RSU, etc.
        self.position = position
        self.state = "GREEN" if infra_type == "TRAFFIC_LIGHT" else "ACTIVE"
        self.state_duration = 0
        self.messages_sent = []
        self.messages_received = []

    def update(self, time_step):
        if self.type == "TRAFFIC_LIGHT":
            self.state_duration += time_step
            if self.state_duration >= 30:  # 30 seconds per cycle
                self.state_duration = 0
                if self.state == "GREEN":
                    self.state = "YELLOW"
                elif self.state == "YELLOW":
                    self.state = "RED"
                elif self.state == "RED":
                    self.state = "GREEN"

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "position": self.position,
            "state": self.state,
        }


class SecurityEvent:
    def __init__(self, event_id, event_type, source, target, severity, description):
        self.id = event_id
        self.type = event_type  # "JAMMING", "SPOOFING", "EAVESDROPPING", etc.
        self.source = source
        self.target = target
        self.severity = severity  # 1-5, with 5 being most severe
        self.timestamp = time.time()
        self.description = description
        self.resolved = False

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "source": self.source,
            "target": self.target,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "description": self.description,
            "resolved": self.resolved,
        }


class TrafficAccident:
    def __init__(self, accident_id, position, vehicles_involved, severity, duration):
        self.id = accident_id
        self.position = position
        self.vehicles_involved = vehicles_involved
        self.severity = severity  # 1-5, with 5 being most severe
        self.start_time = time.time()
        self.duration = duration  # seconds
        self.cleared = False

    def to_dict(self):
        return {
            "id": self.id,
            "position": self.position,
            "vehicles_involved": self.vehicles_involved,
            "severity": self.severity,
            "start_time": self.start_time,
            "duration": self.duration,
            "cleared": self.cleared,
        }


class RoadNetwork:
    def __init__(self):
        self.nodes = []  # Intersections
        self.edges = []  # Road segments

    def generate_grid(self, center_lat, center_lng, size=5, spacing=0.004):
        self.nodes = []
        self.edges = []

        # Generate grid nodes
        for i in range(size):
            for j in range(size):
                lat = center_lat - (size / 2 * spacing) + i * spacing
                lng = center_lng - (size / 2 * spacing) + j * spacing
                self.nodes.append({"id": len(self.nodes), "position": (lat, lng)})

        # Generate edges (roads connecting nodes)
        for i in range(size):
            for j in range(size):
                node_id = i * size + j

                # Connect to node to the right
                if j < size - 1:
                    self.edges.append(
                        {
                            "id": len(self.edges),
                            "start_node": node_id,
                            "end_node": node_id + 1,
                            "bidirectional": True,
                        }
                    )

                # Connect to node below
                if i < size - 1:
                    self.edges.append(
                        {
                            "id": len(self.edges),
                            "start_node": node_id,
                            "end_node": node_id + size,
                            "bidirectional": True,
                        }
                    )

    def to_dict(self):
        return {"nodes": self.nodes, "edges": self.edges}


class Message:
    def __init__(
        self,
        msg_id,
        source_id,
        source_type,
        destination_id,
        destination_type,
        msg_type,
        content,
        source_pos,
        dest_pos,
    ):
        self.id = msg_id
        self.source_id = source_id
        self.source_type = source_type
        self.destination_id = destination_id
        self.destination_type = destination_type
        self.type = msg_type
        self.content = content
        self.timestamp = time.time()
        self.source_position = source_pos
        self.destination_position = dest_pos

    def to_dict(self):
        return {
            "id": self.id,
            "source_id": self.source_id,
            "source_type": self.source_type,
            "destination_id": self.destination_id,
            "destination_type": self.destination_type,
            "type": self.type.name,
            "content": self.content,
            "timestamp": self.timestamp,
            "source_position": self.source_position,
            "destination_position": self.destination_position,
        }


class TrafficSimulator:
    def __init__(self):
        self.vehicles = []
        self.infrastructure = []
        self.road_network = RoadNetwork()
        self.messages = []
        self.time = 0
        self.running = False

        # Basic configuration
        self.config = {
            "vehicle_density": 0.1,  # Vehicles per node
            "communication_range": 300,  # meters
            "communication_reliability": 0.95,  # Probability of successful transmission
            "communication_reliability_effective": 0.95,
            "traffic_light_cycle": 30,  # seconds
            "simulation_speed": 1.0,  # Real-time multiplier
            "map_center": (37.7749, -122.4194),  # San Francisco center
            "communication_protocol": "DSRC",  # DSRC, C-V2X, or HYBRID
            "network_latency": 20,  # ms
            "network_bandwidth": 10,  # Mbps
            "packet_loss": 2.0,  # percent
        }

        # Enhanced features
        self.weather_condition = WeatherCondition.CLEAR
        self.time_of_day = "DAY"
        self.security_events = []
        self.accidents = []
        self.platoons = {}
        self.heatmap_data = None
        self.traffic_prediction = None
        self.enable_accidents = True
        self.enable_security_events = True
        self.enable_platooning = True

        # Vehicle type ratios (percentage)
        self.vehicle_type_ratios = {"CAR": 70, "TRUCK": 15, "BUS": 10, "EMERGENCY": 5}

        # Predefined scenarios
        self.scenarios = [
            SimulationScenario(
                "Urban Rush Hour",
                "Heavy traffic in city center during rush hour",
                0.3,
                WeatherCondition.CLEAR,
                "DAY",
                "urban",
            ),
            SimulationScenario(
                "Highway Night Travel",
                "Highway scenario at night with medium traffic",
                0.15,
                WeatherCondition.CLEAR,
                "NIGHT",
                "highway",
            ),
            SimulationScenario(
                "Rainy City Conditions",
                "Urban area during heavy rain",
                0.2,
                WeatherCondition.RAIN,
                "DAY",
                "urban",
            ),
            SimulationScenario(
                "Winter Highway",
                "Highway during snowfall with reduced visibility",
                0.1,
                WeatherCondition.SNOW,
                "DAY",
                "highway",
            ),
            SimulationScenario(
                "Foggy Rural Roads",
                "Rural roads with dense fog",
                0.05,
                WeatherCondition.FOG,
                "NIGHT",
                "rural",
            ),
        ]

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "message_types": {},
            "vehicle_count": 0,
            "average_speed": 0,
            "traffic_density": 0,
            "message_success_rate": 0,
            "time_series": {
                "vehicle_count": [],
                "message_count": [],
                "average_speed": [],
            },
        }

        # Message queue for visualization
        self.active_messages = []
        self.message_id_counter = 0

    def initialize_scenario(self, scenario_type="urban"):
        # Create road network based on scenario
        if scenario_type == "urban":
            self.road_network.generate_grid(
                self.config["map_center"][0], self.config["map_center"][1], 6, 0.005
            )
        elif scenario_type == "highway":
            # Simplified highway network
            self.road_network.generate_grid(
                self.config["map_center"][0], self.config["map_center"][1], 2, 0.02
            )
        elif scenario_type == "rural":
            # Sparse road network
            self.road_network.generate_grid(
                self.config["map_center"][0], self.config["map_center"][1], 4, 0.01
            )

        # Add vehicles and infrastructure
        self._add_initial_vehicles()
        self._add_infrastructure()

    def _add_initial_vehicles(self):
        self.vehicles = []
        # Number of vehicles based on density and network size
        num_vehicles = int(
            len(self.road_network.nodes) * self.config["vehicle_density"]
        )

        for i in range(num_vehicles):
            # Random position near a random node
            node = random.choice(self.road_network.nodes)
            pos = (
                node["position"][0] + random.uniform(-0.002, 0.002),
                node["position"][1] + random.uniform(-0.002, 0.002),
            )

            # Random vehicle type with distribution based on ratios
            type_weights = [
                self.vehicle_type_ratios["CAR"],
                self.vehicle_type_ratios["TRUCK"],
                self.vehicle_type_ratios["BUS"],
                self.vehicle_type_ratios["EMERGENCY"],
            ]

            # Normalize weights to sum to 1
            total = sum(type_weights)
            if total > 0:
                type_weights = [w / total for w in type_weights]
            else:
                type_weights = [0.7, 0.15, 0.1, 0.05]  # Default if invalid

            type_choices = [
                VehicleType.CAR,
                VehicleType.TRUCK,
                VehicleType.BUS,
                VehicleType.EMERGENCY,
            ]
            veh_type = random.choices(type_choices, weights=type_weights)[0]

            # Random direction and appropriate speed for vehicle type
            direction = random.uniform(0, 360)
            if veh_type == VehicleType.CAR:
                speed = random.uniform(30, 60)
            elif veh_type == VehicleType.TRUCK:
                speed = random.uniform(20, 50)
            elif veh_type == VehicleType.BUS:
                speed = random.uniform(25, 45)
            else:  # Emergency
                speed = random.uniform(40, 80)

            vehicle = Vehicle(i, veh_type, pos, direction, speed)
            self.vehicles.append(vehicle)

    def _add_infrastructure(self):
        self.infrastructure = []
        # Add traffic lights at intersections
        for i, node in enumerate(self.road_network.nodes):
            # Only add infrastructure at some intersections
            if random.random() < 0.5:
                infra_type = "TRAFFIC_LIGHT"
                infra = Infrastructure(i, infra_type, node["position"])
                self.infrastructure.append(infra)

        # Add roadside units (RSUs) along some roads
        rsu_count = len(self.road_network.nodes) // 4
        for i in range(rsu_count):
            # Position RSU between nodes
            edge = random.choice(self.road_network.edges)
            start_node = self.road_network.nodes[edge["start_node"]]["position"]
            end_node = self.road_network.nodes[edge["end_node"]]["position"]

            # Interpolate position
            pos = (
                (start_node[0] + end_node[0]) / 2 + random.uniform(-0.001, 0.001),
                (start_node[1] + end_node[1]) / 2 + random.uniform(-0.001, 0.001),
            )

            infra = Infrastructure(len(self.infrastructure), "RSU", pos)
            self.infrastructure.append(infra)

    def step(self):
        # Advance simulation by one time step
        if not self.running:
            return None

        self.time += 1

        # Move vehicles
        for vehicle in self.vehicles:
            vehicle.update_position(self.road_network, 1.0)

        # Update infrastructure
        for infra in self.infrastructure:
            infra.update(1.0)

        # Simulate V2X communications
        self._simulate_communications()

        # Add advanced features execution
        if self.time % 10 == 0 and self.enable_accidents:
            self.generate_accident()

        self.update_accidents()

        if self.time % 10 == 0 and self.enable_security_events:
            self.generate_security_event()

        if self.time % 20 == 0 and self.enable_platooning:
            self.create_platoon()

        if self.time % 20 == 0:
            self.generate_heatmap()

        if self.time % 50 == 0:
            self.predict_traffic()

        # Weather effects applied frequently
        if self.time % 5 == 0:
            self.apply_weather_effects()

        # Update statistics
        self._update_statistics()

        # Return simulation state for visualization
        return self._get_simulation_state()

    def _simulate_communications(self):
        # Clear old messages that have been displayed
        self.active_messages = [
            m for m in self.active_messages if time.time() - m.timestamp < 2
        ]

        # Effective reliability affected by weather and security events
        effective_reliability = self.config["communication_reliability_effective"]

        # Apply network parameters impact
        packet_loss_factor = max(0, 1 - (self.config["packet_loss"] / 100))
        effective_reliability *= packet_loss_factor

        # V2V communications
        for i, v1 in enumerate(self.vehicles):
            for v2 in self.vehicles[i + 1 :]:
                # Calculate distance between vehicles
                distance = self._calculate_distance(v1.position, v2.position)

                # Check if within communication range
                if distance <= self.config["communication_range"]:
                    # Generate V2V messages with probability based on reliability
                    if random.random() < effective_reliability:
                        msg_type = random.choice(
                            [
                                MessageType.SAFETY_WARNING,
                                MessageType.V2V_POSITION,
                                MessageType.TRAFFIC_INFO,
                            ]
                        )

                        # Create bidirectional messages (for demo purposes)
                        if msg_type == MessageType.V2V_POSITION:
                            content = f"Position update: {v1.position}"
                            self._create_message(
                                v1.id,
                                "VEHICLE",
                                v2.id,
                                "VEHICLE",
                                msg_type,
                                content,
                                v1.position,
                                v2.position,
                            )

                            content = f"Position update: {v2.position}"
                            self._create_message(
                                v2.id,
                                "VEHICLE",
                                v1.id,
                                "VEHICLE",
                                msg_type,
                                content,
                                v2.position,
                                v1.position,
                            )

        # V2I and I2V communications
        for vehicle in self.vehicles:
            for infra in self.infrastructure:
                # Calculate distance
                distance = self._calculate_distance(vehicle.position, infra.position)

                # Check if within communication range
                if distance <= self.config["communication_range"]:
                    # V2I message (Vehicle to Infrastructure)
                    if random.random() < effective_reliability:
                        msg_type = MessageType.V2I_REQUEST
                        content = f"Request: {random.choice(['signal_timing', 'road_condition', 'traffic_info'])}"
                        self._create_message(
                            vehicle.id,
                            "VEHICLE",
                            infra.id,
                            "INFRASTRUCTURE",
                            msg_type,
                            content,
                            vehicle.position,
                            infra.position,
                        )

                    # I2V message (Infrastructure to Vehicle)
                    if random.random() < effective_reliability:
                        if infra.type == "TRAFFIC_LIGHT":
                            msg_type = MessageType.I2V_SIGNAL
                            content = f"Traffic light state: {infra.state}"
                        else:  # RSU
                            msg_type = MessageType.TRAFFIC_INFO
                            content = "Traffic info: Normal flow"

                        self._create_message(
                            infra.id,
                            "INFRASTRUCTURE",
                            vehicle.id,
                            "VEHICLE",
                            msg_type,
                            content,
                            infra.position,
                            vehicle.position,
                        )

    def _create_message(
        self,
        source_id,
        source_type,
        dest_id,
        dest_type,
        msg_type,
        content,
        source_pos,
        dest_pos,
    ):
        # Create a new message
        msg = Message(
            self.message_id_counter,
            source_id,
            source_type,
            dest_id,
            dest_type,
            msg_type,
            content,
            source_pos,
            dest_pos,
        )
        self.message_id_counter += 1

        # Add to active messages for visualization
        self.active_messages.append(msg)

        # Update stats
        self.stats["messages_sent"] += 1
        if msg_type.name not in self.stats["message_types"]:
            self.stats["message_types"][msg_type.name] = 0
        self.stats["message_types"][msg_type.name] += 1

    def _calculate_distance(self, pos1, pos2):
        # Simple Euclidean distance calculation (in a real system, would use proper geospatial distance)
        # Convert lat/lng to meters (very approximate)
        lat_diff = (pos1[0] - pos2[0]) * 111000  # 1 degree lat ≈ 111km
        lng_diff = (pos1[1] - pos2[1]) * 111000 * math.cos(math.radians(pos1[0]))
        return math.sqrt(lat_diff**2 + lng_diff**2)

    def create_platoon(self):
        """Form a platoon of nearby vehicles with high automation"""
        if len(self.vehicles) < 5:
            return

        # Check for vehicles close to each other that could form a platoon
        candidate_vehicles = [
            v for v in self.vehicles if not v.platooning and v.automation_level >= 3
        ]  # Only automated vehicles can platoon

        if len(candidate_vehicles) < 3:
            return

        # Sort by position to find vehicles close to each other
        candidate_vehicles.sort(key=lambda v: v.position[0])

        # Create a platoon with 3-5 vehicles
        platoon_size = random.randint(3, min(5, len(candidate_vehicles)))
        platoon_id = len(self.platoons) + 1
        self.platoons[platoon_id] = []

        # Choose a segment of sorted vehicles to form a platoon
        start_idx = random.randint(0, len(candidate_vehicles) - platoon_size)
        for i in range(start_idx, start_idx + platoon_size):
            vehicle = candidate_vehicles[i]
            vehicle.join_platoon(platoon_id)
            self.platoons[platoon_id].append(vehicle.id)

            # Adjust speed for platoon members
            lead_vehicle = candidate_vehicles[start_idx]
            if vehicle.id != lead_vehicle.id:
                vehicle.speed = lead_vehicle.speed
                # Align direction with lead vehicle
                vehicle.direction = lead_vehicle.direction

    def generate_accident(self):
        """Randomly generate traffic accidents based on conditions"""
        if len(self.vehicles) < 5 or len(self.accidents) >= 2:
            return

        # Probability based on weather
        accident_prob = 0.001  # base probability
        if self.weather_condition == WeatherCondition.RAIN:
            accident_prob *= 2
        elif self.weather_condition == WeatherCondition.SNOW:
            accident_prob *= 3
        elif self.weather_condition == WeatherCondition.FOG:
            accident_prob *= 2.5

        if self.time_of_day == "NIGHT":
            accident_prob *= 1.5

        if random.random() < accident_prob:
            # Choose 1-3 vehicles involved
            num_vehicles = random.randint(1, min(3, len(self.vehicles)))
            vehicles_involved = random.sample(
                [v.id for v in self.vehicles], num_vehicles
            )

            # Location is the position of the first involved vehicle
            position = next(
                v.position for v in self.vehicles if v.id == vehicles_involved[0]
            )

            # Severity and duration
            severity = random.randint(1, 5)
            duration = random.randint(30, 300)  # 30s to 5min

            accident = TrafficAccident(
                len(self.accidents), position, vehicles_involved, severity, duration
            )
            self.accidents.append(accident)

            # Affected vehicles stop or slow down
            for v in self.vehicles:
                if v.id in vehicles_involved:
                    v.speed = 0
                else:
                    # Slow down nearby vehicles
                    distance = self._calculate_distance(v.position, position)
                    if distance < 200:
                        slowdown_factor = max(0.2, distance / 200)
                        v.speed *= slowdown_factor

    def update_accidents(self):
        """Update status of existing accidents"""
        for accident in self.accidents:
            if (
                not accident.cleared
                and time.time() - accident.start_time > accident.duration
            ):
                accident.cleared = True
                # Release affected vehicles
                for v in self.vehicles:
                    if v.id in accident.vehicles_involved:
                        v.speed = random.uniform(20, 40)  # Resume at moderate speed

    def generate_security_event(self):
        """Randomly generate security events"""
        if random.random() < 0.002:  # Low probability
            event_types = ["JAMMING", "SPOOFING", "EAVESDROPPING", "MITM"]
            event_type = random.choice(event_types)

            # Select source and target
            if random.random() < 0.5 and self.vehicles:
                source = f"EXTERNAL_ACTOR_{random.randint(1, 10)}"
                target = f"VEHICLE_{random.choice([v.id for v in self.vehicles])}"
            elif self.infrastructure:
                source = f"EXTERNAL_ACTOR_{random.randint(1, 10)}"
                target = f"INFRASTRUCTURE_{random.choice([i.id for i in self.infrastructure])}"
            else:
                return

            severity = random.randint(1, 5)
            descriptions = {
                "JAMMING": "Signal jamming detected",
                "SPOOFING": "Location spoofing attempt",
                "EAVESDROPPING": "Communication eavesdropping detected",
                "MITM": "Man-in-the-middle attack detected",
            }

            event = SecurityEvent(
                len(self.security_events),
                event_type,
                source,
                target,
                severity,
                descriptions[event_type],
            )
            self.security_events.append(event)

            # Affect communications
            if event_type == "JAMMING":
                self.config[
                    "communication_reliability_effective"
                ] *= 0.7  # Temporary reduction

            # After some time, resolve the event
            event_duration = random.randint(10, 60)
            threading.Timer(
                event_duration, lambda: self._resolve_security_event(event.id)
            ).start()

    def _resolve_security_event(self, event_id):
        """Resolve a security event after some time"""
        for event in self.security_events:
            if event.id == event_id:
                event.resolved = True
                # Restore normal operation
                self.config["communication_reliability_effective"] = min(
                    1.0, self.config["communication_reliability"] / 0.7
                )

    def generate_heatmap(self):
        """Generate traffic density heatmap"""
        # Create grid for the current map area
        grid_size = 50
        heatmap = np.zeros((grid_size, grid_size))

        # Map boundaries
        min_lat, max_lat = 37.76, 37.79
        min_lng, max_lng = -122.43, -122.39

        # Add vehicle positions to heatmap
        for vehicle in self.vehicles:
            lat, lng = vehicle.position
            if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
                x = int((lat - min_lat) / (max_lat - min_lat) * (grid_size - 1))
                y = int((lng - min_lng) / (max_lng - min_lng) * (grid_size - 1))
                heatmap[x, y] += 1

        # Smooth the heatmap
        heatmap = gaussian_filter(heatmap, sigma=1.5)

        # Normalize
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)

        self.heatmap_data = {
            "grid": heatmap.tolist(),
            "bounds": [min_lat, min_lng, max_lat, max_lng],
        }

    def predict_traffic(self):
        """Predict future traffic based on current patterns"""
        if (
            not self.stats["time_series"]["vehicle_count"]
            or len(self.stats["time_series"]["vehicle_count"]) < 5
        ):
            return

        # Simple prediction based on recent trend
        recent_counts = self.stats["time_series"]["vehicle_count"][-5:]
        slope = (recent_counts[-1] - recent_counts[0]) / 5

        # Predict next 5 time steps
        predictions = []
        last_value = recent_counts[-1]
        for i in range(5):
            next_value = last_value + slope
            next_value = max(0, next_value)  # Ensure non-negative
            predictions.append(next_value)
            last_value = next_value

        self.traffic_prediction = predictions

    def apply_weather_effects(self):
        """Apply weather effects to simulation parameters"""
        reliability_factor = 1.0
        speed_factor = 1.0

        if self.weather_condition == WeatherCondition.RAIN:
            reliability_factor = 0.85
            speed_factor = 0.8
        elif self.weather_condition == WeatherCondition.SNOW:
            reliability_factor = 0.7
            speed_factor = 0.6
        elif self.weather_condition == WeatherCondition.FOG:
            reliability_factor = 0.75
            speed_factor = 0.7

        if self.time_of_day == "NIGHT":
            reliability_factor *= 0.9
            speed_factor *= 0.85

        # Apply effects
        base_reliability = self.config["communication_reliability"]
        self.config["communication_reliability_effective"] = (
            base_reliability * reliability_factor
        )

        # Adjust vehicle speeds
        for vehicle in self.vehicles:
            if not hasattr(vehicle, "base_speed"):
                vehicle.base_speed = vehicle.speed

            # Don't modify speed of vehicles involved in accidents
            if any(
                not accident.cleared and vehicle.id in accident.vehicles_involved
                for accident in self.accidents
            ):
                continue

            vehicle.speed = vehicle.base_speed * speed_factor

    def _update_statistics(self):
        # Update simulation statistics
        self.stats["vehicle_count"] = len(self.vehicles)

        if self.vehicles:
            self.stats["average_speed"] = sum(v.speed for v in self.vehicles) / len(
                self.vehicles
            )

        # Calculate traffic density (vehicles per km of road)
        total_road_length = (
            len(self.road_network.edges) * 0.5
        )  # Assuming 0.5km per edge
        if total_road_length > 0:
            self.stats["traffic_density"] = len(self.vehicles) / total_road_length

        # Message success rate
        total_possible = (
            len(self.vehicles) * (len(self.vehicles) - 1) / 2
        )  # All possible V2V pairs
        total_possible += len(self.vehicles) * len(
            self.infrastructure
        )  # All possible V2I pairs
        if total_possible > 0:
            self.stats["message_success_rate"] = min(
                1.0, self.stats["messages_sent"] / (total_possible * self.time)
            )

        # Update time series data (for charts)
        if self.time % 5 == 0:  # Every 5 steps
            self.stats["time_series"]["vehicle_count"].append(
                self.stats["vehicle_count"]
            )
            self.stats["time_series"]["message_count"].append(
                self.stats["messages_sent"]
            )
            self.stats["time_series"]["average_speed"].append(
                self.stats["average_speed"]
            )

            # Keep only the most recent 50 time points
            if len(self.stats["time_series"]["vehicle_count"]) > 50:
                for key in self.stats["time_series"]:
                    self.stats["time_series"][key] = self.stats["time_series"][key][
                        -50:
                    ]

    def _get_simulation_state(self):
        # Return current simulation state for frontend
        state = {
            "time": self.time,
            "vehicles": [v.to_dict() for v in self.vehicles],
            "infrastructure": [i.to_dict() for i in self.infrastructure],
            "messages": [m.to_dict() for m in self.active_messages],
            "stats": self.stats,
            "network": self.road_network.to_dict(),
            "weather": self.weather_condition.name,
            "time_of_day": self.time_of_day,
            "accidents": [a.to_dict() for a in self.accidents if not a.cleared],
            "security_events": [
                e.to_dict() for e in self.security_events if not e.resolved
            ],
            "platoons": self.platoons,
            "heatmap": self.heatmap_data,
            "traffic_prediction": self.traffic_prediction,
        }

        return state

    def update_config(self, config_updates):
        # Update configuration parameters
        for key, value in config_updates.items():
            if key in self.config:
                self.config[key] = value

    def get_config(self):
        return self.config

    def start(self, data=None):
        if not self.running:
            self.running = True
            scenario = data.get("scenario_type", "urban") if data else "urban"
            self.initialize_scenario(scenario)

    def pause(self):
        self.running = False

    def reset(self):
        self.running = False
        self.vehicles = []
        self.infrastructure = []
        self.messages = []
        self.active_messages = []
        self.time = 0
        self.message_id_counter = 0
        self.accidents = []
        self.security_events = []
        self.platoons = {}
        self.heatmap_data = None
        self.traffic_prediction = None
        self.stats = {
            "messages_sent": 0,
            "message_types": {},
            "vehicle_count": 0,
            "average_speed": 0,
            "traffic_density": 0,
            "message_success_rate": 0,
            "time_series": {
                "vehicle_count": [],
                "message_count": [],
                "average_speed": [],
            },
        }

    def is_running(self):
        return self.running

    def load_scenario(self, scenario_name):
        """Load a predefined scenario"""
        for scenario in self.scenarios:
            if scenario.name == scenario_name:
                self.reset()

                # Apply scenario settings
                self.config["vehicle_density"] = scenario.vehicle_density
                self.weather_condition = scenario.weather
                self.time_of_day = scenario.time_of_day

                # Initialize with scenario type
                self.initialize_scenario(scenario.map_type)
                return True
        return False

    def add_custom_vehicle(self, vehicle_type, position, direction, speed):
        """Add a custom vehicle at a specific location"""
        veh_type = VehicleType[vehicle_type]
        new_id = max([v.id for v in self.vehicles], default=-1) + 1
        vehicle = Vehicle(new_id, veh_type, position, direction, speed)
        self.vehicles.append(vehicle)
        return new_id


# ========== Flask Web Application ==========

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Create simulator instance
simulator = TrafficSimulator()

# HTML template for the application (as a string to keep everything in one file)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced V2X Simulation Platform</title>
    
    <!-- External CSS libraries -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css">
    
    <style>
        /* Custom styling */
        body {
            background-color: #f8f9fa;
            color: #333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        
        .navbar {
            background-color: #343a40;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .navbar-brand {
            font-weight: 600;
        }
        
        .card {
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            border: none;
        }
        
        .card-header {
            background-color: #f1f3f5;
            font-weight: 500;
            border-bottom: 1px solid #e9ecef;
        }
        
        #map-container {
            height: 600px;
            border-radius: 4px;
            overflow: hidden;
        }
        
        .btn {
            font-weight: 500;
            border-radius: 4px;
        }
        
        /* Vehicle markers */
        .vehicle-marker {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            border: 1px solid white;
            box-shadow: 0 0 3px rgba(0,0,0,0.3);
        }
        
        .vehicle-marker.CAR {
            background-color: #4285F4;
        }
        
        .vehicle-marker.TRUCK {
            background-color: #34a853;
        }
        
        .vehicle-marker.BUS {
            background-color: #fbbc05;
        }
        
        .vehicle-marker.EMERGENCY {
            background-color: #ea4335;
        }
        
        /* Infrastructure markers */
        .infra-marker {
            width: 14px;
            height: 14px;
            border: 1px solid white;
            box-shadow: 0 0 3px rgba(0,0,0,0.3);
        }
        
        .infra-marker.TRAFFIC_LIGHT {
            background-color: #9c27b0;
            border-radius: 0;
            transform: rotate(45deg);
        }
        
        .infra-marker.RSU {
            background-color: #00bcd4;
            border-radius: 3px;
        }
        
        /* Message visualization */
        .message-path {
            stroke-dasharray: 5, 5;
            animation: dash 1s linear infinite;
        }
        
        @keyframes dash {
            to {
                stroke-dashoffset: -10;
            }
        }
        
        .message-dot {
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0% { r: 3; opacity: 1; }
            100% { r: 8; opacity: 0; }
        }
        
        .stat-box {
            padding: 10px;
            border-radius: 4px;
            margin-bottom: 10px;
            text-align: center;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: 500;
            margin: 5px 0;
        }
        
        .stat-label {
            font-size: 14px;
            color: #6c757d;
        }
        
        .log-container {
            height: 200px;
            overflow-y: auto;
            background-color: #f5f5f5;
            border-radius: 4px;
            padding: 10px;
            font-family: monospace;
            font-size: 12px;
        }
        
        .legend-item {
            display: flex;
            align-items: center;
            margin-bottom: 5px;
        }
        
        .legend-color {
            width: 12px;
            height: 12px;
            margin-right: 8px;
            border-radius: 50%;
            border: 1px solid white;
            box-shadow: 0 0 2px rgba(0,0,0,0.2);
        }
        
        .legend-text {
            font-size: 12px;
        }
        
        .message-log {
            margin-bottom: 5px;
            padding: 5px;
            border-radius: 3px;
            background-color: #f8f9fa;
            border-left: 3px solid #6c757d;
        }
        
        .message-log.SAFETY_WARNING {
            border-left-color: #dc3545;
        }
        
        .message-log.TRAFFIC_INFO {
            border-left-color: #17a2b8;
        }
        
        .message-log.V2V_POSITION {
            border-left-color: #4285F4;
        }
        
        .message-log.I2V_SIGNAL {
            border-left-color: #9c27b0;
        }
        
        .message-log.V2I_REQUEST {
            border-left-color: #34a853;
        }
        
        /* Visualization badges for enhanced features */
        .weather-badge {
            position: absolute;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 20px;
            color: white;
            font-weight: 500;
            font-size: 14px;
            z-index: 1000;
        }
        
        .weather-badge.CLEAR {
            background-color: #17a2b8;
        }
        
        .weather-badge.RAIN {
            background-color: #0d6efd;
        }
        
        .weather-badge.SNOW {
            background-color: #6c757d;
        }
        
        .weather-badge.FOG {
            background-color: #adb5bd;
        }
        
        .time-badge {
            position: absolute;
            top: 50px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 20px;
            font-weight: 500;
            font-size: 14px;
            z-index: 1000;
        }
        
        .time-badge.DAY {
            background-color: #ffc107;
            color: #212529;
        }
        
        .time-badge.NIGHT {
            background-color: #212529;
            color: white;
        }
        
        .platoon-indicator {
            position: absolute;
            width: 8px;
            height: 8px;
            background-color: #00ff00;
            border-radius: 50%;
            top: -4px;
            right: -4px;
            border: 1px solid white;
        }
        
        .accident-icon {
            color: #ff0000;
            font-size: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        /* Weather overlay effects */
        #weather-overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1000;
            opacity: 0.3;
        }
        
        #weather-overlay.RAIN {
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><line x1="20" y1="20" x2="10" y2="40" stroke="blue" stroke-width="1"/><line x1="40" y1="10" x2="30" y2="30" stroke="blue" stroke-width="1"/><line x1="60" y1="25" x2="50" y2="45" stroke="blue" stroke-width="1"/><line x1="80" y1="15" x2="70" y2="35" stroke="blue" stroke-width="1"/><line x1="30" y1="50" x2="20" y2="70" stroke="blue" stroke-width="1"/><line x1="50" y1="40" x2="40" y2="60" stroke="blue" stroke-width="1"/><line x1="70" y1="55" x2="60" y2="75" stroke="blue" stroke-width="1"/><line x1="90" y1="45" x2="80" y2="65" stroke="blue" stroke-width="1"/><line x1="10" y1="75" x2="0" y2="95" stroke="blue" stroke-width="1"/><line x1="30" y1="80" x2="20" y2="100" stroke="blue" stroke-width="1"/><line x1="50" y1="70" x2="40" y2="90" stroke="blue" stroke-width="1"/><line x1="70" y1="85" x2="60" y2="105" stroke="blue" stroke-width="1"/><line x1="90" y1="75" x2="80" y2="95" stroke="blue" stroke-width="1"/></svg>');
            animation: rain 0.5s linear infinite;
        }
        
        #weather-overlay.SNOW {
            background-image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="white"/><circle cx="40" cy="10" r="2" fill="white"/><circle cx="60" cy="25" r="2" fill="white"/><circle cx="80" cy="15" r="2" fill="white"/><circle cx="30" cy="50" r="2" fill="white"/><circle cx="50" cy="40" r="2" fill="white"/><circle cx="70" cy="55" r="2" fill="white"/><circle cx="90" cy="45" r="2" fill="white"/><circle cx="10" cy="75" r="2" fill="white"/><circle cx="30" cy="80" r="2" fill="white"/><circle cx="50" cy="70" r="2" fill="white"/><circle cx="70" cy="85" r="2" fill="white"/><circle cx="90" cy="75" r="2" fill="white"/></svg>');
            animation: snow 5s linear infinite;
        }
        
        #weather-overlay.FOG {
            background-color: rgba(200, 200, 200, 0.6);
        }
        
        #weather-overlay.NIGHT {
            background-color: rgba(0, 0, 30, 0.3);
        }
        
        @keyframes rain {
            0% { background-position: 0 0; }
            100% { background-position: 0 20px; }
        }
        
        @keyframes snow {
            0% { background-position: 0 0; }
            100% { background-position: 20px 100px; }
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            #map-container {
                height: 400px;
            }
            
            .card-body {
                padding: 10px;
            }
            
            .stat-value {
                font-size: 18px;
            }
        }
    </style>
</head>
<body>
    <div class="container-fluid px-4 py-3">
        <nav class="navbar navbar-expand-lg navbar-dark rounded mb-4">
            <div class="container-fluid">
                <a class="navbar-brand" href="#">Enhanced V2X Simulation Platform</a>
                <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                    <span class="navbar-toggler-icon"></span>
                </button>
                <div class="collapse navbar-collapse" id="navbarNav">
                    <ul class="navbar-nav">
                        <li class="nav-item">
                            <a class="nav-link active" href="#simulation">Simulation</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="#analytics">Analytics</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="#messages">Messages</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
        
        <div class="row" id="simulation">
            <!-- Map visualization -->
            <div class="col-lg-9">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Simulation View</h5>
                        <div>
                            <button id="toggle-network" class="btn btn-sm btn-outline-secondary">Toggle Network</button>
                            <button id="toggle-messages" class="btn btn-sm btn-outline-secondary">Toggle Messages</button>
                            <button id="show-3d-view" class="btn btn-sm btn-outline-primary ms-2">3D View</button>
                            <button id="toggle-heatmap" class="btn btn-sm btn-outline-danger ms-2">Traffic Heatmap</button>
                            <button id="add-vehicle-btn" class="btn btn-sm btn-outline-success ms-2">Add Vehicle</button>
                            <button id="show-advanced-settings" class="btn btn-sm btn-outline-dark ms-2">Advanced Settings</button>
                        </div>
                    </div>
                    <div class="card-body p-0 position-relative">
                        <div id="map-container"></div>
                        <div id="weather-overlay"></div>
                        <div id="weather-badge" class="weather-badge CLEAR">Clear</div>
                        <div id="time-badge" class="time-badge DAY">Day</div>
                    </div>
                </div>
                
                <!-- Legend -->
                <div class="card mt-3">
                    <div class="card-header">
                        <h5 class="mb-0">Legend</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <h6>Vehicles</h6>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #4285F4;"></div>
                                    <div class="legend-text">Car</div>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #34a853;"></div>
                                    <div class="legend-text">Truck</div>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #fbbc05;"></div>
                                    <div class="legend-text">Bus</div>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #ea4335;"></div>
                                    <div class="legend-text">Emergency</div>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #4285F4; position: relative;">
                                        <div class="platoon-indicator"></div>
                                    </div>
                                    <div class="legend-text">Platooning</div>
                                </div>
                            </div>
                            <div class="col-md-3">
                                <h6>Infrastructure</h6>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #9c27b0; border-radius: 0; transform: rotate(45deg);"></div>
                                    <div class="legend-text">Traffic Light</div>
                                </div>
                                <div class="legend-item">
                                    <div class="legend-color" style="background-color: #00bcd4; border-radius: 3px;"></div>
                                    <div class="legend-text">RSU</div>
                                </div>
                                <div class="legend-item mt-2">
                                    <div class="accident-icon">⚠️</div>
                                    <div class="legend-text ms-2">Accident</div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <h6>Message Types</h6>
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="legend-item">
                                            <div class="legend-color" style="background-color: #dc3545;"></div>
                                            <div class="legend-text">Safety Warning</div>
                                        </div>
                                        <div class="legend-item">
                                            <div class="legend-color" style="background-color: #17a2b8;"></div>
                                            <div class="legend-text">Traffic Info</div>
                                        </div>
                                        <div class="legend-item">
                                            <div class="legend-color" style="background-color: #4285F4;"></div>
                                            <div class="legend-text">V2V Position</div>
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="legend-item">
                                            <div class="legend-color" style="background-color: #9c27b0;"></div>
                                            <div class="legend-text">I2V Signal</div>
                                        </div>
                                        <div class="legend-item">
                                            <div class="legend-color" style="background-color: #34a853;"></div>
                                            <div class="legend-text">V2I Request</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Control panel -->
            <div class="col-lg-3">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Simulation Controls</h5>
                    </div>
                    <div class="card-body">
                        <!-- Control buttons -->
                        <div class="d-grid gap-2 mb-3">
                            <button id="start-btn" class="btn btn-success">Start</button>
                            <button id="pause-btn" class="btn btn-warning">Pause</button>
                            <button id="reset-btn" class="btn btn-danger">Reset</button>
                        </div>
                        
                        <hr>
                        
                        <!-- Scenario selection -->
                        <div class="mb-3">
                            <label for="scenario-select" class="form-label">Scenario</label>
                            <select id="scenario-select" class="form-select">
                                <option value="urban">Urban Grid</option>
                                <option value="highway">Highway</option>
                                <option value="rural">Rural Roads</option>
                            </select>
                        </div>
                        
                        <!-- Parameter sliders -->
                        <div class="mb-3">
                            <label for="density-slider" class="form-label">Vehicle Density: <span id="density-value">0.1</span></label>
                            <input type="range" class="form-range" id="density-slider" min="0.01" max="0.5" step="0.01" value="0.1">
                        </div>
                        
                        <div class="mb-3">
                            <label for="comm-range-slider" class="form-label">Communication Range: <span id="comm-range-value">300</span>m</label>
                            <input type="range" class="form-range" id="comm-range-slider" min="50" max="1000" step="50" value="300">
                        </div>
                        
                        <div class="mb-3">
                            <label for="reliability-slider" class="form-label">Communication Reliability: <span id="reliability-value">0.95</span></label>
                            <input type="range" class="form-range" id="reliability-slider" min="0.5" max="1" step="0.01" value="0.95">
                        </div>
                        
                        <div class="mb-3">
                            <label for="speed-slider" class="form-label">Simulation Speed: <span id="speed-value">1x</span></label>
                            <input type="range" class="form-range" id="speed-slider" min="0.1" max="5" step="0.1" value="1">
                        </div>
                    </div>
                </div>
                
                <!-- Real-time statistics -->
                <div class="card mt-3">
                    <div class="card-header">
                        <h5 class="mb-0">Real-time Statistics</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-6">
                                <div class="stat-box">
                                    <div class="stat-value" id="vehicle-count">0</div>
                                    <div class="stat-label">Vehicles</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="stat-box">
                                    <div class="stat-value" id="message-count">0</div>
                                    <div class="stat-label">Messages</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="stat-box">
                                    <div class="stat-value" id="avg-speed">0</div>
                                    <div class="stat-label">Avg. Speed (km/h)</div>
                                </div>
                            </div>
                            <div class="col-6">
                                <div class="stat-box">
                                    <div class="stat-value" id="message-success-rate">0%</div>
                                    <div class="stat-label">Msg Success Rate</div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Enhanced status indicators -->
                        <div class="mt-3">
                            <h6>Environmental Conditions</h6>
                            <div class="d-flex justify-content-between mb-2">
                                <span>Weather:</span>
                                <span id="weather-status">Clear</span>
                            </div>
                            <div class="d-flex justify-content-between mb-2">
                                <span>Time of Day:</span>
                                <span id="time-of-day-status">Day</span>
                            </div>
                            <h6 class="mt-3">Network Status</h6>
                            <div class="d-flex justify-content-between mb-2">
                                <span>Protocol:</span>
                                <span id="protocol-status">DSRC</span>
                            </div>
                            <div class="d-flex justify-content-between mb-2">
                                <span>Latency:</span>
                                <span id="latency-status">20ms</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Advanced monitoring -->
                <div class="card mt-3">
                    <div class="card-header">
                        <h5 class="mb-0">Advanced Monitoring</h5>
                    </div>
                    <div class="card-body">
                        <div class="d-flex justify-content-between mb-2">
                            <span>Platoons:</span>
                            <span id="platoon-count">0</span>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span>Accidents:</span>
                            <span id="accident-count">0</span>
                        </div>
                        <div class="d-flex justify-content-between mb-2">
                            <span>Security Alerts:</span>
                            <span id="security-count">0</span>
                        </div>
                        
                        <h6 class="mt-3">Traffic Prediction</h6>
                        <div id="prediction-container" class="text-center">
                            <small class="text-muted">Not enough data yet</small>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Analytics dashboard -->
        <div class="row mt-4" id="analytics">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h5 class="mb-0">Analytics Dashboard</h5>
                        <div class="btn-group">
                            <button class="btn btn-sm btn-outline-secondary active" id="view-basic-charts">Basic</button>
                            <button class="btn btn-sm btn-outline-secondary" id="view-advanced-charts">Advanced</button>
                        </div>
                    </div>
                    <div class="card-body">
                        <!-- Basic charts view -->
                        <div id="basic-charts">
                            <div class="row">
                                <div class="col-md-6">
                                    <canvas id="traffic-chart" height="250"></canvas>
                                </div>
                                <div class="col-md-6">
                                    <canvas id="message-chart" height="250"></canvas>
                                </div>
                            </div>
                            <div class="row mt-4">
                                <div class="col-md-6">
                                    <canvas id="speed-chart" height="250"></canvas>
                                </div>
                                <div class="col-md-6">
                                    <canvas id="message-types-chart" height="250"></canvas>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Advanced charts view (hidden by default) -->
                        <div id="advanced-charts" style="display: none;">
                            <div class="row">
                                <div class="col-md-6">
                                    <canvas id="network-performance-chart" height="250"></canvas>
                                </div>
                                <div class="col-md-6">
                                    <canvas id="vehicle-distribution-chart" height="250"></canvas>
                                </div>
                            </div>
                            <div class="row mt-4">
                                <div class="col-md-6">
                                    <canvas id="weather-impact-chart" height="250"></canvas>
                                </div>
                                <div class="col-md-6">
                                    <canvas id="security-events-chart" height="250"></canvas>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Message log -->
        <div class="row mt-4" id="messages">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Message Log</h5>
                    </div>
                    <div class="card-body">
                        <div class="log-container" id="message-log"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <footer class="mt-4 mb-2 text-center text-muted">
            <p class="small">Enhanced V2X Simulation Platform &copy; 2025</p>
        </footer>
    </div>
    
    <!-- 3D View Modal -->
    <div class="modal fade" id="3d-view-modal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-xl">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">3D Simulation View</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body p-0">
                    <div id="3d-container" style="height: 600px;"></div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    <button type="button" class="btn btn-primary" id="toggle-3d-follow">Follow Vehicle</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Advanced Settings Modal -->
    <div class="modal fade" id="advanced-settings-modal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Advanced Settings</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <ul class="nav nav-tabs" id="settingsTabs" role="tablist">
                        <li class="nav-item" role="presentation">
                            <button class="nav-link active" id="environment-tab" data-bs-toggle="tab" data-bs-target="#environment" type="button" role="tab">Environment</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="vehicles-tab" data-bs-toggle="tab" data-bs-target="#vehicles" type="button" role="tab">Vehicles</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="network-tab" data-bs-toggle="tab" data-bs-target="#network" type="button" role="tab">Network</button>
                        </li>
                        <li class="nav-item" role="presentation">
                            <button class="nav-link" id="scenarios-tab" data-bs-toggle="tab" data-bs-target="#scenarios" type="button" role="tab">Scenarios</button>
                        </li>
                    </ul>
                    <div class="tab-content p-3" id="settingsTabContent">
                        <div class="tab-pane fade show active" id="environment" role="tabpanel">
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label class="form-label">Weather Condition</label>
                                    <select id="weather-select" class="form-select">
                                        <option value="CLEAR">Clear</option>
                                        <option value="RAIN">Rain</option>
                                        <option value="SNOW">Snow</option>
                                        <option value="FOG">Fog</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Time of Day</label>
                                    <select id="time-select" class="form-select">
                                        <option value="DAY">Day</option>
                                        <option value="NIGHT">Night</option>
                                    </select>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-md-12">
                                    <div class="form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="enable-accidents" checked>
                                        <label class="form-check-label" for="enable-accidents">Enable Traffic Accidents</label>
                                    </div>
                                </div>
                                <div class="col-md-12">
                                    <div class="form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="enable-security" checked>
                                        <label class="form-check-label" for="enable-security">Enable Security Events</label>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="vehicles" role="tabpanel">
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label class="form-label">Distribution Pattern</label>
                                    <select id="distribution-select" class="form-select">
                                        <option value="random">Random</option>
                                        <option value="clustered">Clustered</option>
                                        <option value="uniform">Uniform</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Vehicle Type Ratio</label>
                                    <div class="input-group mb-3">
                                        <span class="input-group-text">Cars</span>
                                        <input type="number" class="form-control" id="car-ratio" value="70" min="0" max="100">
                                        <span class="input-group-text">%</span>
                                    </div>
                                    <div class="input-group mb-3">
                                        <span class="input-group-text">Trucks</span>
                                        <input type="number" class="form-control" id="truck-ratio" value="15" min="0" max="100">
                                        <span class="input-group-text">%</span>
                                    </div>
                                    <div class="input-group mb-3">
                                        <span class="input-group-text">Buses</span>
                                        <input type="number" class="form-control" id="bus-ratio" value="10" min="0" max="100">
                                        <span class="input-group-text">%</span>
                                    </div>
                                    <div class="input-group mb-3">
                                        <span class="input-group-text">Emergency</span>
                                        <input type="number" class="form-control" id="emergency-ratio" value="5" min="0" max="100">
                                        <span class="input-group-text">%</span>
                                    </div>
                                </div>
                            </div>
                            <div class="row">
                                <div class="col-md-12">
                                    <div class="form-check form-switch">
                                        <input class="form-check-input" type="checkbox" id="enable-platooning" checked>
                                        <label class="form-check-label" for="enable-platooning">Enable Vehicle Platooning</label>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="network" role="tabpanel">
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label class="form-label">Communication Protocol</label>
                                    <select id="protocol-select" class="form-select">
                                        <option value="DSRC">DSRC (Dedicated Short Range Communications)</option>
                                        <option value="C-V2X">C-V2X (Cellular Vehicle-to-Everything)</option>
                                        <option value="HYBRID">Hybrid (DSRC + C-V2X)</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label for="latency-slider" class="form-label">Network Latency: <span id="latency-value">20</span>ms</label>
                                    <input type="range" class="form-range" id="latency-slider" min="5" max="200" step="5" value="20">
                                </div>
                            </div>
                            <div class="row mb-3">
                                <div class="col-md-6">
                                    <label for="bandwidth-slider" class="form-label">Bandwidth: <span id="bandwidth-value">10</span> Mbps</label>
                                    <input type="range" class="form-range" id="bandwidth-slider" min="1" max="100" step="1" value="10">
                                </div>
                                <div class="col-md-6">
                                    <label for="packet-loss-slider" class="form-label">Packet Loss: <span id="packet-loss-value">2</span>%</label>
                                    <input type="range" class="form-range" id="packet-loss-slider" min="0" max="20" step="0.5" value="2">
                                </div>
                            </div>
                        </div>
                        <div class="tab-pane fade" id="scenarios" role="tabpanel">
                            <div class="list-group" id="scenario-list">
                                <a href="#" class="list-group-item list-group-item-action" data-scenario="Urban Rush Hour">
                                    <div class="d-flex w-100 justify-content-between">
                                        <h5 class="mb-1">Urban Rush Hour</h5>
                                        <span class="badge bg-primary">Urban</span>
                                    </div>
                                    <p class="mb-1">Heavy traffic in city center during rush hour</p>
                                </a>
                                <a href="#" class="list-group-item list-group-item-action" data-scenario="Highway Night Travel">
                                    <div class="d-flex w-100 justify-content-between">
                                        <h5 class="mb-1">Highway Night Travel</h5>
                                        <span class="badge bg-dark">Night</span>
                                    </div>
                                    <p class="mb-1">Highway scenario at night with medium traffic</p>
                                </a>
                                <a href="#" class="list-group-item list-group-item-action" data-scenario="Rainy City Conditions">
                                    <div class="d-flex w-100 justify-content-between">
                                        <h5 class="mb-1">Rainy City Conditions</h5>
                                        <span class="badge bg-info">Rain</span>
                                    </div>
                                    <p class="mb-1">Urban area during heavy rain</p>
                                </a>
                                <a href="#" class="list-group-item list-group-item-action" data-scenario="Winter Highway">
                                    <div class="d-flex w-100 justify-content-between">
                                        <h5 class="mb-1">Winter Highway</h5>
                                        <span class="badge bg-light text-dark">Snow</span>
                                    </div>
                                    <p class="mb-1">Highway during snowfall with reduced visibility</p>
                                </a>
                                <a href="#" class="list-group-item list-group-item-action" data-scenario="Foggy Rural Roads">
                                    <div class="d-flex w-100 justify-content-between">
                                        <h5 class="mb-1">Foggy Rural Roads</h5>
                                        <span class="badge bg-secondary">Fog</span>
                                    </div>
                                    <p class="mb-1">Rural roads with dense fog</p>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    <button type="button" class="btn btn-primary" id="apply-settings">Apply Settings</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Vehicle Inspector Modal -->
    <div class="modal fade" id="vehicle-inspector-modal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Vehicle Inspector</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div id="vehicle-details">
                        <p>Select a vehicle on the map to inspect it.</p>
                    </div>
                    <div id="vehicle-messages" class="mt-3">
                        <h6>Recent Messages</h6>
                        <div class="log-container" style="height: 150px;" id="vehicle-message-log"></div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    <button type="button" class="btn btn-primary" id="follow-vehicle">Follow this Vehicle</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Add Vehicle Modal -->
    <div class="modal fade" id="add-vehicle-modal" tabindex="-1" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">Add Custom Vehicle</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <div class="mb-3">
                        <label class="form-label">Vehicle Type</label>
                        <select id="custom-vehicle-type" class="form-select">
                            <option value="CAR">Car</option>
                            <option value="TRUCK">Truck</option>
                            <option value="BUS">Bus</option>
                            <option value="EMERGENCY">Emergency</option>
                        </select>
                    </div>
                    <div class="mb-3">
                        <p>Click on the map to select the position or enter coordinates:</p>
                        <div class="input-group mb-3">
                            <span class="input-group-text">Latitude</span>
                            <input type="number" class="form-control" id="custom-vehicle-lat" step="0.0001">
                        </div>
                        <div class="input-group mb-3">
                            <span class="input-group-text">Longitude</span>
                            <input type="number" class="form-control" id="custom-vehicle-lng" step="0.0001">
                        </div>
                    </div>
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label class="form-label">Direction (degrees)</label>
                            <input type="number" class="form-control" id="custom-vehicle-direction" value="0" min="0" max="359">
                        </div>
                        <div class="col-md-6">
                            <label class="form-label">Speed (km/h)</label>
                            <input type="number" class="form-control" id="custom-vehicle-speed" value="50" min="0" max="200">
                        </div>
                    </div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Cancel</button>
                    <button type="button" class="btn btn-primary" id="add-custom-vehicle">Add Vehicle</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Event Monitor -->
    <div class="toast-container position-fixed bottom-0 end-0 p-3">
        <div id="accident-toast" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header bg-danger text-white">
                <strong class="me-auto">Traffic Accident</strong>
                <small id="accident-time">just now</small>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                <p id="accident-details">A traffic accident has occurred.</p>
                <button type="button" class="btn btn-sm btn-outline-danger" id="view-accident">View on Map</button>
            </div>
        </div>
        
        <div id="security-toast" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
            <div class="toast-header bg-warning text-dark">
                <strong class="me-auto">Security Alert</strong>
                <small id="security-time">just now</small>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>
            <div class="toast-body">
                <p id="security-details">A security event has been detected.</p>
            </div>
        </div>
    </div>
    
    <!-- JavaScript libraries -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.132.2/examples/js/controls/OrbitControls.js"></script>
    
    <script>
        // Initialize socket connection
        const socket = io();
        
        // Global variables
        let map;
        let vehicleMarkers = {};
        let infraMarkers = {};
        let messageLines = [];
        let roadNetwork = null;
        let networkLayerGroup;
        let showNetworkLayer = true;
        let showMessages = true;
        let charts = {};
        
        // Global variables for enhanced features
        let threeJsRenderer;
        let threeJsScene;
        let threeJsCamera;
        let threeJsControls;
        let threeJsVehicles = {};
        let threeJsInfrastructure = {};
        let threeJsMessages = [];
        let threeJsFollowMode = false;
        let threeJsFollowVehicleId = null;

        let heatmapLayer = null;
        let showHeatmap = false;

        let inspectedVehicleId = null;
        let addingCustomVehicle = false;
        let customVehiclePosition = null;

        let weatherEffect = null;
        let enableAccidents = true;
        let enableSecurityEvents = true;
        let enablePlatooning = true;

        let vehicleRatios = {
            CAR: 70,
            TRUCK: 15,
            BUS: 10,
            EMERGENCY: 5
        };
        
        // Initialize application
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize map
            initializeMap();
            
            // Initialize charts
            initializeCharts();
            
            // Initialize controls
            initializeControls();
            
            // Initialize heatmap
            initializeHeatmap();
            
            // Setup custom vehicle addition
            setupCustomVehicleAddition();
            
            // Setup advanced settings
            setupAdvancedSettings();
            
            // Setup chart toggle
            setupChartToggle();
            
            // Socket event handlers
            socket.on('connect', function() {
                console.log('Connected to simulation server');
            });
            
            socket.on('simulation_update', function(data) {
                updateSimulation(data);
            });
            
            socket.on('vehicle_details', function(data) {
                showVehicleInspector(data.vehicle_id, [data.vehicle]);
            });

            socket.on('custom_vehicle_added', function(data) {
                console.log('Custom vehicle added:', data.vehicle_id);
                showToast(`Vehicle ${data.vehicle_id} added successfully`, 'success');
            });
            
            socket.on('scenario_loaded', function(data) {
                console.log('Scenario loaded:', data.name);
                if (data.success) {
                    showToast(`Scenario "${data.name}" loaded successfully.`, 'success');
                }
            });
        });
        
        // Map initialization
        function initializeMap() {
            // Create Leaflet map
            map = L.map('map-container').setView([37.7749, -122.4194], 15);
            
            // Add tile layer (map background)
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '&copy; OpenStreetMap contributors'
            }).addTo(map);
            
            // Create layer group for road network
            networkLayerGroup = L.layerGroup().addTo(map);
            
            // Set up map click handler for vehicle inspection
            map.on('click', function(e) {
                // Check if we're adding a custom vehicle
                if (addingCustomVehicle) {
                    customVehiclePosition = e.latlng;
                    document.getElementById('custom-vehicle-lat').value = e.latlng.lat.toFixed(6);
                    document.getElementById('custom-vehicle-lng').value = e.latlng.lng.toFixed(6);
                    return;
                }
                
                // Find closest vehicle within a threshold
                const clickedPoint = e.latlng;
                let closestVehicle = null;
                let minDistance = 0.0001; // Threshold in coordinate units
                
                Object.keys(vehicleMarkers).forEach(id => {
                    const vehiclePosition = vehicleMarkers[id].getLatLng();
                    const distance = clickedPoint.distanceTo(vehiclePosition);
                    
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestVehicle = parseInt(id);
                    }
                });
                
                if (closestVehicle !== null) {
                    // Request full vehicle data
                    socket.emit('get_vehicle_details', { vehicle_id: closestVehicle });
                }
            });
        }
        
        // Charts initialization
        function initializeCharts() {
            // Traffic chart (Vehicle count over time)
            charts.traffic = new Chart(
                document.getElementById('traffic-chart'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Vehicle Count',
                            data: [],
                            borderColor: '#4285F4',
                            backgroundColor: 'rgba(66, 133, 244, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Vehicle Count Over Time'
                            },
                            legend: {
                                position: 'top',
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                }
            );
            
            // Message chart (Message count over time)
            charts.messages = new Chart(
                document.getElementById('message-chart'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Message Count',
                            data: [],
                            borderColor: '#34a853',
                            backgroundColor: 'rgba(52, 168, 83, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Messages Exchanged Over Time'
                            },
                            legend: {
                                position: 'top',
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                }
            );
            
            // Speed chart (Average speed over time)
            charts.speed = new Chart(
                document.getElementById('speed-chart'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Average Speed (km/h)',
                            data: [],
                            borderColor: '#fbbc05',
                            backgroundColor: 'rgba(251, 188, 5, 0.1)',
                            tension: 0.4,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Average Vehicle Speed Over Time'
                            },
                            legend: {
                                position: 'top',
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                }
            );
            
            // Message types chart (Pie chart of message types)
            charts.messageTypes = new Chart(
                document.getElementById('message-types-chart'),
                {
                    type: 'doughnut',
                    data: {
                        labels: [],
                        datasets: [{
                            data: [],
                            backgroundColor: [
                                '#dc3545',  // SAFETY_WARNING
                                '#17a2b8',  // TRAFFIC_INFO
                                '#4285F4',  // V2V_POSITION
                                '#9c27b0',  // I2V_SIGNAL
                                '#34a853',  // V2I_REQUEST
                                '#6c757d'   // Others
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Message Types Distribution'
                            },
                            legend: {
                                position: 'right',
                            }
                        }
                    }
                }
            );
            
            // Advanced charts
            charts.networkPerformance = new Chart(
                document.getElementById('network-performance-chart'),
                {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Message Success Rate',
                            data: [],
                            borderColor: '#9c27b0',
                            backgroundColor: 'rgba(156, 39, 176, 0.1)',
                            tension: 0.4,
                            fill: true,
                            yAxisID: 'y'
                        }, {
                            label: 'Latency (ms)',
                            data: [],
                            borderColor: '#fd7e14',
                            backgroundColor: 'rgba(253, 126, 20, 0.1)',
                            tension: 0.4,
                            fill: false,
                            yAxisID: 'y1'
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Network Performance'
                            },
                            legend: {
                                position: 'top',
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                type: 'linear',
                                display: true,
                                position: 'left',
                                title: {
                                    display: true,
                                    text: 'Success Rate'
                                },
                                max: 1
                            },
                            y1: {
                                beginAtZero: true,
                                type: 'linear',
                                display: true,
                                position: 'right',
                                title: {
                                    display: true,
                                    text: 'Latency (ms)'
                                },
                                grid: {
                                    drawOnChartArea: false
                                }
                            }
                        }
                    }
                }
            );
            
            // Vehicle Distribution Chart
            charts.vehicleDistribution = new Chart(
                document.getElementById('vehicle-distribution-chart'),
                {
                    type: 'pie',
                    data: {
                        labels: ['Cars', 'Trucks', 'Buses', 'Emergency'],
                        datasets: [{
                            data: [70, 15, 10, 5],
                            backgroundColor: [
                                '#4285F4',  // CAR
                                '#34a853',  // TRUCK
                                '#fbbc05',  // BUS
                                '#ea4335'   // EMERGENCY
                            ]
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Vehicle Type Distribution'
                            },
                            legend: {
                                position: 'right',
                            }
                        }
                    }
                }
            );
            
            // Weather Impact Chart
            charts.weatherImpact = new Chart(
                document.getElementById('weather-impact-chart'),
                {
                    type: 'bar',
                    data: {
                        labels: ['Clear', 'Rain', 'Snow', 'Fog'],
                        datasets: [{
                            label: 'Avg. Speed (km/h)',
                            data: [50, 40, 30, 35],
                            backgroundColor: 'rgba(66, 133, 244, 0.6)',
                            borderColor: '#4285F4',
                            borderWidth: 1
                        }, {
                            label: 'Message Success %',
                            data: [95, 80, 65, 75],
                            backgroundColor: 'rgba(52, 168, 83, 0.6)',
                            borderColor: '#34a853',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Weather Impact on Performance'
                            },
                            legend: {
                                position: 'top',
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                }
            );
            
            // Security Events Chart
            charts.securityEvents = new Chart(
                document.getElementById('security-events-chart'),
                {
                    type: 'bar',
                    data: {
                        labels: ['Jamming', 'Spoofing', 'Eavesdropping', 'MITM'],
                        datasets: [{
                            label: 'Count',
                            data: [0, 0, 0, 0],
                            backgroundColor: 'rgba(234, 67, 53, 0.6)',
                            borderColor: '#ea4335',
                            borderWidth: 1
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            title: {
                                display: true,
                                text: 'Security Events Distribution'
                            },
                            legend: {
                                position: 'top',
                            }
                        },
                        scales: {
                            y: {
                                beginAtZero: true,
                                ticks: {
                                    stepSize: 1
                                }
                            }
                        }
                    }
                }
            );
        }
        
        // Controls initialization
        function initializeControls() {
            // Start button
            document.getElementById('start-btn').addEventListener('click', function() {
                const scenario = document.getElementById('scenario-select').value;
                socket.emit('start_simulation', { scenario_type: scenario });
            });
            
            // Pause button
            document.getElementById('pause-btn').addEventListener('click', function() {
                socket.emit('pause_simulation');
            });
            
            // Reset button
            document.getElementById('reset-btn').addEventListener('click', function() {
                socket.emit('reset_simulation');
                resetVisualization();
            });
            
            // Toggle network visibility
            document.getElementById('toggle-network').addEventListener('click', function() {
                showNetworkLayer = !showNetworkLayer;
                if (showNetworkLayer) {
                    networkLayerGroup.addTo(map);
                } else {
                    networkLayerGroup.remove();
                }
            });
            
            // Toggle messages visibility
            document.getElementById('toggle-messages').addEventListener('click', function() {
                showMessages = !showMessages;
                updateMessageVisibility();
            });
            
            // Density slider
            document.getElementById('density-slider').addEventListener('input', function() {
                const value = this.value;
                document.getElementById('density-value').textContent = value;
                socket.emit('update_parameter', {
                    parameter: 'vehicle_density',
                    value: parseFloat(value)
                });
            });
            
            // Communication range slider
            document.getElementById('comm-range-slider').addEventListener('input', function() {
                const value = this.value;
                document.getElementById('comm-range-value').textContent = value;
                socket.emit('update_parameter', {
                    parameter: 'communication_range',
                    value: parseFloat(value)
                });
            });
            
            // Reliability slider
            document.getElementById('reliability-slider').addEventListener('input', function() {
                const value = this.value;
                document.getElementById('reliability-value').textContent = value;
                socket.emit('update_parameter', {
                    parameter: 'communication_reliability',
                    value: parseFloat(value)
                });
            });
            
            // Simulation speed slider
            document.getElementById('speed-slider').addEventListener('input', function() {
                const value = this.value;
                document.getElementById('speed-value').textContent = value + 'x';
                socket.emit('update_parameter', {
                    parameter: 'simulation_speed',
                    value: parseFloat(value)
                });
            });
            
            // 3D view button
            document.getElementById('show-3d-view').addEventListener('click', function() {
                // Initialize 3D view if not done yet
                if (!threeJsScene) {
                    initialize3DView();
                }
                
                // Show the modal
                const modal = new bootstrap.Modal(document.getElementById('3d-view-modal'));
                modal.show();
            });
            
            // Toggle 3D follow mode
            document.getElementById('toggle-3d-follow').addEventListener('click', function() {
                if (threeJsFollowMode) {
                    threeJsFollowMode = false;
                    this.textContent = "Follow Vehicle";
                    
                    // Reset camera position
                    threeJsCamera.position.set(0, 500, 500);
                    threeJsCamera.lookAt(0, 0, 0);
                    threeJsControls.reset();
                } else if (threeJsFollowVehicleId !== null) {
                    threeJsFollowMode = true;
                    this.textContent = "Stop Following";
                } else {
                    showToast("Select a vehicle to follow first!", "warning");
                }
            });
            
            // Toggle heatmap
            document.getElementById('toggle-heatmap').addEventListener('click', function() {
                showHeatmap = !showHeatmap;
                
                if (showHeatmap) {
                    this.classList.remove('btn-outline-danger');
                    this.classList.add('btn-danger');
                    if (heatmapLayer) {
                        heatmapLayer.addTo(map);
                    }
                } else {
                    this.classList.remove('btn-danger');
                    this.classList.add('btn-outline-danger');
                    if (heatmapLayer) {
                        map.removeLayer(heatmapLayer);
                    }
                }
            });
        }
        
        // Setup chart toggle
        function setupChartToggle() {
            document.getElementById('view-basic-charts').addEventListener('click', function() {
                document.getElementById('basic-charts').style.display = 'block';
                document.getElementById('advanced-charts').style.display = 'none';
                this.classList.add('active');
                document.getElementById('view-advanced-charts').classList.remove('active');
            });
            
            document.getElementById('view-advanced-charts').addEventListener('click', function() {
                document.getElementById('basic-charts').style.display = 'none';
                document.getElementById('advanced-charts').style.display = 'block';
                this.classList.add('active');
                document.getElementById('view-basic-charts').classList.remove('active');
            });
        }
        
        // Initialize 3D view
        function initialize3DView() {
            const container = document.getElementById('3d-container');
            
            // Create scene
            threeJsScene = new THREE.Scene();
            threeJsScene.background = new THREE.Color(0xf0f0f0);
            
            // Create camera
            threeJsCamera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 10000);
            threeJsCamera.position.set(0, 500, 500);
            threeJsCamera.lookAt(0, 0, 0);
            
            // Create renderer
            threeJsRenderer = new THREE.WebGLRenderer({ antialias: true });
            threeJsRenderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(threeJsRenderer.domElement);
            
            // Add controls
            threeJsControls = new THREE.OrbitControls(threeJsCamera, threeJsRenderer.domElement);
            threeJsControls.enableDamping = true;
            threeJsControls.dampingFactor = 0.25;
            
            // Add lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
            threeJsScene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1000, 1000, 1000);
            threeJsScene.add(directionalLight);
            
            // Add ground plane
            const groundGeometry = new THREE.PlaneGeometry(2000, 2000, 32, 32);
            const groundMaterial = new THREE.MeshStandardMaterial({ 
                color: 0x333333,
                roughness: 0.8,
                metalness: 0.2
            });
            const ground = new THREE.Mesh(groundGeometry, groundMaterial);
            ground.rotation.x = -Math.PI / 2;
            ground.receiveShadow = true;
            threeJsScene.add(ground);
            
            // Add grid for reference
            const grid = new THREE.GridHelper(2000, 20, 0x000000, 0x000000);
            grid.material.opacity = 0.2;
            grid.material.transparent = true;
            threeJsScene.add(grid);
            
            // Handle resize
            window.addEventListener('resize', function() {
                if (threeJsCamera && threeJsRenderer) {
                    threeJsCamera.aspect = container.clientWidth / container.clientHeight;
                    threeJsCamera.updateProjectionMatrix();
                    threeJsRenderer.setSize(container.clientWidth, container.clientHeight);
                }
            });
            
            // Start animation loop
            animate3D();
        }

        // Animation loop for 3D rendering
        function animate3D() {
            requestAnimationFrame(animate3D);
            
            if (threeJsControls) {
                threeJsControls.update();
            }
            
            // Check if we're following a vehicle
            if (threeJsFollowMode && threeJsFollowVehicleId !== null && threeJsVehicles[threeJsFollowVehicleId]) {
                const vehicle = threeJsVehicles[threeJsFollowVehicleId];
                threeJsCamera.position.set(
                    vehicle.position.x - 50 * Math.sin(vehicle.rotation.y),
                    vehicle.position.y + 30,
                    vehicle.position.z - 50 * Math.cos(vehicle.rotation.y)
                );
                threeJsCamera.lookAt(vehicle.position);
            }
            
            // Render the scene
            if (threeJsRenderer && threeJsScene && threeJsCamera) {
                threeJsRenderer.render(threeJsScene, threeJsCamera);
            }
        }

        // Update 3D scene with simulation data
        function update3DScene(data) {
            if (!threeJsScene) return;
            
            // Convert lat/lng coordinates to 3D space
            function convertPosition(lat, lng) {
                // Set some center point to be 0,0 in our 3D world
                const centerLat = 37.7749;
                const centerLng = -122.4194;
                
                // Simple conversion to cartesian space (very approximate)
                const x = (lng - centerLng) * 100000;
                const z = (lat - centerLat) * 100000;
                
                return { x, z };
            }
            
            // Update vehicles
            data.vehicles.forEach(vehicle => {
                const pos = convertPosition(vehicle.position[0], vehicle.position[1]);
                
                if (threeJsVehicles[vehicle.id]) {
                    // Update existing vehicle
                    threeJsVehicles[vehicle.id].position.x = pos.x;
                    threeJsVehicles[vehicle.id].position.z = pos.z;
                    
                    // Update rotation (heading)
                    threeJsVehicles[vehicle.id].rotation.y = THREE.MathUtils.degToRad(vehicle.direction);
                } else {
                    // Create new vehicle
                    let vehicleColor;
                    let vehicleSize;
                    
                    switch (vehicle.type) {
                        case 'CAR':
                            vehicleColor = 0x4285F4;
                            vehicleSize = { length: 4.5, width: 2, height: 1.5 };
                            break;
                        case 'TRUCK':
                            vehicleColor = 0x34a853;
                            vehicleSize = { length: 8, width: 2.5, height: 3 };
                            break;
                        case 'BUS':
                            vehicleColor = 0xfbbc05;
                            vehicleSize = { length: 12, width: 2.5, height: 3 };
                            break;
                        case 'EMERGENCY':
                            vehicleColor = 0xea4335;
                            vehicleSize = { length: 5, width: 2.2, height: 2 };
                            break;
                        default:
                            vehicleColor = 0x4285F4;
                            vehicleSize = { length: 4.5, width: 2, height: 1.5 };
                    }
                    
                    // Create a vehicle mesh (simple box for now)
                    const geometry = new THREE.BoxGeometry(
                        vehicleSize.length,
                        vehicleSize.height,
                        vehicleSize.width
                    );
                    const material = new THREE.MeshStandardMaterial({ color: vehicleColor });
                    const vehicleMesh = new THREE.Mesh(geometry, material);
                    
                    // Position and orient the vehicle
                    vehicleMesh.position.set(pos.x, vehicleSize.height / 2, pos.z);
                    vehicleMesh.rotation.y = THREE.MathUtils.degToRad(vehicle.direction);
                    
                    // Add to scene and tracking object
                    threeJsScene.add(vehicleMesh);
                    threeJsVehicles[vehicle.id] = vehicleMesh;
                    
                    // Add metadata to the mesh for interactions
                    vehicleMesh.userData = {
                        id: vehicle.id,
                        type: vehicle.type,
                        speed: vehicle.speed
                    };
                }
                
                // Update platooning status
                if (vehicle.platooning) {
                    // Add visual indicator for platooning (glowing effect or connection line)
                    if (!threeJsVehicles[vehicle.id].userData.platoonIndicator) {
                        const glowMaterial = new THREE.MeshBasicMaterial({
                            color: 0x00ff00,
                            transparent: true,
                            opacity: 0.3
                        });
                        const glowGeometry = new THREE.BoxGeometry(6, 2, 3);
                        const glow = new THREE.Mesh(glowGeometry, glowMaterial);
                        threeJsVehicles[vehicle.id].add(glow);
                        threeJsVehicles[vehicle.id].userData.platoonIndicator = glow;
                    }
                } else if (threeJsVehicles[vehicle.id].userData.platoonIndicator) {
                    // Remove platooning indicator
                    threeJsVehicles[vehicle.id].remove(threeJsVehicles[vehicle.id].userData.platoonIndicator);
                    threeJsVehicles[vehicle.id].userData.platoonIndicator = null;
                }
            });
            
            // Remove vehicles that no longer exist
            Object.keys(threeJsVehicles).forEach(id => {
                if (!data.vehicles.find(v => v.id === parseInt(id))) {
                    threeJsScene.remove(threeJsVehicles[id]);
                    delete threeJsVehicles[id];
                }
            });
            
            // Update infrastructure
            data.infrastructure.forEach(infra => {
                const pos = convertPosition(infra.position[0], infra.position[1]);
                
                if (threeJsInfrastructure[infra.id]) {
                    // Update existing infrastructure (mainly position doesn't change,
                    // but state might - like traffic lights)
                    if (infra.type === "TRAFFIC_LIGHT") {
                        // Update traffic light color
                        let color = 0x4b0082; // Default purple
                        if (infra.state === "RED") color = 0xff0000;
                        else if (infra.state === "YELLOW") color = 0xffff00;
                        else if (infra.state === "GREEN") color = 0x00ff00;
                        
                        threeJsInfrastructure[infra.id].material.color.setHex(color);
                    }
                } else {
                    // Create new infrastructure
                    let geometry, material;
                    
                    if (infra.type === "TRAFFIC_LIGHT") {
                        geometry = new THREE.BoxGeometry(1, 8, 1);
                        
                        // Color based on state
                        let color = 0x4b0082; // Default purple
                        if (infra.state === "RED") color = 0xff0000;
                        else if (infra.state === "YELLOW") color = 0xffff00;
                        else if (infra.state === "GREEN") color = 0x00ff00;
                        
                        material = new THREE.MeshStandardMaterial({ color: color });
                    } else { // RSU
                        geometry = new THREE.CylinderGeometry(0.5, 0.5, 10, 8);
                        material = new THREE.MeshStandardMaterial({ color: 0x00bcd4 });
                    }
                    
                    const infraMesh = new THREE.Mesh(geometry, material);
                    
                    // Position
                    infraMesh.position.set(pos.x, 5, pos.z);
                    
                    // Add to scene and tracking object
                    threeJsScene.add(infraMesh);
                    threeJsInfrastructure[infra.id] = infraMesh;
                    
                    // Add metadata
                    infraMesh.userData = {
                        id: infra.id,
                        type: infra.type,
                        state: infra.state
                    };
                }
            });
            
            // Remove infrastructure that no longer exists
            Object.keys(threeJsInfrastructure).forEach(id => {
                if (!data.infrastructure.find(i => i.id === parseInt(id))) {
                    threeJsScene.remove(threeJsInfrastructure[id]);
                    delete threeJsInfrastructure[id];
                }
            });
            
            // Visualize messages as animated lines
            data.messages.forEach(message => {
                const sourcePos = convertPosition(message.source_position[0], message.source_position[1]);
                const destPos = convertPosition(message.destination_position[0], message.destination_position[1]);
                
                // Create points for the line
                const points = [];
                points.push(new THREE.Vector3(sourcePos.x, 3, sourcePos.z));
                points.push(new THREE.Vector3(destPos.x, 3, destPos.z));
                
                // Create line geometry
                const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
                
                // Choose color based on message type
                let color;
                switch (message.type) {
                    case "SAFETY_WARNING":
                        color = 0xdc3545;
                        break;
                    case "TRAFFIC_INFO":
                        color = 0x17a2b8;
                        break;
                    case "V2V_POSITION":
                        color = 0x4285F4;
                        break;
                    case "I2V_SIGNAL":
                        color = 0x9c27b0;
                        break;
                    case "V2I_REQUEST":
                        color = 0x34a853;
                        break;
                    default:
                        color = 0x6c757d;
                }
                
                // Create line material
                const lineMaterial = new THREE.LineBasicMaterial({
                    color: color,
                    linewidth: 2,
                    transparent: true,
                    opacity: 0.7
                });
                
                // Create line and add to scene
                const line = new THREE.Line(lineGeometry, lineMaterial);
                threeJsScene.add(line);
                threeJsMessages.push(line);
                
                // Remove the line after a short time
                setTimeout(() => {
                    threeJsScene.remove(line);
                    threeJsMessages.splice(threeJsMessages.indexOf(line), 1);
                }, 2000);
            });
            
            // Add accidents if any
            if (data.accidents) {
                data.accidents.forEach(accident => {
                    if (!threeJsScene.getObjectByName(`accident-${accident.id}`)) {
                        const pos = convertPosition(accident.position[0], accident.position[1]);
                        
                        // Create visual effect for accident
                        const geometry = new THREE.SphereGeometry(5 * accident.severity, 32, 32);
                        const material = new THREE.MeshBasicMaterial({
                            color: 0xff0000,
                            transparent: true,
                            opacity: 0.6
                        });
                        const accidentMesh = new THREE.Mesh(geometry, material);
                        accidentMesh.name = `accident-${accident.id}`;
                        accidentMesh.position.set(pos.x, 1, pos.z);
                        
                        threeJsScene.add(accidentMesh);
                        
                        // Add animated warning effect
                        const warningGeometry = new THREE.RingGeometry(5, 6, 32);
                        const warningMaterial = new THREE.MeshBasicMaterial({
                            color: 0xff0000,
                            transparent: true,
                            opacity: 0.5,
                            side: THREE.DoubleSide
                        });
                        const warning = new THREE.Mesh(warningGeometry, warningMaterial);
                        warning.name = `accident-warning-${accident.id}`;
                        warning.position.set(pos.x, 0.5, pos.z);
                        warning.rotation.x = -Math.PI / 2;
                        threeJsScene.add(warning);
                        
                        // Animate warning ring
                        function pulseWarning() {
                            warning.scale.x = 1 + Math.sin(Date.now() * 0.005) * 0.5;
                            warning.scale.y = 1 + Math.sin(Date.now() * 0.005) * 0.5;
                            warning.scale.z = 1 + Math.sin(Date.now() * 0.005) * 0.5;
                            
                            if (threeJsScene.getObjectByName(`accident-warning-${accident.id}`)) {
                                requestAnimationFrame(pulseWarning);
                            }
                        }
                        pulseWarning();
                        
                        // Remove after duration or when cleared
                        setTimeout(() => {
                            threeJsScene.remove(accidentMesh);
                            threeJsScene.remove(warning);
                        }, accident.duration * 1000);
                    }
                });
            }
            
            // Apply weather effects
            if (data.weather && data.weather !== weatherEffect) {
                weatherEffect = data.weather;
                updateWeatherEffect(weatherEffect, data.time_of_day);
            }
        }

        // Apply weather effects to 3D scene
        function updateWeatherEffect(weather, timeOfDay) {
            // Clear existing effects
            const existingParticles = threeJsScene?.getObjectByName('weather-particles');
            if (existingParticles) {
                threeJsScene.remove(existingParticles);
            }
            
            if (!threeJsScene) return;
            
            // Change scene background color based on time of day
            if (timeOfDay === "DAY") {
                threeJsScene.background = new THREE.Color(0xf0f0f0);
            } else { // NIGHT
                threeJsScene.background = new THREE.Color(0x0a0a15);
            }
            
            // Add weather-specific effects
            if (weather === "RAIN") {
                // Create rain particles
                const rainCount = 15000;
                const rainGeometry = new THREE.BufferGeometry();
                const rainPositions = new Float32Array(rainCount * 3);
                
                for (let i = 0; i < rainCount * 3; i += 3) {
                    rainPositions[i] = Math.random() * 2000 - 1000;
                    rainPositions[i+1] = Math.random() * 1000;
                    rainPositions[i+2] = Math.random() * 2000 - 1000;
                }
                
                rainGeometry.setAttribute('position', new THREE.BufferAttribute(rainPositions, 3));
                
                const rainMaterial = new THREE.PointsMaterial({
                    color: 0x99ccff,
                    size: 1.5,
                    transparent: true,
                    opacity: 0.6
                });
                
                const rainParticles = new THREE.Points(rainGeometry, rainMaterial);
                rainParticles.name = 'weather-particles';
                threeJsScene.add(rainParticles);
                
                // Animate rain
                function animateRain() {
                    const positions = rainParticles.geometry.attributes.position.array;
                    
                    for (let i = 0; i < positions.length; i += 3) {
                        positions[i+1] -= 3; // Move downward
                        
                        // Reset to top when reaching bottom
                        if (positions[i+1] < 0) {
                            positions[i+1] = 1000;
                        }
                    }
                    
                    rainParticles.geometry.attributes.position.needsUpdate = true;
                    
                    if (threeJsScene.getObjectByName('weather-particles')) {
                        requestAnimationFrame(animateRain);
                    }
                }
                
                animateRain();
            } else if (weather === "SNOW") {
                // Create snow particles
                const snowCount = 10000;
                const snowGeometry = new THREE.BufferGeometry();
                const snowPositions = new Float32Array(snowCount * 3);
                
                for (let i = 0; i < snowCount * 3; i += 3) {
                    snowPositions[i] = Math.random() * 2000 - 1000;
                    snowPositions[i+1] = Math.random() * 1000;
                    snowPositions[i+2] = Math.random() * 2000 - 1000;
                }
                
                snowGeometry.setAttribute('position', new THREE.BufferAttribute(snowPositions, 3));
                
                const snowMaterial = new THREE.PointsMaterial({
                    color: 0xffffff,
                    size: 2,
                    transparent: true,
                    opacity: 0.8
                });
                
                const snowParticles = new THREE.Points(snowGeometry, snowMaterial);
                snowParticles.name = 'weather-particles';
                threeJsScene.add(snowParticles);
                
                // Animate snow with gentle swaying
                function animateSnow() {
                    const positions = snowParticles.geometry.attributes.position.array;
                    const time = Date.now() * 0.001;
                    
                    for (let i = 0; i < positions.length; i += 3) {
                        positions[i+1] -= 0.5; // Move downward slowly
                        
                        // Add gentle swaying effect
                        positions[i] += Math.sin(time + positions[i+2] * 0.1) * 0.1;
                        
                        // Reset to top when reaching bottom
                        if (positions[i+1] < 0) {
                            positions[i+1] = 1000;
                            positions[i] = Math.random() * 2000 - 1000;
                            positions[i+2] = Math.random() * 2000 - 1000;
                        }
                    }
                    
                    snowParticles.geometry.attributes.position.needsUpdate = true;
                    
                    if (threeJsScene.getObjectByName('weather-particles')) {
                        requestAnimationFrame(animateSnow);
                    }
                }
                
                animateSnow();
            } else if (weather === "FOG") {
                // Add fog to the scene
                threeJsScene.fog = new THREE.FogExp2(0xcccccc, 0.001);
            } else {
                // Clear fog for clear weather
                threeJsScene.fog = null;
            }
        }

        // Initialize heatmap
        function initializeHeatmap() {
            if (heatmapLayer) {
                map.removeLayer(heatmapLayer);
            }
            
            heatmapLayer = L.layerGroup();
        }

        // Update heatmap with data
        function updateHeatmap(data) {
            if (!showHeatmap || !data.heatmap) return;
            
            // Clear existing heatmap
            if (heatmapLayer) {
                heatmapLayer.clearLayers();
            } else {
                initializeHeatmap();
            }
            
            // Extract data
            const grid = data.heatmap.grid;
            const bounds = data.heatmap.bounds;
            
            // Convert heatmap data to image
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            const gridSize = grid.length;
            canvas.width = gridSize;
            canvas.height = gridSize;
            
            // Create image data
            const imageData = ctx.createImageData(gridSize, gridSize);
            
            // Fill image data
            for (let i = 0; i < gridSize; i++) {
                for (let j = 0; j < gridSize; j++) {
                    const value = grid[i][j];
                    const idx = (i * gridSize + j) * 4;
                    
                    // Convert value to color (red with transparency based on intensity)
                    if (value > 0) {
                        // Use a color gradient from green to red
                        const rgb = heatMapColorToRGB(value);
                        
                        imageData.data[idx] = rgb.r;
                        imageData.data[idx + 1] = rgb.g;
                        imageData.data[idx + 2] = rgb.b;
                        imageData.data[idx + 3] = Math.max(50, Math.min(255, value * 255 * 2));
                    } else {
                        // Transparent
                        imageData.data[idx] = 0;
                        imageData.data[idx + 1] = 0;
                        imageData.data[idx + 2] = 0;
                        imageData.data[idx + 3] = 0;
                    }
                }
            }
            
            ctx.putImageData(imageData, 0, 0);
            
            // Create image URL
            const dataURL = canvas.toDataURL();
            
            // Add image overlay to map
            const imageBounds = [
                [bounds[0], bounds[1]], // SW
                [bounds[2], bounds[3]]  // NE
            ];
            
            L.imageOverlay(dataURL, imageBounds, {
                opacity: 0.7
            }).addTo(heatmapLayer);
            
            // Add heatmap to map if not already added
            if (!map.hasLayer(heatmapLayer)) {
                heatmapLayer.addTo(map);
            }
        }

        // Convert heat value to RGB color (green to red)
        function heatMapColorToRGB(value) {
            // Value should be between 0 and 1
            const h = (1.0 - value) * 0.7; // Hue (from green to red)
            const s = 0.9; // Saturation
            const v = 0.9; // Value (brightness)
            
            // Convert HSV to RGB
            let r, g, b;
            
            const i = Math.floor(h * 6);
            const f = h * 6 - i;
            const p = v * (1 - s);
            const q = v * (1 - f * s);
            const t = v * (1 - (1 - f) * s);
            
            switch (i % 6) {
                case 0: r = v; g = t; b = p; break;
                case 1: r = q; g = v; b = p; break;
                case 2: r = p; g = v; b = t; break;
                case 3: r = p; g = q; b = v; break;
                case 4: r = t; g = p; b = v; break;
                case 5: r = v; g = p; b = q; break;
            }
            
            return {
                r: Math.round(r * 255),
                g: Math.round(g * 255),
                b: Math.round(b * 255)
            };
        }

        // Display vehicle inspector
        function showVehicleInspector(vehicleId, vehicles) {
            const vehicle = vehicles.find(v => v.id === vehicleId);
            
            if (!vehicle) {
                return;
            }
            
            inspectedVehicleId = vehicleId;
            
            // Display vehicle details
            document.getElementById('vehicle-details').innerHTML = `
                <h6>Vehicle #${vehicle.id} (${vehicle.type})</h6>
                <div class="table-responsive">
                    <table class="table table-sm">
                        <tr>
                            <th>Position</th>
                            <td>${vehicle.position[0].toFixed(6)}, ${vehicle.position[1].toFixed(6)}</td>
                        </tr>
                        <tr>
                            <th>Speed</th>
                            <td>${vehicle.speed.toFixed(1)} km/h</td>
                        </tr>
                        <tr>
                            <th>Direction</th>
                            <td>${vehicle.direction.toFixed(1)}°</td>
                        </tr>
                        <tr>
                            <th>Fuel/Battery</th>
                            <td>
                                ${vehicle.fuel_level ? 'Fuel: ' + vehicle.fuel_level.toFixed(1) + '%' : ''}
                                ${vehicle.battery_level > 0 ? 'Battery: ' + vehicle.battery_level.toFixed(1) + '%' : ''}
                            </td>
                        </tr>
                        <tr>
                            <th>Automation Level</th>
                            <td>Level ${vehicle.automation_level}</td>
                        </tr>
                        ${vehicle.platooning ? 
                            `<tr>
                                <th>Platooning</th>
                                <td>Active (Platoon #${vehicle.platoon_id})</td>
                            </tr>` : ''}
                    </table>
                </div>
            `;
            
            // Show the modal
            const inspectorModal = new bootstrap.Modal(document.getElementById('vehicle-inspector-modal'));
            inspectorModal.show();
            
            // Set follow button
            document.getElementById('follow-vehicle').onclick = function() {
                threeJsFollowVehicleId = vehicleId;
                threeJsFollowMode = true;
                inspectorModal.hide();
                
                // Show 3D view if not already visible
                if (!document.getElementById('3d-view-modal').classList.contains('show')) {
                    if (!threeJsScene) {
                        initialize3DView();
                    }
                    const modal = new bootstrap.Modal(document.getElementById('3d-view-modal'));
                    modal.show();
                    document.getElementById('toggle-3d-follow').textContent = "Stop Following";
                }
                
                showToast(`Now following vehicle #${vehicleId}`, 'info');
            };
        }

        // Handle custom vehicle addition
        function setupCustomVehicleAddition() {
            // When the add vehicle button is clicked
            document.getElementById('add-vehicle-btn').addEventListener('click', function() {
                addingCustomVehicle = true;
                
                // Default to center of the map
                const center = map.getCenter();
                document.getElementById('custom-vehicle-lat').value = center.lat.toFixed(6);
                document.getElementById('custom-vehicle-lng').value = center.lng.toFixed(6);
                
                // Show the modal
                const modal = new bootstrap.Modal(document.getElementById('add-vehicle-modal'));
                modal.show();
                
                // Alert user to click on map
                showToast('Click on the map to select a position for the new vehicle', 'info');
            });
            
            // Add vehicle button click
            document.getElementById('add-custom-vehicle').addEventListener('click', function() {
                const type = document.getElementById('custom-vehicle-type').value;
                const lat = parseFloat(document.getElementById('custom-vehicle-lat').value);
                const lng = parseFloat(document.getElementById('custom-vehicle-lng').value);
                const direction = parseInt(document.getElementById('custom-vehicle-direction').value);
                const speed = parseInt(document.getElementById('custom-vehicle-speed').value);
                
                // Send to server
                socket.emit('add_custom_vehicle', {
                    vehicle_type: type,
                    position: [lat, lng],
                    direction: direction,
                    speed: speed
                });
                
                // Reset and close modal
                addingCustomVehicle = false;
                customVehiclePosition = null;
                
                // Close the modal
                bootstrap.Modal.getInstance(document.getElementById('add-vehicle-modal')).hide();
            });
        }

        // Setup advanced settings
        function setupAdvancedSettings() {
            // Show advanced settings modal
            document.getElementById('show-advanced-settings').addEventListener('click', function() {
                const modal = new bootstrap.Modal(document.getElementById('advanced-settings-modal'));
                modal.show();
            });
            
            // Apply settings button
            document.getElementById('apply-settings').addEventListener('click', function() {
                // Collect settings
                const weather = document.getElementById('weather-select').value;
                const timeOfDay = document.getElementById('time-select').value;
                enableAccidents = document.getElementById('enable-accidents').checked;
                enableSecurityEvents = document.getElementById('enable-security').checked;
                enablePlatooning = document.getElementById('enable-platooning').checked;
                
                // Vehicle ratios
                vehicleRatios.CAR = parseInt(document.getElementById('car-ratio').value);
                vehicleRatios.TRUCK = parseInt(document.getElementById('truck-ratio').value);
                vehicleRatios.BUS = parseInt(document.getElementById('bus-ratio').value);
                vehicleRatios.EMERGENCY = parseInt(document.getElementById('emergency-ratio').value);
                
                // Network settings
                const protocol = document.getElementById('protocol-select').value;
                const latency = parseInt(document.getElementById('latency-slider').value);
                const bandwidth = parseInt(document.getElementById('bandwidth-slider').value);
                const packetLoss = parseFloat(document.getElementById('packet-loss-slider').value);
                
                // Send to server
                socket.emit('update_advanced_settings', {
                    weather: weather,
                    time_of_day: timeOfDay,
                    enable_accidents: enableAccidents,
                    enable_security: enableSecurityEvents,
                    enable_platooning: enablePlatooning,
                    vehicle_ratios: vehicleRatios,
                    network: {
                        protocol: protocol,
                        latency: latency,
                        bandwidth: bandwidth,
                        packet_loss: packetLoss
                    }
                });
                
                // Show confirmation
                showToast('Advanced settings applied', 'success');
                
                // Close modal
                bootstrap.Modal.getInstance(document.getElementById('advanced-settings-modal')).hide();
            });
            
            // Scenario selection
            document.querySelectorAll('#scenario-list a').forEach(item => {
                item.addEventListener('click', function(e) {
                    e.preventDefault();
                    const scenarioName = this.getAttribute('data-scenario');
                    
                    socket.emit('load_scenario', {
                        scenario_name: scenarioName
                    });
                    
                    // Close modal
                    bootstrap.Modal.getInstance(document.getElementById('advanced-settings-modal')).hide();
                });
            });
            
            // Update sliders
            document.getElementById('latency-slider').addEventListener('input', function() {
                document.getElementById('latency-value').textContent = this.value;
            });
            
            document.getElementById('bandwidth-slider').addEventListener('input', function() {
                document.getElementById('bandwidth-value').textContent = this.value;
            });
            
            document.getElementById('packet-loss-slider').addEventListener('input', function() {
                document.getElementById('packet-loss-value').textContent = this.value;
            });
            
            // Initialize checkboxes
            document.getElementById('enable-accidents').checked = enableAccidents;
            document.getElementById('enable-security').checked = enableSecurityEvents;
            document.getElementById('enable-platooning').checked = enablePlatooning;
        }

        // Update map with simulation data
        function updateSimulation(data) {
            // Update vehicles on the map
            updateVehicles(data.vehicles);
            
            // Update infrastructure on the map
            updateInfrastructure(data);
            
            // Visualize messages
            visualizeMessages(data.messages);
            
            // Update statistics display
            updateStatistics(data.stats);
            
            // Update charts
            updateCharts(data.stats);
            
            // Update message log
            updateMessageLog(data.messages);
            
            // Update road network if it changed
            if (data.network && (!roadNetwork || roadNetwork.nodes.length !== data.network.nodes.length)) {
                roadNetwork = data.network;
                drawRoadNetwork();
            }
            
            // Update 3D view if initialized
            if (threeJsScene) {
                update3DScene(data);
            }
            
            // Update heatmap
            updateHeatmap(data);
            
            // Update environment visualizations
            updateEnvironmentVisuals(data);
            
            // Handle event notifications
            setupEventNotifications(data);
            
            // Update advanced monitoring
            updateAdvancedMonitoring(data);
        }

        // Update environment visuals
        function updateEnvironmentVisuals(data) {
            if (data.weather) {
                // Update weather badge
                const weatherBadge = document.getElementById('weather-badge');
                weatherBadge.className = 'weather-badge ' + data.weather;
                weatherBadge.textContent = data.weather.charAt(0) + data.weather.slice(1).toLowerCase();
                
                // Update weather status
                document.getElementById('weather-status').textContent = data.weather.charAt(0) + data.weather.slice(1).toLowerCase();
                
                // Update weather overlay
                const weatherOverlay = document.getElementById('weather-overlay');
                weatherOverlay.className = data.weather;
            }
            
            if (data.time_of_day) {
                // Update time badge
                const timeBadge = document.getElementById('time-badge');
                timeBadge.className = 'time-badge ' + data.time_of_day;
                timeBadge.textContent = data.time_of_day.charAt(0) + data.time_of_day.slice(1).toLowerCase();
                
                // Update time status
                document.getElementById('time-of-day-status').textContent = data.time_of_day.charAt(0) + data.time_of_day.slice(1).toLowerCase();
                
                // Add night effect
                if (data.time_of_day === "NIGHT") {
                    if (!document.getElementById('weather-overlay').classList.contains('NIGHT')) {
                        document.getElementById('weather-overlay').classList.add('NIGHT');
                    }
                } else {
                    document.getElementById('weather-overlay').classList.remove('NIGHT');
                }
            }
        }

        // Setup event notifications
        function setupEventNotifications(data) {
            // Check for new accidents
            if (data.accidents && data.accidents.length > 0) {
                // Get non-cleared accidents
                const activeAccidents = data.accidents.filter(a => !a.cleared);
                
                if (activeAccidents.length > 0) {
                    // Update the accident counter
                    document.getElementById('accident-count').textContent = activeAccidents.length;
                    
                    // Take the most recent/severe accident
                    const accident = activeAccidents.sort((a, b) => b.severity - a.severity)[0];
                    
                    // Show toast notification if it's new
                    if (!document.getElementById('accident-toast').classList.contains('show')) {
                        document.getElementById('accident-details').textContent = 
                            `A severity ${accident.severity} accident involving ${accident.vehicles_involved.length} vehicles has occurred.`;
                        
                        const accidentToast = new bootstrap.Toast(document.getElementById('accident-toast'));
                        accidentToast.show();
                        
                        // Set up view accident button
                        document.getElementById('view-accident').onclick = function() {
                            map.setView([accident.position[0], accident.position[1]], 18);
                            accidentToast.hide();
                        };
                    }
                } else {
                    document.getElementById('accident-count').textContent = "0";
                }
            }
            
            // Check for new security events
            if (data.security_events && data.security_events.length > 0) {
                // Get active security events
                const activeEvents = data.security_events.filter(e => !e.resolved);
                
                // Update the security counter
                document.getElementById('security-count').textContent = activeEvents.length;
                
                if (activeEvents.length > 0) {
                    // Take the most severe event
                    const event = activeEvents.sort((a, b) => b.severity - a.severity)[0];
                    
                    // Show toast notification if there's a new event
                    if (!document.getElementById('security-toast').classList.contains('show')) {
                        document.getElementById('security-details').textContent = 
                            `${event.type}: ${event.description} (Severity: ${event.severity})`;
                        
                        const securityToast = new bootstrap.Toast(document.getElementById('security-toast'));
                        securityToast.show();
                    }
                } else {
                    document.getElementById('security-count').textContent = "0";
                }
            }
        }

        // Update advanced monitoring
        function updateAdvancedMonitoring(data) {
            // Update platoon count
            if (data.platoons) {
                document.getElementById('platoon-count').textContent = Object.keys(data.platoons).length;
            }
            
            // Update network status
            if (data.config) {
                document.getElementById('protocol-status').textContent = data.config.communication_protocol || "DSRC";
                document.getElementById('latency-status').textContent = (data.config.network_latency || "20") + "ms";
            }
            
            // Update traffic prediction
            if (data.traffic_prediction && data.traffic_prediction.length > 0) {
                const predictions = data.traffic_prediction;
                const container = document.getElementById('prediction-container');
                
                let html = '<div class="d-flex justify-content-between mb-2">';
                html += '<span>Current:</span>';
                html += `<span class="fw-bold">${data.stats.vehicle_count}</span>`;
                html += '</div>';
                
                html += '<div class="d-flex justify-content-between">';
                html += '<span>Predicted (5 steps):</span>';
                html += `<span class="fw-bold">${Math.round(predictions[predictions.length-1])}</span>`;
                html += '</div>';
                
                // Add a mini chart
                html += '<div class="mt-2" style="height: 40px;">';
                html += `<div class="d-flex align-items-end h-100 justify-content-between">`;
                
                // Current value bar
                const currentHeight = Math.max(10, Math.min(40, data.stats.vehicle_count / 2));
                html += `<div title="Current" style="width: 15%; height: ${currentHeight}px; background-color: #4285F4;"></div>`;
                
                // Prediction bars
                predictions.forEach(val => {
                    const height = Math.max(10, Math.min(40, val / 2));
                    const color = val > data.stats.vehicle_count ? '#34a853' : '#ea4335';
                    html += `<div title="Predicted: ${Math.round(val)}" style="width: 15%; height: ${height}px; background-color: ${color};"></div>`;
                });
                
                html += '</div>';
                html += '</div>';
                
                container.innerHTML = html;
            }
        }

        // Update vehicle markers
        function updateVehicles(vehicles) {
            // Update existing vehicle markers and add new ones
            vehicles.forEach(vehicle => {
                if (vehicleMarkers[vehicle.id]) {
                    // Update existing marker position
                    vehicleMarkers[vehicle.id].setLatLng([vehicle.position[0], vehicle.position[1]]);
                    
                    // Update rotation based on direction
                    if (vehicleMarkers[vehicle.id]._icon) {
                        vehicleMarkers[vehicle.id]._icon.style.transform = 
                            `${vehicleMarkers[vehicle.id]._icon.style.transform.replace(/rotate\\([^)]*\\)/, '')} rotate(${vehicle.direction}deg)`;
                        
                        // Update platooning indicator
                        const platoonIndicator = vehicleMarkers[vehicle.id]._icon.querySelector('.platoon-indicator');
                        if (vehicle.platooning) {
                            if (!platoonIndicator) {
                                const indicator = document.createElement('div');
                                indicator.className = 'platoon-indicator';
                                vehicleMarkers[vehicle.id]._icon.appendChild(indicator);
                            }
                        } else if (platoonIndicator) {
                            platoonIndicator.remove();
                        }
                    }
                    
                    // Update tooltip content
                    vehicleMarkers[vehicle.id].setTooltipContent(
                        `<div>
                            <strong>${vehicle.type} #${vehicle.id}</strong><br>
                            Speed: ${vehicle.speed.toFixed(1)} km/h<br>
                            Direction: ${vehicle.direction.toFixed(0)}°
                            ${vehicle.platooning ? '<br><span class="text-success">Platooning</span>' : ''}
                            ${vehicle.automation_level >= 3 ? '<br>Automation: Level ' + vehicle.automation_level : ''}
                        </div>`
                    );
                    
                } else {
                    // Create new marker
                    let html = `<div class="vehicle-marker ${vehicle.type}"></div>`;
                    
                    // Add platooning indicator if needed
                    if (vehicle.platooning) {
                        html = `<div class="vehicle-marker ${vehicle.type}"><div class="platoon-indicator"></div></div>`;
                    }
                    
                    const icon = L.divIcon({
                        className: '',
                        html: html,
                        iconSize: [12, 12]
                    });
                    
                    vehicleMarkers[vehicle.id] = L.marker([vehicle.position[0], vehicle.position[1]], {
                        icon: icon
                    }).addTo(map);
                    
                    // Add tooltip
                    vehicleMarkers[vehicle.id].bindTooltip(
                        `<div>
                            <strong>${vehicle.type} #${vehicle.id}</strong><br>
                            Speed: ${vehicle.speed.toFixed(1)} km/h<br>
                            Direction: ${vehicle.direction.toFixed(0)}°
                            ${vehicle.platooning ? '<br><span class="text-success">Platooning</span>' : ''}
                            ${vehicle.automation_level >= 3 ? '<br>Automation: Level ' + vehicle.automation_level : ''}
                        </div>`,
                        {offset: [0, -5]}
                    );
                }
            });
            
            // Remove vehicles that no longer exist
            Object.keys(vehicleMarkers).forEach(id => {
                if (!vehicles.find(v => v.id === parseInt(id))) {
                    map.removeLayer(vehicleMarkers[id]);
                    delete vehicleMarkers[id];
                }
            });
        }
        
        // Update infrastructure markers
        function updateInfrastructure(data) {
            // Update existing infrastructure markers and add new ones
            data.infrastructure.forEach(infra => {
                if (infraMarkers[infra.id]) {
                    // Just update tooltip content (position doesn't change)
                    infraMarkers[infra.id].setTooltipContent(
                        `<div>
                            <strong>${infra.type} #${infra.id}</strong><br>
                            State: ${infra.state}
                        </div>`
                    );
                    
                    // Update color for traffic lights
                    if (infra.type === "TRAFFIC_LIGHT") {
                        const element = infraMarkers[infra.id].getElement();
                        if (element && element.firstChild) {
                            let color = "#9c27b0";
                            if (infra.state === "RED") color = "#dc3545";
                            else if (infra.state === "YELLOW") color = "#ffc107";
                            else if (infra.state === "GREEN") color = "#28a745";
                            element.firstChild.style.backgroundColor = color;
                        }
                    }
                    
                } else {
                    // Create new marker
                    let className = 'infra-marker ' + infra.type;
                    let color = "#00bcd4";
                    
                    if (infra.type === "TRAFFIC_LIGHT") {
                        if (infra.state === "RED") color = "#dc3545";
                        else if (infra.state === "YELLOW") color = "#ffc107";
                        else if (infra.state === "GREEN") color = "#28a745";
                        else color = "#9c27b0";
                    }
                    
                    const icon = L.divIcon({
                        className: '',
                        html: `<div class="${className}" style="background-color: ${color};"></div>`,
                        iconSize: [14, 14]
                    });
                    
                    infraMarkers[infra.id] = L.marker([infra.position[0], infra.position[1]], {
                        icon: icon
                    }).addTo(map);
                    
                    // Add tooltip
                    infraMarkers[infra.id].bindTooltip(
                        `<div>
                            <strong>${infra.type} #${infra.id}</strong><br>
                            State: ${infra.state}
                        </div>`,
                        {offset: [0, -5]}
                    );
                }
            });
            
            // Remove infrastructure that no longer exists
            Object.keys(infraMarkers).forEach(id => {
                // Skip accident markers which have a prefix
                if (id.startsWith('accident_') || id.startsWith('accident_circle_')) {
                    return;
                }
                
                if (!data.infrastructure.find(i => i.id === parseInt(id))) {
                    map.removeLayer(infraMarkers[id]);
                    delete infraMarkers[id];
                }
            });
            
            // Add accident markers
            if (data.accidents) {
                data.accidents.forEach(accident => {
                    const accidentId = `accident_${accident.id}`;
                    
                    if (!accident.cleared) {
                        if (!infraMarkers[accidentId]) {
                            // Create accident marker
                            const icon = L.divIcon({
                                className: '',
                                html: `<div class="accident-icon">⚠️</div>`,
                                iconSize: [24, 24]
                            });
                            
                            infraMarkers[accidentId] = L.marker([accident.position[0], accident.position[1]], {
                                icon: icon
                            }).addTo(map);
                            
                            // Add tooltip
                            infraMarkers[accidentId].bindTooltip(
                                `<div>
                                    <strong>Traffic Accident</strong><br>
                                    Severity: ${accident.severity}/5<br>
                                    Vehicles: ${accident.vehicles_involved.length}
                                </div>`,
                                {offset: [0, -5]}
                            );
                            
                            // Add pulsing circle
                            const radius = accident.severity * 20; // Size based on severity
                            const circle = L.circle([accident.position[0], accident.position[1]], {
                                radius: radius,
                                color: '#dc3545',
                                weight: 2,
                                opacity: 0.7,
                                fillColor: '#dc3545',
                                fillOpacity: 0.2
                            }).addTo(map);
                            
                            infraMarkers[`${accidentId}_circle`] = circle;
                            
                            // Animate the circle
                            let direction = 1;
                            let factor = 0.5;
                            
                            function animateCircle() {
                                if (!infraMarkers[`${accidentId}_circle`]) return;
                                
                                factor += 0.05 * direction;
                                if (factor >= 1) direction = -1;
                                if (factor <= 0.5) direction = 1;
                                
                                circle.setRadius(radius * factor);
                                
                                requestAnimationFrame(animateCircle);
                            }
                            
                            animateCircle();
                        }
                    } else {
                        // Remove cleared accident marker
                        if (infraMarkers[accidentId]) {
                            map.removeLayer(infraMarkers[accidentId]);
                            delete infraMarkers[accidentId];
                        }
                        
                        if (infraMarkers[`${accidentId}_circle`]) {
                            map.removeLayer(infraMarkers[`${accidentId}_circle`]);
                            delete infraMarkers[`${accidentId}_circle`];
                        }
                    }
                });
            }
        }
        
        // Visualize message exchanges
        function visualizeMessages(messages) {
            // Remove old message visualizations
            messageLines.forEach(line => {
                if (line && map.hasLayer(line)) {
                    map.removeLayer(line);
                }
            });
            messageLines = [];
            
            // Only proceed if message visualization is enabled
            if (!showMessages) {
                return;
            }
            
            // Add new message visualizations
            messages.forEach(message => {
                // Define line color based on message type
                let color = "#6c757d";  // Default gray
                
                switch (message.type) {
                    case "SAFETY_WARNING":
                        color = "#dc3545";  // Red
                        break;
                    case "TRAFFIC_INFO":
                        color = "#17a2b8";  // Cyan
                        break;
                    case "V2V_POSITION":
                        color = "#4285F4";  // Blue
                        break;
                    case "I2V_SIGNAL":
                        color = "#9c27b0";  // Purple
                        break;
                    case "V2I_REQUEST":
                        color = "#34a853";  // Green
                        break;
                }
                
                // Create a polyline for the message path
                const line = L.polyline([
                    [message.source_position[0], message.source_position[1]],
                    [message.destination_position[0], message.destination_position[1]]
                ], {
                    color: color,
                    weight: 2,
                    opacity: 0.7,
                    className: 'message-path'
                }).addTo(map);
                
                // Add to message lines array for later removal
                messageLines.push(line);
                
                // Add tooltip with message details
                line.bindTooltip(
                    `<div>
                        <strong>${message.type}</strong><br>
                        From: ${message.source_type} #${message.source_id}<br>
                        To: ${message.destination_type} #${message.destination_id}<br>
                        Content: ${message.content}
                    </div>`
                );
                
                // Remove the line after 2 seconds
                setTimeout(() => {
                    if (map.hasLayer(line)) {
                        map.removeLayer(line);
                    }
                }, 2000);
            });
        }
        
        // Update message visibility
        function updateMessageVisibility() {
            if (!showMessages) {
                // Remove all message lines
                messageLines.forEach(line => {
                    if (line && map.hasLayer(line)) {
                        map.removeLayer(line);
                    }
                });
                messageLines = [];
            }
        }
        
        // Draw road network
        function drawRoadNetwork() {
            // Clear current network layer
            networkLayerGroup.clearLayers();
            
            if (!roadNetwork || !showNetworkLayer) {
                return;
            }
            
            // Draw road edges
            roadNetwork.edges.forEach(edge => {
                const startNode = roadNetwork.nodes[edge.start_node];
                const endNode = roadNetwork.nodes[edge.end_node];
                
                // Create a polyline for the road
                const road = L.polyline([
                    [startNode.position[0], startNode.position[1]],
                    [endNode.position[0], endNode.position[1]]
                ], {
                    color: '#666',
                    weight: 3,
                    opacity: 0.5
                }).addTo(networkLayerGroup);
            });
            
            // Draw nodes (intersections)
            roadNetwork.nodes.forEach(node => {
                // Create circle marker for intersection
                const intersection = L.circleMarker([node.position[0], node.position[1]], {
                    color: '#333',
                    fillColor: '#666',
                    fillOpacity: 0.5,
                    radius: 3
                }).addTo(networkLayerGroup);
            });
        }
        
        // Update statistics display
        function updateStatistics(stats) {
            document.getElementById('vehicle-count').textContent = stats.vehicle_count;
            document.getElementById('message-count').textContent = stats.messages_sent;
            document.getElementById('avg-speed').textContent = stats.average_speed.toFixed(1);
            document.getElementById('message-success-rate').textContent = 
                (stats.message_success_rate * 100).toFixed(1) + '%';
        }
        
        // Update charts with new data
        function updateCharts(stats) {
            // Only update charts if time series data is available
            if (!stats.time_series || 
                !stats.time_series.vehicle_count || 
                stats.time_series.vehicle_count.length === 0) {
                return;
            }
            
            // Create labels (time points)
            const labels = Array.from({ length: stats.time_series.vehicle_count.length }, (_, i) => i * 5);
            
            // Update Traffic Chart
            charts.traffic.data.labels = labels;
            charts.traffic.data.datasets[0].data = stats.time_series.vehicle_count;
            charts.traffic.update();
            
            // Update Message Chart
            charts.messages.data.labels = labels;
            charts.messages.data.datasets[0].data = stats.time_series.message_count;
            charts.messages.update();
            
            // Update Speed Chart
            charts.speed.data.labels = labels;
            charts.speed.data.datasets[0].data = stats.time_series.average_speed;
            charts.speed.update();
            
            // Update Message Types Chart
            if (stats.message_types) {
                const messageTypeLabels = Object.keys(stats.message_types);
                const messageTypeCounts = messageTypeLabels.map(type => stats.message_types[type]);
                
                charts.messageTypes.data.labels = messageTypeLabels;
                charts.messageTypes.data.datasets[0].data = messageTypeCounts;
                charts.messageTypes.update();
            }
            
            // Update advanced charts
            
            // Network Performance
            if (charts.networkPerformance) {
                charts.networkPerformance.data.labels = labels;
                
                // Success rate data
                let successRates = [];
                for (let i = 0; i < labels.length; i++) {
                    successRates.push(Math.min(1, (stats.message_success_rate * 0.9) + (Math.random() * 0.2))); 
                }
                charts.networkPerformance.data.datasets[0].data = successRates;
                
                // Latency data 
                let latencies = [];
                for (let i = 0; i < labels.length; i++) {
                    latencies.push(20 + Math.random() * 10);
                }
                charts.networkPerformance.data.datasets[1].data = latencies;
                
                charts.networkPerformance.update();
            }
            
            // Vehicle Distribution
            if (charts.vehicleDistribution) {
                // Count vehicles by type
                let vehicleCounts = {
                    'Cars': 0,
                    'Trucks': 0,
                    'Buses': 0,
                    'Emergency': 0
                };
                
                // Update from most recent data if available 
                if (vehicleRatios) {
                    const total = vehicleRatios.CAR + vehicleRatios.TRUCK + vehicleRatios.BUS + vehicleRatios.EMERGENCY;
                    if (total > 0) {
                        vehicleCounts = {
                            'Cars': vehicleRatios.CAR,
                            'Trucks': vehicleRatios.TRUCK,
                            'Buses': vehicleRatios.BUS,
                            'Emergency': vehicleRatios.EMERGENCY
                        };
                    }
                }
                
                charts.vehicleDistribution.data.datasets[0].data = [
                    vehicleCounts['Cars'],
                    vehicleCounts['Trucks'],
                    vehicleCounts['Buses'],
                    vehicleCounts['Emergency']
                ];
                
                charts.vehicleDistribution.update();
            }
            
            // Security Events Chart
            if (charts.securityEvents && stats.security_events) {
                let securityCounts = {
                    'Jamming': 0,
                    'Spoofing': 0,
                    'Eavesdropping': 0,
                    'MITM': 0
                };
                
                // Count security events by type
                if (stats.security_events && Array.isArray(stats.security_events)) {
                    stats.security_events.forEach(event => {
                        if (securityCounts[event.type]) {
                            securityCounts[event.type]++;
                        }
                    });
                }
                
                charts.securityEvents.data.datasets[0].data = [
                    securityCounts['Jamming'],
                    securityCounts['Spoofing'],
                    securityCounts['Eavesdropping'],
                    securityCounts['MITM']
                ];
                
                charts.securityEvents.update();
            }
        }
        
        // Update message log
        function updateMessageLog(messages) {
            const logContainer = document.getElementById('message-log');
            
            // Only show a few latest messages to avoid overwhelming the log
            const messagesToShow = messages.slice(0, 3);
            
            messagesToShow.forEach(message => {
                const logEntry = document.createElement('div');
                logEntry.className = `message-log ${message.type}`;
                logEntry.innerHTML = `
                    <strong>${message.type}</strong>: 
                    ${message.source_type} #${message.source_id} → 
                    ${message.destination_type} #${message.destination_id} | 
                    ${message.content}
                `;
                
                // Add to the top of the log
                logContainer.insertBefore(logEntry, logContainer.firstChild);
                
                // Keep log size manageable (remove old entries)
                if (logContainer.children.length > 50) {
                    logContainer.removeChild(logContainer.lastChild);
                }
            });
        }
        
        // Reset visualization
        function resetVisualization() {
            // Clear vehicle markers
            Object.values(vehicleMarkers).forEach(marker => map.removeLayer(marker));
            vehicleMarkers = {};
            
            // Clear infrastructure markers
            Object.values(infraMarkers).forEach(marker => map.removeLayer(marker));
            infraMarkers = {};
            
            // Clear message lines
            messageLines.forEach(line => {
                if (line && map.hasLayer(line)) {
                    map.removeLayer(line);
                }
            });
            messageLines = [];
            
            // Clear road network
            networkLayerGroup.clearLayers();
            
            // Reset charts
            charts.traffic.data.labels = [];
            charts.traffic.data.datasets[0].data = [];
            charts.traffic.update();
            
            charts.messages.data.labels = [];
            charts.messages.data.datasets[0].data = [];
            charts.messages.update();
            
            charts.speed.data.labels = [];
            charts.speed.data.datasets[0].data = [];
            charts.speed.update();
            
            charts.messageTypes.data.labels = [];
            charts.messageTypes.data.datasets[0].data = [];
            charts.messageTypes.update();
            
            // Reset advanced charts
            if (charts.networkPerformance) {
                charts.networkPerformance.data.labels = [];
                charts.networkPerformance.data.datasets.forEach(dataset => dataset.data = []);
                charts.networkPerformance.update();
            }
            
            if (charts.securityEvents) {
                charts.securityEvents.data.datasets[0].data = [0, 0, 0, 0];
                charts.securityEvents.update();
            }
            
            // Clear message log
            document.getElementById('message-log').innerHTML = '';
            
            // Reset statistics
            document.getElementById('vehicle-count').textContent = '0';
            document.getElementById('message-count').textContent = '0';
            document.getElementById('avg-speed').textContent = '0';
            document.getElementById('message-success-rate').textContent = '0%';
            document.getElementById('platoon-count').textContent = '0';
            document.getElementById('accident-count').textContent = '0';
            document.getElementById('security-count').textContent = '0';
            document.getElementById('prediction-container').innerHTML = '<small class="text-muted">Not enough data yet</small>';
            
            // Reset 3D view if initialized
            if (threeJsScene) {
                // Remove all vehicles
                Object.keys(threeJsVehicles).forEach(id => {
                    threeJsScene.remove(threeJsVehicles[id]);
                });
                threeJsVehicles = {};
                
                // Remove all infrastructure
                Object.keys(threeJsInfrastructure).forEach(id => {
                    threeJsScene.remove(threeJsInfrastructure[id]);
                });
                threeJsInfrastructure = {};
                
                // Remove all messages
                threeJsMessages.forEach(msg => {
                    threeJsScene.remove(msg);
                });
                threeJsMessages = [];
                
                // Reset camera
                threeJsCamera.position.set(0, 500, 500);
                threeJsCamera.lookAt(0, 0, 0);
                if (threeJsControls) threeJsControls.reset();
                threeJsFollowMode = false;
                threeJsFollowVehicleId = null;
                
                // Reset weather effects
                const existingParticles = threeJsScene.getObjectByName('weather-particles');
                if (existingParticles) {
                    threeJsScene.remove(existingParticles);
                }
                threeJsScene.fog = null;
                threeJsScene.background = new THREE.Color(0xf0f0f0);
            }
            
            // Reset heatmap
            if (heatmapLayer) {
                heatmapLayer.clearLayers();
                if (map.hasLayer(heatmapLayer)) {
                    map.removeLayer(heatmapLayer);
                }
            }
            
            // Reset weather and time
            document.getElementById('weather-badge').className = 'weather-badge CLEAR';
            document.getElementById('weather-badge').textContent = 'Clear';
            document.getElementById('time-badge').className = 'time-badge DAY';
            document.getElementById('time-badge').textContent = 'Day';
            document.getElementById('weather-overlay').className = '';
            document.getElementById('weather-status').textContent = 'Clear';
            document.getElementById('time-of-day-status').textContent = 'Day';
        }
        
        // Generic toast message
        function showToast(message, type = 'info') {
            const toastContainer = document.querySelector('.toast-container');
            
            // Create toast element
            const toastEl = document.createElement('div');
            toastEl.className = 'toast';
            toastEl.setAttribute('role', 'alert');
            toastEl.setAttribute('aria-live', 'assertive');
            toastEl.setAttribute('aria-atomic', 'true');
            
            // Determine header style based on type
            let headerClass = 'bg-info text-white';
            let headerTitle = 'Information';
            
            if (type === 'success') {
                headerClass = 'bg-success text-white';
                headerTitle = 'Success';
            } else if (type === 'error') {
                headerClass = 'bg-danger text-white';
                headerTitle = 'Error';
            } else if (type === 'warning') {
                headerClass = 'bg-warning text-dark';
                headerTitle = 'Warning';
            }
            
            // Add toast content
            toastEl.innerHTML = `
                <div class="toast-header ${headerClass}">
                    <strong class="me-auto">${headerTitle}</strong>
                    <small>just now</small>
                    <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            `;
            
            // Add to container
            toastContainer.appendChild(toastEl);
            
            // Create and show toast
            const toast = new bootstrap.Toast(toastEl);
            toast.show();
            
            // Remove after hiding
            toastEl.addEventListener('hidden.bs.toast', () => {
                toastEl.remove();
            });
        }
    </script>
</body>
</html>
"""


# Server-side routes
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/simulation/config", methods=["GET", "POST"])
def simulation_config():
    if request.method == "POST":
        config = request.json
        simulator.update_config(config)
        return jsonify({"status": "success"})
    return jsonify(simulator.get_config())


# Socket.IO event handlers
@socketio.on("start_simulation")
def handle_start_simulation(data):
    simulator.start(data)


@socketio.on("pause_simulation")
def handle_pause_simulation():
    simulator.pause()


@socketio.on("reset_simulation")
def handle_reset_simulation():
    simulator.reset()


@socketio.on("update_parameter")
def handle_update_parameter(data):
    simulator.update_config({data["parameter"]: data["value"]})


@socketio.on("get_vehicle_details")
def handle_get_vehicle_details(data):
    vehicle_id = data.get("vehicle_id")
    if vehicle_id is not None:
        # Find the vehicle
        vehicle = next((v for v in simulator.vehicles if v.id == vehicle_id), None)
        if vehicle:
            socketio.emit(
                "vehicle_details",
                {"vehicle_id": vehicle_id, "vehicle": vehicle.to_dict()},
            )


@socketio.on("add_custom_vehicle")
def handle_add_custom_vehicle(data):
    vehicle_type = data.get("vehicle_type")
    position = data.get("position")
    direction = data.get("direction")
    speed = data.get("speed")

    vehicle_id = simulator.add_custom_vehicle(vehicle_type, position, direction, speed)

    socketio.emit("custom_vehicle_added", {"vehicle_id": vehicle_id})


@socketio.on("update_advanced_settings")
def handle_update_advanced_settings(data):
    # Update weather and time of day
    weather = data.get("weather")
    if weather in [w.name for w in WeatherCondition]:
        simulator.weather_condition = WeatherCondition[weather]

    time_of_day = data.get("time_of_day")
    if time_of_day in ["DAY", "NIGHT"]:
        simulator.time_of_day = time_of_day

    # Update feature flags
    if "enable_accidents" in data:
        simulator.enable_accidents = data["enable_accidents"]

    if "enable_security" in data:
        simulator.enable_security_events = data["enable_security"]

    if "enable_platooning" in data:
        simulator.enable_platooning = data["enable_platooning"]

    # Update vehicle ratios if provided
    if "vehicle_ratios" in data:
        ratios = data["vehicle_ratios"]
        # Store for next vehicle generation
        simulator.vehicle_type_ratios = ratios

    # Update network parameters if provided
    if "network" in data:
        network = data["network"]
        if "protocol" in network:
            simulator.config["communication_protocol"] = network["protocol"]
        if "latency" in network:
            simulator.config["network_latency"] = network["latency"]
        if "bandwidth" in network:
            simulator.config["network_bandwidth"] = network["bandwidth"]
        if "packet_loss" in network:
            simulator.config["packet_loss"] = network["packet_loss"]


@socketio.on("load_scenario")
def handle_load_scenario(data):
    scenario_name = data.get("scenario_name")
    if scenario_name:
        success = simulator.load_scenario(scenario_name)
        if success:
            socketio.emit("scenario_loaded", {"name": scenario_name, "success": True})


def background_thread():
    while True:
        if simulator.is_running():
            simulation_data = simulator.step()
            socketio.emit("simulation_update", simulation_data)
        socketio.sleep(0.1)  # Update every 100ms


@socketio.on("connect")
def handle_connect():
    socketio.start_background_task(background_thread)


if __name__ == "__main__":
    # Start the Flask application with Socket.IO
    print("Starting Enhanced V2X Simulation Server...")
    print("Open a browser and navigate to http://127.0.0.1:5000")
    socketio.run(app, debug=True)
