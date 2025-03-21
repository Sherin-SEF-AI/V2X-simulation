# v2x_simulation_app.py
from flask import Flask, render_template_string, request, jsonify
from flask_socketio import SocketIO, emit
import time
import random
import math
import numpy as np
from enum import Enum
import threading
import json

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

        # Random speed variations
        self.speed += random.uniform(-2, 2)
        self.speed = max(5, min(120, self.speed))  # Speed limits

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type.name,
            "position": self.position,
            "direction": self.direction,
            "speed": self.speed,
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
        self.config = {
            "vehicle_density": 0.1,  # Vehicles per node
            "communication_range": 300,  # meters
            "communication_reliability": 0.95,  # Probability of successful transmission
            "traffic_light_cycle": 30,  # seconds
            "simulation_speed": 1.0,  # Real-time multiplier
            "map_center": (37.7749, -122.4194),  # San Francisco center
        }
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

            # Random vehicle type with distribution
            type_choices = [
                VehicleType.CAR,
                VehicleType.TRUCK,
                VehicleType.BUS,
                VehicleType.EMERGENCY,
            ]
            weights = [
                0.7,
                0.15,
                0.1,
                0.05,
            ]  # 70% cars, 15% trucks, 10% buses, 5% emergency
            veh_type = random.choices(type_choices, weights=weights)[0]

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

        # Update statistics
        self._update_statistics()

        # Return simulation state for visualization
        return self._get_simulation_state()

    def _simulate_communications(self):
        # Clear old messages that have been displayed
        self.active_messages = [
            m for m in self.active_messages if time.time() - m.timestamp < 2
        ]

        # V2V communications
        for i, v1 in enumerate(self.vehicles):
            for v2 in self.vehicles[i + 1 :]:
                # Calculate distance between vehicles
                distance = self._calculate_distance(v1.position, v2.position)

                # Check if within communication range
                if distance <= self.config["communication_range"]:
                    # Generate V2V messages with probability based on reliability
                    if random.random() < self.config["communication_reliability"]:
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
                    if random.random() < self.config["communication_reliability"]:
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
                    if random.random() < self.config["communication_reliability"]:
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
        return {
            "time": self.time,
            "vehicles": [v.to_dict() for v in self.vehicles],
            "infrastructure": [i.to_dict() for i in self.infrastructure],
            "messages": [m.to_dict() for m in self.active_messages],
            "stats": self.stats,
            "network": self.road_network.to_dict(),
        }

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
    <title>V2X Simulation Platform</title>
    
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
                <a class="navbar-brand" href="#">V2X Simulation Platform</a>
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
                        </div>
                    </div>
                    <div class="card-body p-0">
                        <div id="map-container"></div>
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
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Analytics dashboard -->
        <div class="row mt-4" id="analytics">
            <div class="col-md-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Analytics Dashboard</h5>
                    </div>
                    <div class="card-body">
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
            <p class="small">V2X Simulation Platform &copy; 2025</p>
        </footer>
    </div>
    
    <!-- JavaScript libraries -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    
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
        
        // Initialize application
        document.addEventListener('DOMContentLoaded', function() {
            // Initialize map
            initializeMap();
            
            // Initialize charts
            initializeCharts();
            
            // Initialize controls
            initializeControls();
            
            // Socket event handlers
            socket.on('connect', function() {
                console.log('Connected to simulation server');
            });
            
            socket.on('simulation_update', function(data) {
                updateSimulation(data);
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
        }
        
        // Update map with simulation data
        function updateSimulation(data) {
            // Update vehicles on the map
            updateVehicles(data.vehicles);
            
            // Update infrastructure on the map
            updateInfrastructure(data.infrastructure);
            
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
                    }
                    
                    // Update tooltip content
                    vehicleMarkers[vehicle.id].setTooltipContent(
                        `<div>
                            <strong>${vehicle.type} #${vehicle.id}</strong><br>
                            Speed: ${vehicle.speed.toFixed(1)} km/h<br>
                            Direction: ${vehicle.direction.toFixed(0)}°
                        </div>`
                    );
                    
                } else {
                    // Create new marker
                    const icon = L.divIcon({
                        className: '',
                        html: `<div class="vehicle-marker ${vehicle.type}"></div>`,
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
        function updateInfrastructure(infrastructure) {
            // Update existing infrastructure markers and add new ones
            infrastructure.forEach(infra => {
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
                if (!infrastructure.find(i => i.id === parseInt(id))) {
                    map.removeLayer(infraMarkers[id]);
                    delete infraMarkers[id];
                }
            });
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
            
            // Clear message log
            document.getElementById('message-log').innerHTML = '';
            
            // Reset statistics
            document.getElementById('vehicle-count').textContent = '0';
            document.getElementById('message-count').textContent = '0';
            document.getElementById('avg-speed').textContent = '0';
            document.getElementById('message-success-rate').textContent = '0%';
        }
    </script>
</body>
</html>
"""


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
    print("Starting V2X Simulation Server...")
    print("Open a browser and navigate to http://127.0.0.1:5000")
    socketio.run(app, debug=True)
