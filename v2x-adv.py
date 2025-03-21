#!/usr/bin/env python3
# Enhanced V2X Simulation Application with Plotly Visualizations

import sys
import random
import math
import time
import os
from collections import deque, defaultdict
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional, Union, Callable, Any

# PyQt Imports
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QGraphicsItemGroup,
    QGraphicsEllipseItem,
    QGraphicsRectItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QGraphicsSimpleTextItem,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QComboBox,
    QSpinBox,
    QTabWidget,
    QGroupBox,
    QFormLayout,
    QCheckBox,
    QDoubleSpinBox,
    QStatusBar,
    QRadioButton,
    QGridLayout,
    QFrame,
    QSplitter,
    QMenu,
    QDialog,
    QDialogButtonBox,
    QToolBar,
    QFileDialog,
    QScrollArea,
)
from PyQt6.QtGui import (
    QPen,
    QBrush,
    QColor,
    QPainterPath,
    QPainter,
    QFont,
    QIcon,
    QRadialGradient,
    QLinearGradient,
    QPainterPath,
    QTransform,
    QPolygonF,
    QKeySequence,
    QAction,
    QPixmap,
)
from PyQt6.QtCore import (
    Qt,
    QTimer,
    QPointF,
    QRectF,
    QLineF,
    pyqtSignal,
    QObject,
    QEvent,
    QSize,
    QTime,
    QElapsedTimer,
    QPropertyAnimation,
    QVariantAnimation,
    QSettings,
    QUrl,
)
from PyQt6.QtWebEngineWidgets import QWebEngineView

# Plotly for advanced visualizations
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import numpy as np


# Constants and Configuration
class Config:
    # Simulation parameters
    ROAD_WIDTH = 20
    VEHICLE_SIZE = 16
    INFRASTRUCTURE_SIZE = 24
    MAX_SPEED = 150  # km/h
    MAX_RANGE = 300  # meters (communication range)
    UPDATE_INTERVAL = 30  # milliseconds (~33 fps)

    # Performance settings
    CULLING_MARGIN = 100  # pixels outside view to still render
    ANIMATION_LIMIT = 20  # max concurrent animations
    LOD_THRESHOLD = 0.5  # scale below which to use simplified rendering

    # Visualization settings
    ROAD_COLOR = QColor(50, 50, 50)
    BACKGROUND_COLOR = QColor(230, 230, 230)
    GRID_COLOR = QColor(200, 200, 200, 100)
    GRID_SIZE = 100

    # Communication parameters
    MESSAGE_TTL = 15  # message time-to-live in frames
    COMM_VISUAL_CHANCE = 0.3  # probability to visualize a message

    # Physics parameters
    PHYSICS_STEP = 0.05  # scale factor for movement calculations


# Enum definitions
class VehicleType(Enum):
    CAR = "Car"
    TRUCK = "Truck"
    BUS = "Bus"
    EMERGENCY = "Emergency"
    AUTONOMOUS = "Autonomous"

    @classmethod
    def get_color(cls, type_value):
        colors = {
            cls.CAR: QColor(30, 144, 255),
            cls.TRUCK: QColor(0, 128, 0),
            cls.BUS: QColor(255, 140, 0),
            cls.EMERGENCY: QColor(255, 0, 0),
            cls.AUTONOMOUS: QColor(138, 43, 226),
        }
        return colors.get(type_value, QColor(100, 100, 100))

    @classmethod
    def get_plotly_color(cls, type_value):
        colors = {
            cls.CAR: "rgb(30, 144, 255)",
            cls.TRUCK: "rgb(0, 128, 0)",
            cls.BUS: "rgb(255, 140, 0)",
            cls.EMERGENCY: "rgb(255, 0, 0)",
            cls.AUTONOMOUS: "rgb(138, 43, 226)",
        }
        return colors.get(type_value, "rgb(100, 100, 100)")


class InfrastructureType(Enum):
    TRAFFIC_LIGHT = "Traffic Light"
    ROAD_SIGN = "Road Sign"
    BASE_STATION = "Base Station"
    SENSOR = "Sensor"

    @classmethod
    def get_color(cls, type_value):
        colors = {
            cls.TRAFFIC_LIGHT: QColor(255, 215, 0),
            cls.ROAD_SIGN: QColor(255, 192, 203),
            cls.BASE_STATION: QColor(0, 255, 255),
            cls.SENSOR: QColor(255, 105, 180),
        }
        return colors.get(type_value, QColor(150, 150, 150))

    @classmethod
    def get_plotly_color(cls, type_value):
        colors = {
            cls.TRAFFIC_LIGHT: "rgb(255, 215, 0)",
            cls.ROAD_SIGN: "rgb(255, 192, 203)",
            cls.BASE_STATION: "rgb(0, 255, 255)",
            cls.SENSOR: "rgb(255, 105, 180)",
        }
        return colors.get(type_value, "rgb(150, 150, 150)")


class MessageType(Enum):
    BASIC_SAFETY = "Basic Safety Message"
    SIGNAL_PHASE = "Signal Phase and Timing"
    MAP_DATA = "Map Data"
    EMERGENCY_ALERT = "Emergency Vehicle Alert"
    TRAVELER_INFO = "Traveler Information"
    ROADSIDE_ALERT = "Roadside Alert"

    @classmethod
    def get_color(cls, type_value):
        colors = {
            cls.BASIC_SAFETY: QColor(255, 0, 0, 150),
            cls.SIGNAL_PHASE: QColor(0, 255, 0, 150),
            cls.MAP_DATA: QColor(0, 0, 255, 150),
            cls.EMERGENCY_ALERT: QColor(255, 165, 0, 150),
            cls.TRAVELER_INFO: QColor(0, 255, 255, 150),
            cls.ROADSIDE_ALERT: QColor(255, 0, 255, 150),
        }
        return colors.get(type_value, QColor(255, 255, 255, 150))

    @classmethod
    def get_plotly_color(cls, type_value):
        colors = {
            cls.BASIC_SAFETY: "rgba(255, 0, 0, 0.6)",
            cls.SIGNAL_PHASE: "rgba(0, 255, 0, 0.6)",
            cls.MAP_DATA: "rgba(0, 0, 255, 0.6)",
            cls.EMERGENCY_ALERT: "rgba(255, 165, 0, 0.6)",
            cls.TRAVELER_INFO: "rgba(0, 255, 255, 0.6)",
            cls.ROADSIDE_ALERT: "rgba(255, 0, 255, 0.6)",
        }
        return colors.get(type_value, "rgba(255, 255, 255, 0.6)")


class TrafficLightState(Enum):
    RED = "red"
    YELLOW = "yellow"
    GREEN = "green"

    @classmethod
    def get_color(cls, state):
        colors = {
            cls.RED: QColor(255, 0, 0),
            cls.YELLOW: QColor(255, 255, 0),
            cls.GREEN: QColor(0, 255, 0),
        }
        return colors.get(state, QColor(100, 100, 100))


class SimulationMode(Enum):
    NORMAL = auto()
    COMMUNICATION_FOCUS = auto()
    TRAFFIC_ANALYSIS = auto()
    EMERGENCY_RESPONSE = auto()


# Utility Functions
def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2)


def angle_between_points(p1, p2):
    """Calculate angle in radians between two points"""
    return math.atan2(p2.y() - p1.y(), p2.x() - p1.x())


def normalize_angle(angle):
    """Normalize angle to range [0, 2*pi]"""
    return angle % (2 * math.pi)


def angle_difference(angle1, angle2):
    """Calculate smallest difference between two angles"""
    return ((angle2 - angle1 + math.pi) % (2 * math.pi)) - math.pi


def create_vehicle_polygon(size, direction):
    """Create a vehicle polygon (arrow-shaped)"""
    points = []
    # Front point
    points.append(QPointF(size * math.cos(direction), size * math.sin(direction)))
    # Left back point
    angle_left = direction + 2.5
    points.append(
        QPointF(size * 0.7 * math.cos(angle_left), size * 0.7 * math.sin(angle_left))
    )
    # Right back point
    angle_right = direction - 2.5
    points.append(
        QPointF(size * 0.7 * math.cos(angle_right), size * 0.7 * math.sin(angle_right))
    )
    return QPolygonF(points)


def build_path_from_points(points):
    """Create a QPainterPath from a list of points"""
    path = QPainterPath()
    if not points:
        return path
    path.moveTo(points[0])
    for point in points[1:]:
        path.lineTo(point)
    return path


# Event Management System
class EventManager(QObject):
    """Central event dispatcher for simulation events"""

    vehicle_added = pyqtSignal(object)
    vehicle_removed = pyqtSignal(object)
    infrastructure_added = pyqtSignal(object)
    infrastructure_removed = pyqtSignal(object)
    road_added = pyqtSignal(object)
    message_sent = pyqtSignal(object, list)  # message, receivers
    collision_detected = pyqtSignal(object, object)  # vehicle1, vehicle2
    simulation_reset = pyqtSignal()
    statistics_updated = pyqtSignal(dict)
    simulation_mode_changed = pyqtSignal(object)  # SimulationMode


# Model Classes
class SimulationObject:
    """Base class for all simulation objects"""

    _next_id = 1

    def __init__(self, x, y, object_type):
        self.id = SimulationObject._next_id
        SimulationObject._next_id += 1
        self.x = x
        self.y = y
        self.type = object_type
        self.messages_sent = 0
        self.messages_received = 0
        self.comm_range = Config.MAX_RANGE
        self.item = None  # Reference to graphics item
        self.visible = True

    def position(self):
        return QPointF(self.x, self.y)

    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.update_graphics_position()

    def update_graphics_position(self):
        if self.item:
            self.item.setPos(self.x, self.y)

    def send_message(self, message_type, sim_environment):
        """Send a V2X message to nearby objects"""
        self.messages_sent += 1
        return sim_environment.transmit_message(self, message_type)

    def receive_message(self, message):
        """Process an incoming V2X message"""
        self.messages_received += 1

    def get_state_dict(self):
        """Get object state as a dictionary for serialization"""
        return {
            "id": self.id,
            "type": str(self.type),
            "x": self.x,
            "y": self.y,
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "comm_range": self.comm_range,
        }

    def to_dict(self):
        """Convert object to dictionary for serialization"""
        return self.get_state_dict()


class Vehicle(SimulationObject):
    """Represents a vehicle in the simulation"""

    def __init__(self, x, y, vehicle_type, speed=0, direction=0):
        super().__init__(x, y, vehicle_type)
        self.speed = speed  # km/h
        self.target_speed = speed
        self.direction = direction  # radians
        self.size = Config.VEHICLE_SIZE
        self.is_braking = False
        self.is_autonomous = vehicle_type == VehicleType.AUTONOMOUS
        self.route = []  # List of waypoints
        self.current_waypoint = None
        self.acceleration = 0
        self.turning_rate = 0
        self.proximity_alert = False
        self.recent_messages = deque(maxlen=10)
        self.driver_behavior = {
            "aggression": random.uniform(0.1, 1.0),
            "caution": random.uniform(0.1, 1.0),
            "reaction_time": random.uniform(0.5, 2.0),
        }

    def update(self, delta_time, environment=None):
        """Update vehicle position based on speed and direction"""
        # Convert km/h to m/s, then to pixels per time step
        speed_ms = self.speed * 0.277778  # km/h to m/s
        distance_per_step = speed_ms * delta_time * Config.PHYSICS_STEP

        # Gradually adjust speed toward target speed
        if abs(self.speed - self.target_speed) > 1.0:
            if self.speed < self.target_speed:
                self.speed += min(
                    5.0 * self.driver_behavior["aggression"],
                    self.target_speed - self.speed,
                )  # Accelerate
                self.is_braking = False
            else:
                self.speed -= min(
                    8.0 * self.driver_behavior["caution"],
                    self.speed - self.target_speed,
                )  # Decelerate faster
                self.is_braking = True
        else:
            self.speed = self.target_speed  # Reached target
            self.is_braking = False

        # Check for nearby vehicles and adjust speed if needed
        if environment:
            self.check_proximity(environment)

        # Move vehicle
        self.x += math.cos(self.direction) * distance_per_step
        self.y += math.sin(self.direction) * distance_per_step

        # Update graphics position
        self.update_graphics_position()

        # Check if we reached a waypoint
        if (
            self.current_waypoint
            and distance(self.position(), self.current_waypoint) < 15
        ):
            if self.route:
                self.current_waypoint = self.route.pop(0)
                new_direction = angle_between_points(
                    self.position(), self.current_waypoint
                )

                # Smooth direction change
                angle_diff = angle_difference(self.direction, new_direction)

                # Adjust speed based on turn sharpness
                turn_factor = abs(angle_diff) / math.pi
                if turn_factor > 0.3:  # If turning significantly, slow down
                    self.target_speed = max(20, self.target_speed * (1 - turn_factor))
                else:
                    self.target_speed = min(
                        Config.MAX_SPEED,
                        self.target_speed + 10 * self.driver_behavior["aggression"],
                    )

                # Apply turning rate based on vehicle properties and turn angle
                max_turn_rate = 1.5 * (1.0 - turn_factor)  # radians per second
                turn_direction = 1 if angle_diff > 0 else -1
                self.turning_rate = turn_direction * max_turn_rate

                # Apply partial turn immediately
                self.direction += self.turning_rate * delta_time
            else:
                self.current_waypoint = None
                self.turning_rate = 0

        # Continue any ongoing turn
        if self.turning_rate != 0:
            # Apply turning rate
            self.direction += self.turning_rate * delta_time

            # Reduce turning rate over time (damping)
            self.turning_rate *= 0.95

            # If turning rate is very small, stop turning
            if abs(self.turning_rate) < 0.01:
                self.turning_rate = 0

        # Normalize direction to keep it in [0, 2π]
        self.direction = normalize_angle(self.direction)

        # Check for boundary conditions - wrap around screen
        boundary = 2000  # Screen boundary
        if self.x < -boundary:
            self.x = boundary
        if self.x > boundary:
            self.x = -boundary
        if self.y < -boundary:
            self.y = boundary
        if self.y > boundary:
            self.y = -boundary

    def check_proximity(self, environment):
        """Check proximity to other vehicles and adjust behavior"""
        self.proximity_alert = False

        # Detection zone in front of vehicle
        detection_range = (
            self.speed * 0.277778 * self.driver_behavior["reaction_time"] * 2
        )
        detection_range = max(
            50, min(detection_range, 200)
        )  # Between 50 and 200 pixels

        # Look for vehicles in front
        for other in environment.vehicles:
            if other == self:
                continue

            # Calculate relative position vector
            dx = other.x - self.x
            dy = other.y - self.y

            # Distance between vehicles
            dist = math.sqrt(dx * dx + dy * dy)

            # Skip if too far
            if dist > detection_range + other.size:
                continue

            # Check if other vehicle is in front
            # Calculate angle to other vehicle
            angle_to_other = math.atan2(dy, dx)
            angle_diff = abs(angle_difference(self.direction, angle_to_other))

            # If vehicle is roughly in front (within 45 degrees)
            if angle_diff < math.pi / 4:
                # Calculate relative speed
                rel_speed = self.speed - other.speed * math.cos(angle_diff)

                # If we're approaching them
                if rel_speed > 0:
                    # Calculate time-to-collision
                    ttc = (
                        dist / (rel_speed * 0.277778) if rel_speed > 0 else float("inf")
                    )

                    # If collision is imminent
                    if ttc < 3.0:  # seconds
                        self.proximity_alert = True

                        # Apply braking proportional to urgency
                        brake_intensity = (
                            min(1.0, 2.5 / ttc) * self.driver_behavior["caution"]
                        )
                        self.apply_brake(intensity=brake_intensity)

                        # If very close, also steer slightly
                        if ttc < 1.0 and abs(angle_diff) > 0.1:
                            # Steer away from the other vehicle
                            steer_direction = (
                                1
                                if angle_difference(self.direction, angle_to_other) > 0
                                else -1
                            )
                            self.turning_rate = steer_direction * 0.5

    def apply_brake(self, intensity=0.3):
        """Apply brakes to slow down the vehicle"""
        self.target_speed *= 1 - intensity
        self.is_braking = True

    def accelerate(self, amount=5):
        """Increase vehicle speed"""
        self.target_speed = min(self.target_speed + amount, Config.MAX_SPEED)
        self.is_braking = False

    def set_route(self, waypoints):
        """Set a route for the vehicle to follow"""
        self.route = waypoints.copy()
        if self.route:
            self.current_waypoint = self.route.pop(0)
            self.direction = angle_between_points(
                self.position(), self.current_waypoint
            )

    def get_state_dict(self):
        """Get vehicle state as a dictionary for serialization"""
        state = super().get_state_dict()
        state.update(
            {
                "speed": self.speed,
                "target_speed": self.target_speed,
                "direction": self.direction,
                "is_braking": self.is_braking,
                "is_autonomous": self.is_autonomous,
                "route": [(p.x(), p.y()) for p in self.route] if self.route else [],
                "current_waypoint": (
                    (self.current_waypoint.x(), self.current_waypoint.y())
                    if self.current_waypoint
                    else None
                ),
                "driver_behavior": self.driver_behavior,
                "proximity_alert": self.proximity_alert,
            }
        )
        return state


class Infrastructure(SimulationObject):
    """Represents infrastructure elements like traffic lights, road signs, etc."""

    def __init__(self, x, y, infra_type, state=None):
        super().__init__(x, y, infra_type)
        self.state = state or "idle"
        self.size = Config.INFRASTRUCTURE_SIZE
        self.connected_vehicles = set()
        self.data_collected = defaultdict(int)

        # Traffic light specific properties
        if infra_type == InfrastructureType.TRAFFIC_LIGHT:
            self.cycle_times = {"red": 30, "yellow": 5, "green": 30}
            phases = [
                TrafficLightState.RED,
                TrafficLightState.GREEN,
                TrafficLightState.YELLOW,
            ]
            self.current_phase = random.choice(phases)
            self.phase_time = random.uniform(
                0, self.cycle_times[self.current_phase.value]
            )

        # Base station properties
        if infra_type == InfrastructureType.BASE_STATION:
            self.comm_range = 500  # Extended range for base stations
            self.connected_count = 0
            self.data_rate = 0  # Mbps

    def update(self, delta_time, environment=None):
        """Update infrastructure state"""
        if self.type == InfrastructureType.TRAFFIC_LIGHT:
            self.phase_time += delta_time

            # Check if we need to change the traffic light phase
            current_phase_value = self.current_phase.value
            if (
                current_phase_value == "red"
                and self.phase_time >= self.cycle_times["red"]
            ):
                self.current_phase = TrafficLightState.GREEN
                self.phase_time = 0
            elif (
                current_phase_value == "green"
                and self.phase_time >= self.cycle_times["green"]
            ):
                self.current_phase = TrafficLightState.YELLOW
                self.phase_time = 0
            elif (
                current_phase_value == "yellow"
                and self.phase_time >= self.cycle_times["yellow"]
            ):
                self.current_phase = TrafficLightState.RED
                self.phase_time = 0

            # Signal the graphics item to update
            if self.item:
                self.item.update()  # Just trigger a repaint instead of calling setBrush

        # Base station updates
        elif self.type == InfrastructureType.BASE_STATION:
            if environment:
                # Count connected vehicles within range
                connected = 0
                for vehicle in environment.vehicles:
                    if distance(self.position(), vehicle.position()) <= self.comm_range:
                        connected += 1
                        # Collect data from vehicle
                        message_count = (
                            vehicle.messages_sent + vehicle.messages_received
                        )
                        self.data_collected[vehicle.id] += message_count

                self.connected_count = connected
                # Calculate data rate based on connected vehicles
                self.data_rate = (
                    connected * 2.5
                )  # Simple model: each vehicle uses 2.5 Mbps

                # Signal the graphics item to update
                if self.item:
                    self.item.update()  # Just trigger a repaint

        # Road signs might update state based on nearby vehicles
        elif self.type == InfrastructureType.ROAD_SIGN:
            if environment:
                nearby_count = 0
                for vehicle in environment.vehicles:
                    if distance(self.position(), vehicle.position()) <= self.comm_range:
                        nearby_count += 1

                # Update display based on vehicle count
                if nearby_count > 5:
                    self.state = "congested"
                elif nearby_count > 2:
                    self.state = "busy"
                elif nearby_count > 0:
                    self.state = "active"
                else:
                    self.state = "idle"

                # Signal the graphics item to update
                if self.item:
                    self.item.update()  # Just trigger a repaint

        # Sensor nodes collect data
        elif self.type == InfrastructureType.SENSOR:
            if (
                environment and random.random() < 0.1
            ):  # 10% chance to record data each update
                for vehicle in environment.vehicles:
                    if distance(self.position(), vehicle.position()) <= self.comm_range:
                        self.data_collected["vehicle_detections"] += 1

                # Signal the graphics item to update if needed
                if self.item:
                    self.item.update()

    def get_state_dict(self):
        """Get infrastructure state as a dictionary for serialization"""
        state = super().get_state_dict()

        # Add type-specific properties
        if self.type == InfrastructureType.TRAFFIC_LIGHT:
            state.update(
                {
                    "current_phase": self.current_phase.value,
                    "phase_time": self.phase_time,
                    "cycle_times": self.cycle_times,
                }
            )
        elif self.type == InfrastructureType.BASE_STATION:
            state.update(
                {"connected_count": self.connected_count, "data_rate": self.data_rate}
            )
        elif self.type == InfrastructureType.ROAD_SIGN:
            state.update({"state": self.state})
        elif self.type == InfrastructureType.SENSOR:
            state.update({"data_collected": dict(self.data_collected)})

        return state


class Road:
    """Represents a road segment in the simulation"""

    def __init__(
        self,
        start_x,
        start_y,
        end_x,
        end_y,
        lanes=2,
        bidirectional=True,
        speed_limit=100,
        name="",
    ):
        self.start = QPointF(start_x, start_y)
        self.end = QPointF(end_x, end_y)
        self.lanes = lanes
        self.bidirectional = bidirectional
        self.length = distance(self.start, self.end)
        self.angle = angle_between_points(self.start, self.end)
        self.width = Config.ROAD_WIDTH * lanes * (2 if bidirectional else 1)
        self.item = None  # Reference to graphics item
        self.speed_limit = speed_limit
        self.name = name or f"Road {id(self) % 1000}"
        self.traffic_density = 0.0  # 0.0 to 1.0
        self.traffic_flow = (
            []
        )  # List of (time, count) tuples for traffic flow over time

    def get_lane_position(self, distance_along, lane=0, direction=1):
        """Get position at specified distance along the road in a specific lane"""
        lane_offset = (lane + 0.5) * Config.ROAD_WIDTH
        if direction == -1 and self.bidirectional:
            lane_offset += self.lanes * Config.ROAD_WIDTH

        # Calculate base position along road
        base_x = self.start.x() + (distance_along / self.length) * (
            self.end.x() - self.start.x()
        )
        base_y = self.start.y() + (distance_along / self.length) * (
            self.end.y() - self.start.y()
        )

        # Calculate perpendicular offset for lane
        perp_angle = self.angle + math.pi / 2
        offset_x = lane_offset * math.cos(perp_angle)
        offset_y = lane_offset * math.sin(perp_angle)

        return QPointF(base_x + offset_x, base_y + offset_y)

    def update(self, delta_time, environment=None):
        """Update road properties based on traffic"""
        if environment:
            # Count vehicles on this road (simplified)
            count = 0
            for vehicle in environment.vehicles:
                # Check if vehicle is near this road (simplified)
                if self.is_point_near_road(
                    vehicle.position(), max_distance=self.width / 2
                ):
                    count += 1

            # Calculate density (vehicles per unit length)
            capacity = max(1, self.length / 50)  # One vehicle per 50 units
            self.traffic_density = min(1.0, count / capacity)

            # Record traffic flow every few seconds
            current_time = environment.time
            if not self.traffic_flow or current_time - self.traffic_flow[-1][0] > 5:
                self.traffic_flow.append((current_time, count))
                # Keep only the last 100 measurements
                if len(self.traffic_flow) > 100:
                    self.traffic_flow.pop(0)

            # Update visualization
            if self.item:
                self.item.update_traffic_density(self.traffic_density)

    def is_point_near_road(self, point, max_distance=None):
        """Check if a point is near this road segment"""
        if max_distance is None:
            max_distance = self.width / 2

        # Vector from start to end
        road_vec_x = self.end.x() - self.start.x()
        road_vec_y = self.end.y() - self.start.y()
        road_length_sq = road_vec_x * road_vec_x + road_vec_y * road_vec_y

        # Vector from start to point
        point_vec_x = point.x() - self.start.x()
        point_vec_y = point.y() - self.start.y()

        # Calculate projection factor
        projection = (
            point_vec_x * road_vec_x + point_vec_y * road_vec_y
        ) / road_length_sq

        # If projection is outside [0,1], use distance to endpoints
        if projection < 0:
            return distance(point, self.start) <= max_distance
        elif projection > 1:
            return distance(point, self.end) <= max_distance

        # Calculate perpendicular distance
        proj_x = self.start.x() + projection * road_vec_x
        proj_y = self.start.y() + projection * road_vec_y
        perp_dist = distance(point, QPointF(proj_x, proj_y))

        return perp_dist <= max_distance

    def get_state_dict(self):
        """Get road state as a dictionary for serialization"""
        return {
            "start": (self.start.x(), self.start.y()),
            "end": (self.end.x(), self.end.y()),
            "lanes": self.lanes,
            "bidirectional": self.bidirectional,
            "speed_limit": self.speed_limit,
            "name": self.name,
            "traffic_density": self.traffic_density,
        }


class Message:
    """Represents a V2X message"""

    def __init__(self, sender, message_type, content=None):
        self.sender = sender
        self.type = message_type
        self.content = content or {}
        self.timestamp = time.time()
        self.size = random.randint(20, 500)  # Message size in bytes
        self.ttl = 1.0  # Time-to-live in seconds
        self.priority = self._calculate_priority()
        self.id = f"{int(self.timestamp*1000)}-{sender.id}-{random.randint(0, 9999)}"

    def _calculate_priority(self):
        """Calculate message priority based on type"""
        if self.type == MessageType.EMERGENCY_ALERT:
            return 10
        elif self.type == MessageType.BASIC_SAFETY:
            return 8
        elif self.type == MessageType.SIGNAL_PHASE:
            return 6
        elif self.type == MessageType.ROADSIDE_ALERT:
            return 5
        elif self.type == MessageType.TRAVELER_INFO:
            return 3
        else:
            return 1

    def __str__(self):
        return (
            f"Message({self.type.value}) from {self.sender.type.value} {self.sender.id}"
        )

    def get_state_dict(self):
        """Get message as a dictionary for serialization"""
        return {
            "sender_id": self.sender.id,
            "sender_type": str(self.sender.type),
            "type": self.type.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "size": self.size,
            "priority": self.priority,
            "id": self.id,
        }


class SimulationEnvironment:
    """Manages the entire simulation environment"""

    def __init__(self, event_manager=None):
        self.vehicles = []
        self.infrastructure = []
        self.roads = []
        self.messages = []
        self.active_message_graphics = []
        self.time = 0
        self.event_manager = event_manager or EventManager()
        self.statistics = {
            "messages_sent": 0,
            "messages_received": 0,
            "vehicle_count": 0,
            "infrastructure_count": 0,
            "average_speed": 0,
            "collisions": 0,
            "message_types": {},
            "time": 0,
            "emergency_response_time": 0,
            "traffic_efficiency": 1.0,
            "communication_reliability": 1.0,
            "network_load": 0.0,
            "vehicles_by_type": {},
            "message_success_rate": 100.0,
        }

        # History data for charts
        self.history = {
            "time_points": [],
            "vehicle_counts": [],
            "avg_speeds": [],
            "message_counts": [],
            "network_loads": [],
        }

        # Message history - for keeping a limited buffer
        self.message_history = deque(maxlen=500)

        # Simulation mode
        self.mode = SimulationMode.NORMAL

        # Communication models
        self.communication_model = "simple"  # "simple", "distance-based", "realistic"

        # Weather and environment effects
        self.weather_condition = "clear"  # "clear", "rain", "fog", "snow"
        self.weather_effects = {
            "clear": {
                "comm_range_factor": 1.0,
                "max_speed_factor": 1.0,
                "reliability_factor": 1.0,
            },
            "rain": {
                "comm_range_factor": 0.8,
                "max_speed_factor": 0.8,
                "reliability_factor": 0.9,
            },
            "fog": {
                "comm_range_factor": 0.6,
                "max_speed_factor": 0.6,
                "reliability_factor": 0.7,
            },
            "snow": {
                "comm_range_factor": 0.5,
                "max_speed_factor": 0.5,
                "reliability_factor": 0.8,
            },
        }

        # Performance optimization
        self.update_counter = 0
        self.statistics_interval = 5  # Update statistics every 5 frames

    def add_vehicle(self, vehicle):
        """Add a vehicle to the simulation"""
        self.vehicles.append(vehicle)
        self.statistics["vehicle_count"] += 1

        # Update vehicle type statistics
        vtype = vehicle.type.value
        if vtype in self.statistics["vehicles_by_type"]:
            self.statistics["vehicles_by_type"][vtype] += 1
        else:
            self.statistics["vehicles_by_type"][vtype] = 1

        # Notify listeners
        if self.event_manager:
            self.event_manager.vehicle_added.emit(vehicle)

        return vehicle

    def remove_vehicle(self, vehicle):
        """Remove a vehicle from the simulation"""
        if vehicle in self.vehicles:
            self.vehicles.remove(vehicle)
            self.statistics["vehicle_count"] -= 1

            # Update vehicle type statistics
            vtype = vehicle.type.value
            if vtype in self.statistics["vehicles_by_type"]:
                self.statistics["vehicles_by_type"][vtype] -= 1

            # Notify listeners
            if self.event_manager:
                self.event_manager.vehicle_removed.emit(vehicle)

    def add_infrastructure(self, infrastructure):
        """Add an infrastructure element to the simulation"""
        self.infrastructure.append(infrastructure)
        self.statistics["infrastructure_count"] += 1

        # Notify listeners
        if self.event_manager:
            self.event_manager.infrastructure_added.emit(infrastructure)

        return infrastructure

    def add_road(self, road):
        """Add a road to the simulation"""
        self.roads.append(road)

        # Notify listeners
        if self.event_manager:
            self.event_manager.road_added.emit(road)

        return road

    def update(self, delta_time):
        """Update the entire simulation for one time step"""
        self.time += delta_time

        # Get weather effects
        weather = self.weather_effects.get(
            self.weather_condition, self.weather_effects["clear"]
        )

        # Apply weather effects to vehicles
        max_speed_factor = weather["max_speed_factor"]

        # Update all vehicles
        for vehicle in self.vehicles:
            # Adjust max speed for weather
            vehicle.update(delta_time, self)

        # Update all infrastructure
        for infra in self.infrastructure:
            infra.update(delta_time, self)

        # Update all roads
        for road in self.roads:
            road.update(delta_time, self)

        # Simulate V2X communication
        self.simulate_communication()

        # Update statistics periodically for performance
        self.update_counter += 1
        if self.update_counter >= self.statistics_interval:
            self.update_counter = 0
            self.update_statistics()

            # Record history data
            self.record_history_data()

        return self.active_message_graphics

    def record_history_data(self):
        """Record historical data for charts and analysis"""
        self.history["time_points"].append(self.time)
        self.history["vehicle_counts"].append(self.statistics["vehicle_count"])
        self.history["avg_speeds"].append(self.statistics["average_speed"])
        self.history["message_counts"].append(self.statistics["messages_sent"])
        self.history["network_loads"].append(self.statistics["network_load"])

        # Keep history to a reasonable size
        max_history = 1000
        if len(self.history["time_points"]) > max_history:
            for key in self.history:
                self.history[key] = self.history[key][-max_history:]

    def simulate_communication(self):
        """Simulate V2X communication between entities"""
        # Get communication model adjustments based on weather
        weather = self.weather_effects.get(
            self.weather_condition, self.weather_effects["clear"]
        )
        comm_range_factor = weather["comm_range_factor"]
        reliability_factor = weather["reliability_factor"]

        # Different types of entities send different message types with different frequencies
        for vehicle in self.vehicles:
            # Each entity has a chance to send a message, weighted by type
            if random.random() < 0.05:  # 5% chance per update
                # Autonomous vehicles communicate more frequently
                if vehicle.is_autonomous:
                    message_type = MessageType.BASIC_SAFETY
                elif vehicle.is_braking:
                    message_type = MessageType.EMERGENCY_ALERT
                elif vehicle.proximity_alert:
                    message_type = MessageType.BASIC_SAFETY
                else:
                    message_type = random.choice(
                        [MessageType.BASIC_SAFETY, MessageType.TRAVELER_INFO]
                    )

                # Create message content
                content = {
                    "speed": vehicle.speed,
                    "direction": vehicle.direction,
                    "position": (vehicle.x, vehicle.y),
                }

                if vehicle.is_braking:
                    content["braking"] = True

                self.transmit_message(vehicle, message_type, content)

        # Infrastructure communications
        for infra in self.infrastructure:
            if random.random() < 0.08:  # 8% chance per update
                content = {
                    "position": (infra.x, infra.y),
                    "infrastructure_id": infra.id,
                }

                if infra.type == InfrastructureType.TRAFFIC_LIGHT:
                    message_type = MessageType.SIGNAL_PHASE
                    content["phase"] = infra.current_phase.value
                    content["time_remaining"] = (
                        infra.cycle_times[infra.current_phase.value] - infra.phase_time
                    )
                elif infra.type == InfrastructureType.ROAD_SIGN:
                    message_type = MessageType.ROADSIDE_ALERT
                    content["state"] = infra.state
                elif infra.type == InfrastructureType.BASE_STATION:
                    message_type = MessageType.MAP_DATA
                    content["connected_count"] = infra.connected_count
                    content["data_rate"] = infra.data_rate
                else:
                    message_type = random.choice(list(MessageType))

                self.transmit_message(infra, message_type, content)

    def transmit_message(self, sender, message_type, content=None):
        """Transmit a message from sender to all receivers in range"""
        # Create the message
        message = Message(sender, message_type, content)
        self.message_history.append(message)
        self.statistics["messages_sent"] += 1

        # Track message types for statistics
        message_type_value = message_type.value
        if message_type_value in self.statistics["message_types"]:
            self.statistics["message_types"][message_type_value] += 1
        else:
            self.statistics["message_types"][message_type_value] = 1

        # Get weather effects on communication
        weather = self.weather_effects.get(
            self.weather_condition, self.weather_effects["clear"]
        )
        comm_range_factor = weather["comm_range_factor"]
        reliability_factor = weather["reliability_factor"]

        # Find all receivers in range
        receivers = []
        sender_pos = sender.position()
        effective_range = sender.comm_range * comm_range_factor

        # Apply communication model
        if self.communication_model == "simple":
            # Simple model: fixed range, perfect transmission
            for receiver in self.vehicles + self.infrastructure:
                if receiver != sender:
                    if distance(sender_pos, receiver.position()) <= effective_range:
                        # Check reliability
                        if random.random() < reliability_factor:
                            receivers.append(receiver)
                            receiver.receive_message(message)
                            self.statistics["messages_received"] += 1

        elif self.communication_model == "distance-based":
            # Distance-based model: probability decreases with distance
            for receiver in self.vehicles + self.infrastructure:
                if receiver != sender:
                    dist = distance(sender_pos, receiver.position())
                    if dist <= effective_range:
                        # Probability decreases with distance
                        prob = (1 - (dist / effective_range)) * reliability_factor
                        if random.random() < prob:
                            receivers.append(receiver)
                            receiver.receive_message(message)
                            self.statistics["messages_received"] += 1

        elif self.communication_model == "realistic":
            # Realistic model: considers obstacles, interference
            for receiver in self.vehicles + self.infrastructure:
                if receiver != sender:
                    dist = distance(sender_pos, receiver.position())
                    if dist <= effective_range:
                        # Check for line-of-sight (simplified)
                        has_los = True
                        obstacle_count = 0

                        # Count potential obstacles (other vehicles) between sender and receiver
                        for vehicle in self.vehicles:
                            if vehicle != sender and vehicle != receiver:
                                # Check if vehicle is between sender and receiver
                                if self.is_point_between(
                                    vehicle.position(),
                                    sender_pos,
                                    receiver.position(),
                                    tolerance=20,
                                ):
                                    obstacle_count += 1
                                    if obstacle_count >= 3:  # If too many obstacles
                                        has_los = False
                                        break

                        # Calculate success probability
                        base_prob = (1 - (dist / effective_range)) * reliability_factor
                        if not has_los:
                            base_prob *= (
                                0.3  # Reduced probability for non-line-of-sight
                            )

                        # Network congestion effect
                        if self.statistics["network_load"] > 0.7:
                            base_prob *= 1 - (self.statistics["network_load"] - 0.7)

                        if random.random() < base_prob:
                            receivers.append(receiver)
                            receiver.receive_message(message)
                            self.statistics["messages_received"] += 1

        # Notify event listeners
        if self.event_manager and receivers:
            self.event_manager.message_sent.emit(message, receivers)

        return receivers

    def is_point_between(self, point, start, end, tolerance=10):
        """Check if a point is between start and end points (with some tolerance)"""
        # Calculate distance from start to end
        d_start_end = distance(start, end)

        # Calculate distances from point to start and end
        d_point_start = distance(point, start)
        d_point_end = distance(point, end)

        # If point is roughly on the line between start and end
        return abs(d_point_start + d_point_end - d_start_end) <= tolerance

    def update_statistics(self):
        """Update simulation statistics"""
        if self.vehicles:
            self.statistics["average_speed"] = sum(
                v.speed for v in self.vehicles
            ) / len(self.vehicles)
        else:
            self.statistics["average_speed"] = 0

        # Check for collisions (simplified)
        collision_count_before = self.statistics["collisions"]

        for i, v1 in enumerate(self.vehicles):
            for v2 in self.vehicles[i + 1 :]:
                if distance(v1.position(), v2.position()) < (v1.size + v2.size) / 2:
                    self.statistics["collisions"] += 1
                    # Simulate collision response
                    v1.speed = 0
                    v2.speed = 0
                    v1.target_speed = 0
                    v2.target_speed = 0

                    # Notify listeners of collision
                    if self.event_manager:
                        self.event_manager.collision_detected.emit(v1, v2)

        # Calculate traffic efficiency
        if self.roads:
            avg_density = sum(road.traffic_density for road in self.roads) / len(
                self.roads
            )
            # Traffic is most efficient at medium density (around 0.5)
            self.statistics["traffic_efficiency"] = 1.0 - abs(avg_density - 0.5)

        # Calculate network load (based on recent message volume)
        recent_messages = len(self.message_history)
        max_messages = 500  # Maximum number of messages in buffer
        self.statistics["network_load"] = min(1.0, recent_messages / max_messages)

        # Calculate message success rate
        if self.statistics["messages_sent"] > 0:
            self.statistics["message_success_rate"] = (
                self.statistics["messages_received"] / self.statistics["messages_sent"]
            ) * 100
        else:
            self.statistics["message_success_rate"] = 100.0

        # Calculate emergency response time (if in emergency mode)
        if self.mode == SimulationMode.EMERGENCY_RESPONSE:
            # Find emergency vehicles
            emergency_vehicles = [
                v for v in self.vehicles if v.type == VehicleType.EMERGENCY
            ]

            if emergency_vehicles:
                # Calculate average emergency vehicle speed as percentage of max
                avg_emergency_speed = sum(v.speed for v in emergency_vehicles) / len(
                    emergency_vehicles
                )
                speed_percentage = avg_emergency_speed / Config.MAX_SPEED

                # Response time improves with higher speeds
                self.statistics["emergency_response_time"] = max(
                    1.0, 10.0 * (1.0 - speed_percentage)
                )

        # Communication reliability depends on weather and network load
        weather = self.weather_effects.get(
            self.weather_condition, self.weather_effects["clear"]
        )
        reliability_base = weather["reliability_factor"]
        network_impact = max(0, (self.statistics["network_load"] - 0.5) * 0.6)
        self.statistics["communication_reliability"] = max(
            0.1, reliability_base - network_impact
        )

        # Update time in statistics
        self.statistics["time"] = self.time

        # Add history to statistics for charts
        self.statistics["history"] = self.history

        # Emit statistics update event
        if self.event_manager:
            self.event_manager.statistics_updated.emit(self.statistics)

    def set_mode(self, mode):
        """Set simulation mode with associated behavioral changes"""
        if mode == self.mode:
            return

        self.mode = mode

        # Apply mode-specific changes
        if mode == SimulationMode.NORMAL:
            # Reset to normal operation
            for vehicle in self.vehicles:
                if vehicle.type != VehicleType.EMERGENCY:
                    vehicle.target_speed = random.uniform(60, 100)

        elif mode == SimulationMode.COMMUNICATION_FOCUS:
            # Increase communication rate
            self.communication_model = "realistic"

        elif mode == SimulationMode.TRAFFIC_ANALYSIS:
            # Adjust vehicle behaviors for traffic analysis
            for i, vehicle in enumerate(self.vehicles):
                # Create some traffic patterns
                if i % 3 == 0:
                    vehicle.target_speed = random.uniform(40, 60)
                elif i % 3 == 1:
                    vehicle.target_speed = random.uniform(80, 100)

        elif mode == SimulationMode.EMERGENCY_RESPONSE:
            # Give priority to emergency vehicles
            for vehicle in self.vehicles:
                if vehicle.type == VehicleType.EMERGENCY:
                    vehicle.target_speed = Config.MAX_SPEED
                else:
                    # Other vehicles slow down and yield
                    vehicle.target_speed = vehicle.target_speed * 0.7

        # Notify listeners
        if self.event_manager:
            self.event_manager.simulation_mode_changed.emit(mode)

    def set_weather(self, condition):
        """Set weather condition and apply effects"""
        self.weather_condition = condition

    def get_state_dict(self):
        """Get simulation state as a dictionary for serialization"""
        return {
            "time": self.time,
            "vehicles": [v.get_state_dict() for v in self.vehicles],
            "infrastructure": [i.get_state_dict() for i in self.infrastructure],
            "roads": [r.get_state_dict() for r in self.roads],
            "statistics": self.statistics,
            "mode": self.mode.name,
            "weather": self.weather_condition,
            "communication_model": self.communication_model,
        }


# Graphics Classes
class VehicleGraphicsItem(QGraphicsItem):
    """Visual representation of a vehicle"""

    def __init__(self, vehicle):
        super().__init__()
        self.vehicle = vehicle
        self.vehicle.item = self
        self.size = vehicle.size
        self.detailed_mode = True
        self.show_range = False
        self.range_indicator = None

        # For showing communication range
        self.range_circle = None

        # For message animation
        self.message_animation = None

        # Set position in the scene
        self.setPos(vehicle.x, vehicle.y)

        # Enable item caching for better performance
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

        # Set z-value to ensure vehicles are above roads
        self.setZValue(10)

    def boundingRect(self):
        """Define the bounding rectangle for the vehicle"""
        return QRectF(-self.size, -self.size, self.size * 2, self.size * 2)

    def shape(self):
        """Define the shape for collision detection"""
        path = QPainterPath()
        # Create a polygon shape based on vehicle direction
        polygon = create_vehicle_polygon(self.size, self.vehicle.direction)
        path.addPolygon(polygon)
        return path

    def paint(self, painter, option, widget):
        """Paint the vehicle"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get base color
        base_color = VehicleType.get_color(self.vehicle.type)

        # Use either detailed or simplified rendering based on view scale
        if self.detailed_mode:
            # Create a polygon shape based on vehicle direction
            polygon = create_vehicle_polygon(self.size, self.vehicle.direction)

            # Show braking state with color
            if self.vehicle.is_braking:
                painter.setBrush(QBrush(QColor(255, 0, 0)))
            else:
                painter.setBrush(QBrush(base_color))

            # Draw the vehicle body
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawPolygon(polygon)

            # Draw vehicle ID
            painter.setPen(QPen(Qt.GlobalColor.white, 1))
            painter.drawText(
                QRectF(-self.size / 2, -self.size / 2, self.size, self.size),
                Qt.AlignmentFlag.AlignCenter,
                str(self.vehicle.id),
            )

            # Draw warning indicators if needed
            if self.vehicle.proximity_alert:
                painter.setPen(QPen(QColor(255, 165, 0), 2))
                painter.drawEllipse(
                    QRectF(
                        -self.size - 3,
                        -self.size - 3,
                        (self.size + 3) * 2,
                        (self.size + 3) * 2,
                    )
                )

        else:
            # Simplified rendering for distant view
            if self.vehicle.is_braking:
                painter.setBrush(QBrush(QColor(255, 0, 0)))
            else:
                painter.setBrush(QBrush(base_color))

            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.drawEllipse(
                QRectF(-self.size / 2, -self.size / 2, self.size, self.size)
            )

    def update_graphics(self, view_scale=1.0):
        """Update graphics based on vehicle state"""
        self.setPos(self.vehicle.x, self.vehicle.y)

        # Determine rendering detail based on view scale
        self.detailed_mode = view_scale > Config.LOD_THRESHOLD

        # Update range circle if visible
        if self.show_range and self.range_circle:
            self.range_circle.setRect(
                -self.vehicle.comm_range,
                -self.vehicle.comm_range,
                self.vehicle.comm_range * 2,
                self.vehicle.comm_range * 2,
            )
            self.range_circle.setPos(self.vehicle.x, self.vehicle.y)

        # Trigger redraw
        self.update()


class InfrastructureGraphicsItem(QGraphicsItem):
    """Visual representation of infrastructure elements"""

    def __init__(self, infrastructure):
        super().__init__()
        self.infrastructure = infrastructure
        self.infrastructure.item = self
        self.size = infrastructure.size
        self.detailed_mode = True
        self.show_range = False

        # For showing communication range
        self.range_circle = None

        # Label font
        self.label_font = QFont()
        self.label_font.setPointSize(8)

        # Set position in the scene
        self.setPos(infrastructure.x, infrastructure.y)

        # Enable item caching for better performance
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

        # Set z-value to ensure infrastructure is above roads
        self.setZValue(5)

    def boundingRect(self):
        """Define the bounding rectangle for the infrastructure"""
        # Make it big enough to include the label
        return QRectF(-self.size, -self.size, self.size * 2, self.size * 2 + 20)

    def paint(self, painter, option, widget):
        """Paint the infrastructure element"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Get base color from infrastructure type
        base_color = InfrastructureType.get_color(self.infrastructure.type)

        # Use either detailed or simplified rendering based on view scale
        if self.detailed_mode:
            # Special rendering for traffic lights
            if self.infrastructure.type == InfrastructureType.TRAFFIC_LIGHT:
                # Get traffic light state color
                if hasattr(self.infrastructure, "current_phase"):
                    color = TrafficLightState.get_color(
                        self.infrastructure.current_phase
                    )
                else:
                    color = base_color

                # Draw traffic light housing
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.setBrush(QBrush(QColor(50, 50, 50)))
                painter.drawRect(
                    QRectF(-self.size / 2, -self.size / 2, self.size, self.size)
                )

                # Draw light
                painter.setBrush(QBrush(color))
                painter.drawEllipse(
                    QRectF(
                        -self.size / 3,
                        -self.size / 3,
                        self.size * 2 / 3,
                        self.size * 2 / 3,
                    )
                )

            elif self.infrastructure.type == InfrastructureType.BASE_STATION:
                # Draw base station with antenna
                painter.setPen(QPen(Qt.GlobalColor.black, 1))

                # Determine color based on load
                if hasattr(self.infrastructure, "data_rate"):
                    if self.infrastructure.data_rate > 50:  # Heavy load
                        brush_color = QColor(255, 0, 0, 180)
                    elif self.infrastructure.data_rate > 20:  # Medium load
                        brush_color = QColor(255, 165, 0, 180)
                    else:  # Light load
                        brush_color = QColor(0, 255, 255, 180)
                else:
                    brush_color = base_color

                painter.setBrush(QBrush(brush_color))
                painter.drawRect(
                    QRectF(-self.size / 2, -self.size / 3, self.size, self.size * 2 / 3)
                )

                # Draw antenna
                painter.setPen(QPen(Qt.GlobalColor.black, 2))
                painter.drawLine(QLineF(0, -self.size / 3, 0, -self.size))
                painter.drawLine(
                    QLineF(
                        -self.size / 3, -self.size / 2, self.size / 3, -self.size / 2
                    )
                )

                # Draw connectivity indicator
                if hasattr(self.infrastructure, "connected_count"):
                    # Draw small connectivity indicator
                    if self.infrastructure.connected_count > 10:
                        indicator_color = QColor(255, 0, 0)  # Red for high load
                    elif self.infrastructure.connected_count > 5:
                        indicator_color = QColor(255, 165, 0)  # Orange for medium load
                    else:
                        indicator_color = QColor(0, 255, 0)  # Green for low load

                    painter.setBrush(QBrush(indicator_color))
                    painter.drawEllipse(
                        QRectF(
                            self.size / 4, -self.size / 6, self.size / 4, self.size / 4
                        )
                    )

            elif self.infrastructure.type == InfrastructureType.ROAD_SIGN:
                # Determine color based on state
                if hasattr(self.infrastructure, "state"):
                    if self.infrastructure.state == "congested":
                        brush_color = QColor(255, 0, 0, 180)
                    elif self.infrastructure.state == "busy":
                        brush_color = QColor(255, 165, 0, 180)
                    elif self.infrastructure.state == "active":
                        brush_color = QColor(0, 255, 0, 180)
                    else:
                        brush_color = base_color
                else:
                    brush_color = base_color

                # Draw road sign
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.setBrush(QBrush(brush_color))
                painter.drawRect(
                    QRectF(-self.size / 2, -self.size / 2, self.size, self.size)
                )

            else:
                # Draw standard rectangle for other infrastructure
                painter.setPen(QPen(Qt.GlobalColor.black, 1))
                painter.setBrush(QBrush(base_color))
                painter.drawRect(
                    QRectF(-self.size / 2, -self.size / 2, self.size, self.size)
                )

            # Draw infrastructure type label
            painter.setFont(self.label_font)
            painter.setPen(QPen(Qt.GlobalColor.black, 1))

            label_text = self.infrastructure.type.value
            if self.infrastructure.type == InfrastructureType.ROAD_SIGN and hasattr(
                self.infrastructure, "state"
            ):
                label_text += f"\n({self.infrastructure.state})"

            # Calculate text rect
            text_width = painter.fontMetrics().horizontalAdvance(
                label_text.split("\n")[0]
            )
            text_height = painter.fontMetrics().height() * (1 + label_text.count("\n"))

            painter.drawText(
                QRectF(-text_width / 2, self.size / 2 + 2, text_width, text_height),
                Qt.AlignmentFlag.AlignCenter,
                label_text,
            )

        else:
            # Simplified rendering for distant view
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.setBrush(QBrush(base_color))
            painter.drawRect(
                QRectF(-self.size / 2, -self.size / 2, self.size, self.size)
            )

    def update_graphics(self, view_scale=1.0):
        """Update graphics based on infrastructure state"""
        self.setPos(self.infrastructure.x, self.infrastructure.y)

        # Determine rendering detail based on view scale
        self.detailed_mode = view_scale > Config.LOD_THRESHOLD

        # Update range circle if visible
        if self.show_range and self.range_circle:
            self.range_circle.setRect(
                -self.infrastructure.comm_range,
                -self.infrastructure.comm_range,
                self.infrastructure.comm_range * 2,
                self.infrastructure.comm_range * 2,
            )
            self.range_circle.setPos(self.infrastructure.x, self.infrastructure.y)

        # Trigger redraw
        self.update()


class RoadGraphicsItem(QGraphicsItem):
    """Visual representation of a road"""

    def __init__(self, road):
        super().__init__()
        self.road = road
        self.road.item = self
        self.width = road.width
        self.length = road.length
        self.angle = road.angle
        self.detailed_mode = True

        # Calculate road polygon points
        self.road_points = self.calculate_road_points()

        # Lane markings
        self.lane_markings = self.calculate_lane_markings()

        # Heat map for traffic density
        self.traffic_density = 0.0

        # Enable item caching for better performance
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

        # Set z-value to ensure roads are below vehicles
        self.setZValue(0)

    def calculate_road_points(self):
        """Calculate points for the road polygon"""
        # Calculate road outline
        angle = self.road.angle
        perp_angle = angle + math.pi / 2
        half_width = self.road.width / 2

        # Road corners
        p1 = QPointF(
            self.road.start.x() + half_width * math.cos(perp_angle),
            self.road.start.y() + half_width * math.sin(perp_angle),
        )
        p2 = QPointF(
            self.road.start.x() - half_width * math.cos(perp_angle),
            self.road.start.y() - half_width * math.sin(perp_angle),
        )
        p3 = QPointF(
            self.road.end.x() - half_width * math.cos(perp_angle),
            self.road.end.y() - half_width * math.sin(perp_angle),
        )
        p4 = QPointF(
            self.road.end.x() + half_width * math.cos(perp_angle),
            self.road.end.y() + half_width * math.sin(perp_angle),
        )

        return [p1, p2, p3, p4]

    def calculate_lane_markings(self):
        """Calculate lane markings"""
        markings = []
        lanes = self.road.lanes
        start = self.road.start
        end = self.road.end
        angle = self.road.angle
        length = self.road.length

        # Add center line if bidirectional
        if self.road.bidirectional:
            markings.append(("center", QLineF(start, end)))

        # Add lane markings
        for i in range(1, lanes):
            lane_pos_start = self.road.get_lane_position(0, i, 1)
            lane_pos_end = self.road.get_lane_position(length, i, 1)

            markings.append(("lane", QLineF(lane_pos_start, lane_pos_end)))

            if self.road.bidirectional:
                lane_pos_start = self.road.get_lane_position(0, i, -1)
                lane_pos_end = self.road.get_lane_position(length, i, -1)

                markings.append(("lane", QLineF(lane_pos_start, lane_pos_end)))

        return markings

    def boundingRect(self):
        """Define the bounding rectangle for the road"""
        # Calculate from the polygon points
        min_x = min(p.x() for p in self.road_points)
        max_x = max(p.x() for p in self.road_points)
        min_y = min(p.y() for p in self.road_points)
        max_y = max(p.y() for p in self.road_points)

        return QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def paint(self, painter, option, widget):
        """Paint the road"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw road surface
        if self.traffic_density > 0:
            # Use gradient based on traffic density
            if self.traffic_density > 0.8:
                # Heavy traffic - red
                road_color = QColor(255, 50, 50, 200)
            elif self.traffic_density > 0.5:
                # Medium traffic - orange/yellow
                road_color = QColor(255, 165, 0, 200)
            else:
                # Light traffic - green tint
                road_color = QColor(50, 100, 50, 200)
        else:
            # Default road color
            road_color = Config.ROAD_COLOR

        painter.setBrush(QBrush(road_color))
        painter.setPen(QPen(Qt.GlobalColor.black, 1))

        # Draw road polygon
        path = QPainterPath()
        path.moveTo(self.road_points[0])
        for point in self.road_points[1:]:
            path.lineTo(point)
        path.closeSubpath()
        painter.drawPath(path)

        # Draw lane markings if detailed mode
        if self.detailed_mode:
            for marking_type, line in self.lane_markings:
                if marking_type == "center":
                    painter.setPen(QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.DashLine))
                else:
                    painter.setPen(QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine))
                painter.drawLine(line)

            # Draw road name if available
            if hasattr(self.road, "name") and self.road.name:
                # Position text in the middle of the road
                mid_x = (self.road.start.x() + self.road.end.x()) / 2
                mid_y = (self.road.start.y() + self.road.end.y()) / 2

                # Rotate text to align with road
                painter.save()
                painter.translate(mid_x, mid_y)

                # Adjust rotation based on road angle
                text_angle = self.road.angle * 180 / math.pi
                if text_angle > 90 and text_angle < 270:
                    text_angle += 180  # Flip text if road is pointing down/left

                painter.rotate(text_angle)

                # Draw text
                font = QFont()
                font.setPointSize(8)
                painter.setFont(font)
                painter.setPen(QPen(Qt.GlobalColor.white, 1))

                text = self.road.name
                if hasattr(self.road, "speed_limit"):
                    text += f" ({self.road.speed_limit} km/h)"

                text_width = painter.fontMetrics().horizontalAdvance(text)
                text_height = painter.fontMetrics().height()

                painter.drawText(
                    QRectF(-text_width / 2, -text_height / 2, text_width, text_height),
                    Qt.AlignmentFlag.AlignCenter,
                    text,
                )

                painter.restore()

    def update_traffic_density(self, density):
        """Update traffic density visualization"""
        self.traffic_density = density
        self.update()

    def update_graphics(self, view_scale=1.0):
        """Update graphics based on road state"""
        # Determine rendering detail based on view scale
        self.detailed_mode = view_scale > Config.LOD_THRESHOLD

        # Trigger redraw
        self.update()


class CommunicationVisualizerItem(QGraphicsItem):
    """Visualizes communication paths between entities"""

    def __init__(self, sender, receivers, message_type):
        super().__init__()
        self.sender = sender
        self.receivers = receivers
        self.message_type = message_type
        self.time_to_live = Config.MESSAGE_TTL
        self.lines = []

        # Set position at origin (we'll draw lines using scene coordinates)
        self.setPos(0, 0)

        # Create message dots that will travel along lines
        self.message_dots = []

        # Set z-value to be above most items
        self.setZValue(15)

        # Animation properties
        self.animation_progress = 0.0

    def boundingRect(self):
        """Define the bounding rectangle"""
        # This needs to encompass all possible lines
        min_x = min(
            self.sender.x,
            min([r.x for r in self.receivers]) if self.receivers else self.sender.x,
        )
        max_x = max(
            self.sender.x,
            max([r.x for r in self.receivers]) if self.receivers else self.sender.x,
        )
        min_y = min(
            self.sender.y,
            min([r.y for r in self.receivers]) if self.receivers else self.sender.y,
        )
        max_y = max(
            self.sender.y,
            max([r.y for r in self.receivers]) if self.receivers else self.sender.y,
        )

        return QRectF(min_x - 10, min_y - 10, max_x - min_x + 20, max_y - min_y + 20)

    def paint(self, painter, option, widget):
        """Paint communication visualization"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Set color based on message type
        color = MessageType.get_color(self.message_type)

        # Calculate opacity based on time to live
        opacity = self.time_to_live / Config.MESSAGE_TTL
        color.setAlpha(int(150 * opacity))

        # Draw lines to receivers
        painter.setPen(QPen(color, 1, Qt.PenStyle.DotLine))

        sender_pos = QPointF(self.sender.x, self.sender.y)

        for receiver in self.receivers:
            receiver_pos = QPointF(receiver.x, receiver.y)
            painter.drawLine(QLineF(sender_pos, receiver_pos))

            # Draw message dot along the line
            if self.animation_progress > 0:
                dot_pos = QPointF(
                    sender_pos.x()
                    + (receiver_pos.x() - sender_pos.x()) * self.animation_progress,
                    sender_pos.y()
                    + (receiver_pos.y() - sender_pos.y()) * self.animation_progress,
                )

                painter.setBrush(QBrush(color))
                painter.setPen(QPen(Qt.GlobalColor.transparent))
                painter.drawEllipse(dot_pos, 5, 5)

    def update_animation(self):
        """Update animation state"""
        self.time_to_live -= 1

        # Update animation progress (smooth dot movement along lines)
        self.animation_progress += 0.1
        if self.animation_progress > 1.0:
            self.animation_progress = 1.0

        # Force redraw
        self.update()

        # Return True if animation is complete
        return self.time_to_live <= 0


class CommunicationRangeCircle(QGraphicsEllipseItem):
    """Visual indicator of communication range"""

    def __init__(self, entity, parent=None):
        super().__init__(parent)
        self.entity = entity
        self.range = entity.comm_range

        # Create and configure the ellipse
        self.setRect(-self.range, -self.range, self.range * 2, self.range * 2)

        # Set appearance
        color = QColor(0, 200, 255, 40)  # Light blue, semi-transparent
        self.setPen(QPen(QColor(0, 200, 255, 100), 1, Qt.PenStyle.DashLine))
        self.setBrush(QBrush(color))

        # Position at entity
        self.setPos(entity.x, entity.y)

        # Set z-value to be below most items
        self.setZValue(1)

    def update_position(self):
        """Update position to match entity"""
        self.setPos(self.entity.x, self.entity.y)

    def update_range(self, new_range):
        """Update the communication range"""
        self.range = new_range
        self.setRect(-self.range, -self.range, self.range * 2, self.range * 2)


class GridGraphicsItem(QGraphicsItem):
    """Background grid for the simulation scene"""

    def __init__(self, rect, grid_size=Config.GRID_SIZE):
        super().__init__()
        self.rect = rect
        self.grid_size = grid_size

        # Set z-value to be below everything
        self.setZValue(-10)

        # Don't participate in scene collision detection
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)

        # Enable caching for better performance
        self.setCacheMode(QGraphicsItem.CacheMode.DeviceCoordinateCache)

    def boundingRect(self):
        """Define the bounding rectangle"""
        return self.rect

    def paint(self, painter, option, widget):
        """Paint the grid"""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Set grid pen
        painter.setPen(QPen(Config.GRID_COLOR, 1, Qt.PenStyle.DotLine))

        # Draw vertical lines
        x = int(self.rect.left() / self.grid_size) * self.grid_size
        while x <= self.rect.right():
            # Use QLineF for floating point coordinates
            painter.drawLine(QLineF(x, self.rect.top(), x, self.rect.bottom()))
            x += self.grid_size

        # Draw horizontal lines
        y = int(self.rect.top() / self.grid_size) * self.grid_size
        while y <= self.rect.bottom():
            # Use QLineF for floating point coordinates
            painter.drawLine(QLineF(self.rect.left(), y, self.rect.right(), y))
            y += self.grid_size

        # Draw axes with different color
        painter.setPen(QPen(QColor(100, 100, 100), 2))
        painter.drawLine(QLineF(0, self.rect.top(), 0, self.rect.bottom()))
        painter.drawLine(QLineF(self.rect.left(), 0, self.rect.right(), 0))


# GUI Classes
class OptimizedGraphicsView(QGraphicsView):
    """Optimized graphics view with pan and zoom capabilities"""

    scaleChanged = pyqtSignal(float)

    def __init__(self, scene=None, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Set background color
        self.setBackgroundBrush(QBrush(Config.BACKGROUND_COLOR))

        # Zoom factor
        self.zoom_factor = 1.15
        self.current_scale = 1.0

        # Track previous center for smooth panning
        self.previous_position = QPointF(0, 0)

        # Performance optimizations
        self.setOptimizationFlags(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing
            | QGraphicsView.OptimizationFlag.DontSavePainterState
        )

        # For tracking FPS
        self.frame_times = deque(maxlen=60)
        self.fps_timer = QElapsedTimer()
        self.fps_timer.start()
        self.fps = 0

        # Enable caching
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        # Save the scene position
        old_pos = self.mapToScene(event.position().toPoint())

        # Zoom in or out
        if event.angleDelta().y() > 0:
            # Zoom in
            factor = self.zoom_factor
        else:
            # Zoom out
            factor = 1 / self.zoom_factor

        # Update current scale
        self.current_scale *= factor

        # Apply scaling
        self.scale(factor, factor)

        # Emit scale change
        self.scaleChanged.emit(self.current_scale)

        # Ensure the point under the mouse stays the same
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def keyPressEvent(self, event):
        """Handle keyboard navigation"""
        # Arrow keys for panning
        pan_distance = 30
        if event.key() == Qt.Key.Key_Left:
            self.pan(pan_distance, 0)
        elif event.key() == Qt.Key.Key_Right:
            self.pan(-pan_distance, 0)
        elif event.key() == Qt.Key.Key_Up:
            self.pan(0, pan_distance)
        elif event.key() == Qt.Key.Key_Down:
            self.pan(0, -pan_distance)
        # Plus and minus for zooming
        elif event.key() == Qt.Key.Key_Plus:
            self.zoom(self.zoom_factor)
        elif event.key() == Qt.Key.Key_Minus:
            self.zoom(1 / self.zoom_factor)
        # Home key to reset view
        elif event.key() == Qt.Key.Key_Home:
            self.resetView()
        else:
            super().keyPressEvent(event)

    def pan(self, dx, dy):
        """Pan the view by the given delta"""
        self.translate(dx, dy)

    def zoom(self, factor):
        """Zoom the view by the given factor"""
        # Update current scale
        self.current_scale *= factor

        # Apply scaling
        self.scale(factor, factor)

        # Emit scale change
        self.scaleChanged.emit(self.current_scale)

    def resetView(self):
        """Reset the view to its original state"""
        self.resetTransform()
        self.current_scale = 1.0
        self.scaleChanged.emit(self.current_scale)

    def updateFPS(self):
        """Calculate and update frames per second"""
        elapsed = self.fps_timer.elapsed()
        self.frame_times.append(elapsed)
        self.fps_timer.restart()

        if len(self.frame_times) >= 10:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time > 0:
                self.fps = 1000 / avg_frame_time
            else:
                self.fps = 0

    def drawForeground(self, painter, rect):
        """Draw FPS counter and other overlay information"""
        # Draw FPS counter in top-left corner
        painter.setPen(QPen(Qt.GlobalColor.white))
        painter.setFont(QFont("Arial", 10))
        painter.drawText(10, 20, f"FPS: {self.fps:.1f}")

        # Update FPS calculation
        self.updateFPS()


# Plotly Chart Classes
class PlotlyWidget(QWidget):
    """Widget for embedding Plotly charts in PyQt"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.web_view = QWebEngineView()
        self.layout.addWidget(self.web_view)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Default chart data
        self.chart_data = {"data": [], "layout": {"title": "No Data"}}

    def update_chart(self, fig):
        """Update the chart with a Plotly figure"""
        # Convert figure to HTML
        html = fig.to_html(include_plotlyjs="cdn", full_html=False)

        # Construct full HTML with responsive layout
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ margin: 0; padding: 0; }}
                #chart {{ width: 100%; height: 100%; }}
            </style>
        </head>
        <body>
            <div id="chart">
                {html}
            </div>
        </body>
        </html>
        """

        # Load HTML into web view
        self.web_view.setHtml(full_html)


class MessageDistributionChart(PlotlyWidget):
    """Chart showing message type distribution"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.update_empty()

    def update_empty(self):
        """Update with empty data"""
        fig = go.Figure(go.Pie(labels=["No Data"], values=[1], textinfo="label"))
        fig.update_layout(
            title="Message Types", margin=dict(l=10, r=10, t=30, b=10), height=300
        )
        self.update_chart(fig)

    def update_data(self, message_types):
        """Update with new message type data"""
        if not message_types:
            self.update_empty()
            return

        labels = []
        values = []
        colors = []

        # Sort by count, descending
        for msg_type, count in sorted(
            message_types.items(), key=lambda x: x[1], reverse=True
        ):
            labels.append(msg_type)
            values.append(count)

            # Get color for this message type
            for mt in MessageType:
                if mt.value == msg_type:
                    colors.append(MessageType.get_plotly_color(mt))
                    break
            else:
                colors.append("rgb(150, 150, 150)")  # Default color

        fig = go.Figure(
            go.Pie(
                labels=labels,
                values=values,
                textinfo="percent",
                marker=dict(colors=colors),
            )
        )

        fig.update_layout(
            title="Message Types",
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )

        self.update_chart(fig)


class TrafficFlowChart(PlotlyWidget):
    """Chart showing traffic flow over time"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.update_empty()

    def update_empty(self):
        """Update with empty data"""
        fig = go.Figure()
        fig.update_layout(
            title="Vehicle Count Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Vehicle Count",
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )
        self.update_chart(fig)

    def update_data(self, history):
        """Update with new traffic flow data"""
        if not history or not history.get("time_points"):
            self.update_empty()
            return

        time_points = history["time_points"]
        vehicle_counts = history["vehicle_counts"]

        fig = go.Figure()

        # Add vehicle count line
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=vehicle_counts,
                mode="lines",
                name="Vehicle Count",
                line=dict(color="rgb(0, 100, 255)", width=2),
            )
        )

        # Add average speed line on secondary axis if available
        if "avg_speeds" in history and history["avg_speeds"]:
            fig.add_trace(
                go.Scatter(
                    x=time_points,
                    y=history["avg_speeds"],
                    mode="lines",
                    name="Avg Speed (km/h)",
                    line=dict(color="rgb(255, 100, 0)", width=2),
                    yaxis="y2",
                )
            )

        fig.update_layout(
            title="Traffic Flow",
            xaxis_title="Time (s)",
            yaxis_title="Vehicle Count",
            yaxis2=dict(
                title="Speed (km/h)", overlaying="y", side="right", showgrid=False
            ),
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        )

        self.update_chart(fig)


class VehicleDistributionChart(PlotlyWidget):
    """Chart showing vehicle type distribution"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.update_empty()

    def update_empty(self):
        """Update with empty data"""
        fig = go.Figure()
        fig.update_layout(
            title="Vehicle Types",
            xaxis_title="Type",
            yaxis_title="Count",
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )
        self.update_chart(fig)

    def update_data(self, vehicles_by_type):
        """Update with new vehicle distribution data"""
        if not vehicles_by_type:
            self.update_empty()
            return

        types = []
        counts = []
        colors = []

        # Get all vehicle types and sort by count
        for vtype in VehicleType:
            type_name = vtype.value
            count = vehicles_by_type.get(type_name, 0)
            types.append(type_name)
            counts.append(count)
            colors.append(VehicleType.get_plotly_color(vtype))

        fig = go.Figure(go.Bar(x=types, y=counts, marker_color=colors))

        fig.update_layout(
            title="Vehicle Types",
            xaxis_title="Type",
            yaxis_title="Count",
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )

        self.update_chart(fig)


class NetworkLoadChart(PlotlyWidget):
    """Chart showing network load over time"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.update_empty()

    def update_empty(self):
        """Update with empty data"""
        fig = go.Figure()
        fig.update_layout(
            title="Network Load",
            xaxis_title="Time (s)",
            yaxis_title="Load (%)",
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )
        self.update_chart(fig)

    def update_data(self, history):
        """Update with new network load data"""
        if not history or not history.get("time_points"):
            self.update_empty()
            return

        time_points = history["time_points"]
        network_loads = history["network_loads"]

        # Convert to percentage
        network_loads_pct = [load * 100 for load in network_loads]

        fig = go.Figure()

        # Add network load line
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=network_loads_pct,
                mode="lines",
                name="Network Load",
                line=dict(color="rgb(255, 0, 0)", width=2),
                fill="tozeroy",
            )
        )

        fig.update_layout(
            title="Network Load Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Load (%)",
            yaxis_range=[0, 100],
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )

        self.update_chart(fig)


class PerformanceIndicatorsChart(PlotlyWidget):
    """Chart showing key performance indicators"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.update_empty()

    def update_empty(self):
        """Update with empty data"""
        fig = go.Figure()
        fig.update_layout(
            title="Performance Indicators",
            margin=dict(l=10, r=10, t=30, b=10),
            height=300,
        )
        self.update_chart(fig)

    def update_data(self, stats):
        """Update with new performance indicator data"""
        if not stats:
            self.update_empty()
            return

        # Create a gauge chart with performance indicators
        indicators = [
            dict(
                title={"text": "Traffic Efficiency"},
                value=stats.get("traffic_efficiency", 0) * 100,
                domain={"row": 0, "column": 0},
            ),
            dict(
                title={"text": "Comm. Reliability"},
                value=stats.get("communication_reliability", 0) * 100,
                domain={"row": 0, "column": 1},
            ),
            dict(
                title={"text": "Msg Success Rate"},
                value=stats.get("message_success_rate", 0),
                domain={"row": 1, "column": 0},
            ),
            dict(
                title={"text": "Network Load"},
                value=stats.get("network_load", 0) * 100,
                domain={"row": 1, "column": 1},
            ),
        ]

        fig = go.Figure()

        for indicator in indicators:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=indicator["value"],
                    title=indicator["title"],
                    domain=indicator["domain"],
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "rgba(50, 120, 200, 0.8)"},
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": 90,
                        },
                    },
                )
            )

        fig.update_layout(
            grid={"rows": 2, "columns": 2},
            title="Performance Indicators",
            margin=dict(l=20, r=20, t=50, b=20),
            height=400,
        )

        self.update_chart(fig)


class SimulationControlPanel(QWidget):
    """Advanced control panel for simulation parameters"""

    simulationParamsChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Create tab widget for organized controls
        tab_widget = QTabWidget()

        # Tab 1: Simulation Control
        sim_tab = QWidget()
        sim_layout = QVBoxLayout()

        # Simulation control group
        sim_group = QGroupBox("Simulation Control")
        sim_controls = QVBoxLayout()

        # Simulation speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Simulation Speed:"))
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(10)
        self.speed_slider.setMaximum(200)
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(10)
        speed_layout.addWidget(self.speed_slider)
        self.speed_label = QLabel("100%")
        speed_layout.addWidget(self.speed_label)
        sim_controls.addLayout(speed_layout)

        # Control buttons
        buttons_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.pause_button = QPushButton("Pause")
        self.pause_button.setIcon(QIcon.fromTheme("media-playback-pause"))
        self.reset_button = QPushButton("Reset")
        self.reset_button.setIcon(QIcon.fromTheme("edit-undo"))
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.reset_button)
        sim_controls.addLayout(buttons_layout)

        # Simulation mode group
        mode_group = QGroupBox("Simulation Mode")
        mode_layout = QVBoxLayout()

        self.mode_buttons = {}
        for mode in SimulationMode:
            radio = QRadioButton(mode.name.replace("_", " ").title())
            if mode == SimulationMode.NORMAL:
                radio.setChecked(True)
            mode_layout.addWidget(radio)
            self.mode_buttons[mode] = radio
            radio.toggled.connect(
                lambda checked, m=mode: self.onModeChanged(m, checked)
            )

        mode_group.setLayout(mode_layout)
        sim_controls.addWidget(mode_group)

        # Weather effects
        weather_group = QGroupBox("Weather Conditions")
        weather_layout = QVBoxLayout()

        self.weather_combo = QComboBox()
        self.weather_combo.addItems(["clear", "rain", "fog", "snow"])
        self.weather_combo.setCurrentText("clear")
        self.weather_combo.currentTextChanged.connect(self.onWeatherChanged)
        weather_layout.addWidget(self.weather_combo)

        weather_group.setLayout(weather_layout)
        sim_controls.addWidget(weather_group)

        sim_group.setLayout(sim_controls)
        sim_layout.addWidget(sim_group)

        # View controls
        view_group = QGroupBox("View Controls")
        view_layout = QVBoxLayout()

        self.show_grid_checkbox = QCheckBox("Show Grid")
        self.show_grid_checkbox.setChecked(True)
        view_layout.addWidget(self.show_grid_checkbox)

        self.show_comm_checkbox = QCheckBox("Show Communications")
        self.show_comm_checkbox.setChecked(True)
        view_layout.addWidget(self.show_comm_checkbox)

        self.show_ranges_checkbox = QCheckBox("Show Communication Ranges")
        self.show_ranges_checkbox.setChecked(False)
        view_layout.addWidget(self.show_ranges_checkbox)

        self.show_stats_checkbox = QCheckBox("Show Statistics Overlay")
        self.show_stats_checkbox.setChecked(True)
        view_layout.addWidget(self.show_stats_checkbox)

        view_group.setLayout(view_layout)
        sim_layout.addWidget(view_group)

        # Save/load controls
        io_layout = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.setIcon(QIcon.fromTheme("document-save"))
        self.load_button = QPushButton("Load")
        self.load_button.setIcon(QIcon.fromTheme("document-open"))
        io_layout.addWidget(self.save_button)
        io_layout.addWidget(self.load_button)
        sim_layout.addLayout(io_layout)

        sim_tab.setLayout(sim_layout)
        tab_widget.addTab(sim_tab, "Simulation")

        # Tab 2: Vehicles
        vehicles_tab = QWidget()
        vehicles_layout = QVBoxLayout()

        # Vehicle control group
        vehicle_group = QGroupBox("Vehicle Settings")
        vehicle_layout = QFormLayout()

        # Vehicle number control
        self.vehicle_count = QSpinBox()
        self.vehicle_count.setMinimum(0)
        self.vehicle_count.setMaximum(500)
        self.vehicle_count.setValue(20)
        vehicle_layout.addRow("Number of Vehicles:", self.vehicle_count)

        # Vehicle types with colors
        self.vehicle_type_combo = QComboBox()
        for vtype in VehicleType:
            self.vehicle_type_combo.addItem(vtype.value)
        vehicle_layout.addRow("Add Vehicle Type:", self.vehicle_type_combo)

        # Vehicle speed range
        speed_layout = QHBoxLayout()
        self.min_speed = QSpinBox()
        self.min_speed.setMinimum(0)
        self.min_speed.setMaximum(100)
        self.min_speed.setValue(30)
        speed_layout.addWidget(self.min_speed)
        speed_layout.addWidget(QLabel("-"))
        self.max_speed = QSpinBox()
        self.max_speed.setMinimum(0)
        self.max_speed.setMaximum(150)
        self.max_speed.setValue(90)
        speed_layout.addWidget(self.max_speed)
        speed_layout.addWidget(QLabel("km/h"))
        vehicle_layout.addRow("Speed Range:", speed_layout)

        # Driver behavior
        behavior_layout = QHBoxLayout()
        self.driver_caution = QSlider(Qt.Orientation.Horizontal)
        self.driver_caution.setMinimum(1)
        self.driver_caution.setMaximum(10)
        self.driver_caution.setValue(5)
        self.driver_caution.setTickPosition(QSlider.TickPosition.TicksBelow)
        behavior_layout.addWidget(QLabel("Risky"))
        behavior_layout.addWidget(self.driver_caution)
        behavior_layout.addWidget(QLabel("Cautious"))
        vehicle_layout.addRow("Driver Behavior:", behavior_layout)

        # Add vehicle button
        self.add_vehicle_button = QPushButton("Add Vehicle")
        self.add_vehicle_button.setIcon(QIcon.fromTheme("list-add"))
        vehicle_layout.addRow("", self.add_vehicle_button)

        vehicle_group.setLayout(vehicle_layout)
        vehicles_layout.addWidget(vehicle_group)

        # Vehicle actions
        actions_group = QGroupBox("Vehicle Actions")
        actions_layout = QVBoxLayout()

        # Action buttons
        self.emergency_button = QPushButton("Trigger Emergency Vehicle")
        self.emergency_button.setIcon(QIcon.fromTheme("dialog-warning"))
        actions_layout.addWidget(self.emergency_button)

        self.brake_test_button = QPushButton("Brake Test Scenario")
        actions_layout.addWidget(self.brake_test_button)

        self.platooning_button = QPushButton("Form Vehicle Platoon")
        actions_layout.addWidget(self.platooning_button)

        actions_group.setLayout(actions_layout)
        vehicles_layout.addWidget(actions_group)

        vehicles_tab.setLayout(vehicles_layout)
        tab_widget.addTab(vehicles_tab, "Vehicles")

        # Tab 3: Infrastructure
        infra_tab = QWidget()
        infra_layout = QVBoxLayout()

        # Infrastructure control group
        infra_group = QGroupBox("Infrastructure Settings")
        infra_form = QFormLayout()

        # Infrastructure type selection
        self.infra_type_combo = QComboBox()
        for itype in InfrastructureType:
            self.infra_type_combo.addItem(itype.value)
        infra_form.addRow("Infrastructure Type:", self.infra_type_combo)

        # Infrastructure density slider
        self.infra_density = QSlider(Qt.Orientation.Horizontal)
        self.infra_density.setMinimum(1)
        self.infra_density.setMaximum(10)
        self.infra_density.setValue(3)
        density_layout = QHBoxLayout()
        density_layout.addWidget(QLabel("Low"))
        density_layout.addWidget(self.infra_density)
        density_layout.addWidget(QLabel("High"))
        infra_form.addRow("Infrastructure Density:", density_layout)

        # Add infrastructure button
        self.add_infra_button = QPushButton("Add Infrastructure")
        self.add_infra_button.setIcon(QIcon.fromTheme("list-add"))
        infra_form.addRow("", self.add_infra_button)

        infra_group.setLayout(infra_form)
        infra_layout.addWidget(infra_group)

        # Traffic light controls
        tl_group = QGroupBox("Traffic Light Controls")
        tl_layout = QFormLayout()

        self.traffic_sync_checkbox = QCheckBox("Synchronize Traffic Lights")
        tl_layout.addRow("", self.traffic_sync_checkbox)

        self.cycle_time_spin = QSpinBox()
        self.cycle_time_spin.setMinimum(10)
        self.cycle_time_spin.setMaximum(120)
        self.cycle_time_spin.setValue(30)
        self.cycle_time_spin.setSuffix(" sec")
        tl_layout.addRow("Cycle Time:", self.cycle_time_spin)

        tl_group.setLayout(tl_layout)
        infra_layout.addWidget(tl_group)

        infra_tab.setLayout(infra_layout)
        tab_widget.addTab(infra_tab, "Infrastructure")

        # Tab 4: Communication
        comm_tab = QWidget()
        comm_layout = QVBoxLayout()

        # Communication settings group
        comm_group = QGroupBox("Communication Settings")
        comm_form = QFormLayout()

        # Communication model
        self.comm_model_combo = QComboBox()
        self.comm_model_combo.addItems(["simple", "distance-based", "realistic"])
        self.comm_model_combo.currentTextChanged.connect(self.onCommModelChanged)
        comm_form.addRow("Communication Model:", self.comm_model_combo)

        # Communication range control
        range_layout = QHBoxLayout()
        self.comm_range_slider = QSlider(Qt.Orientation.Horizontal)
        self.comm_range_slider.setMinimum(50)
        self.comm_range_slider.setMaximum(500)
        self.comm_range_slider.setValue(Config.MAX_RANGE)
        self.comm_range_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.comm_range_slider.setTickInterval(50)
        range_layout.addWidget(self.comm_range_slider)
        self.comm_range_label = QLabel(f"{Config.MAX_RANGE}m")
        range_layout.addWidget(self.comm_range_label)
        comm_form.addRow("Communication Range:", range_layout)

        # Message frequency
        self.msg_freq_slider = QSlider(Qt.Orientation.Horizontal)
        self.msg_freq_slider.setMinimum(1)
        self.msg_freq_slider.setMaximum(10)
        self.msg_freq_slider.setValue(5)
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(QLabel("Low"))
        freq_layout.addWidget(self.msg_freq_slider)
        freq_layout.addWidget(QLabel("High"))
        comm_form.addRow("Message Frequency:", freq_layout)

        # Packet loss
        self.packet_loss_slider = QSlider(Qt.Orientation.Horizontal)
        self.packet_loss_slider.setMinimum(0)
        self.packet_loss_slider.setMaximum(50)
        self.packet_loss_slider.setValue(5)
        self.packet_loss_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.packet_loss_slider.setTickInterval(5)
        loss_layout = QHBoxLayout()
        loss_layout.addWidget(self.packet_loss_slider)
        self.packet_loss_label = QLabel("5%")
        loss_layout.addWidget(self.packet_loss_label)
        comm_form.addRow("Packet Loss:", loss_layout)

        comm_group.setLayout(comm_form)
        comm_layout.addWidget(comm_group)

        # V2X Settings
        v2x_group = QGroupBox("V2X Protocol Settings")
        v2x_layout = QFormLayout()

        self.v2v_checkbox = QCheckBox("Vehicle-to-Vehicle (V2V)")
        self.v2v_checkbox.setChecked(True)
        v2x_layout.addRow("", self.v2v_checkbox)

        self.v2i_checkbox = QCheckBox("Vehicle-to-Infrastructure (V2I)")
        self.v2i_checkbox.setChecked(True)
        v2x_layout.addRow("", self.v2i_checkbox)

        self.v2p_checkbox = QCheckBox("Vehicle-to-Pedestrian (V2P)")
        self.v2p_checkbox.setChecked(False)
        v2x_layout.addRow("", self.v2p_checkbox)

        v2x_group.setLayout(v2x_layout)
        comm_layout.addWidget(v2x_group)

        comm_tab.setLayout(comm_layout)
        tab_widget.addTab(comm_tab, "Communication")

        # Add tab widget to main layout
        layout.addWidget(tab_widget)

        # Connect signals
        self.speed_slider.valueChanged.connect(self.updateSimulationSpeed)
        self.comm_range_slider.valueChanged.connect(self.updateCommRange)
        self.packet_loss_slider.valueChanged.connect(self.updatePacketLoss)
        self.vehicle_count.valueChanged.connect(self.updateSimulationParams)
        self.show_ranges_checkbox.toggled.connect(self.updateSimulationParams)
        self.show_comm_checkbox.toggled.connect(self.updateSimulationParams)
        self.show_grid_checkbox.toggled.connect(self.updateSimulationParams)
        self.show_stats_checkbox.toggled.connect(self.updateSimulationParams)

        self.setLayout(layout)

    def updateSimulationSpeed(self, value):
        """Update simulation speed label and emit signal"""
        self.speed_label.setText(f"{value}%")
        params = {"sim_speed": value / 100.0}
        self.simulationParamsChanged.emit(params)

    def updateCommRange(self, value):
        """Update communication range label and emit signal"""
        self.comm_range_label.setText(f"{value}m")
        params = {"comm_range": value}
        self.simulationParamsChanged.emit(params)

    def updatePacketLoss(self, value):
        """Update packet loss label and emit signal"""
        self.packet_loss_label.setText(f"{value}%")
        params = {"packet_loss": value / 100.0}
        self.simulationParamsChanged.emit(params)

    def updateSimulationParams(self):
        """Emit signal with updated simulation parameters"""
        params = {
            "vehicle_count": self.vehicle_count.value(),
            "show_ranges": self.show_ranges_checkbox.isChecked(),
            "show_comm": self.show_comm_checkbox.isChecked(),
            "show_grid": self.show_grid_checkbox.isChecked(),
            "show_stats": self.show_stats_checkbox.isChecked(),
            "driver_caution": self.driver_caution.value() / 10.0,
            "v2v_enabled": self.v2v_checkbox.isChecked(),
            "v2i_enabled": self.v2i_checkbox.isChecked(),
            "v2p_enabled": self.v2p_checkbox.isChecked(),
            "min_speed": self.min_speed.value(),
            "max_speed": self.max_speed.value(),
        }
        self.simulationParamsChanged.emit(params)

    def onModeChanged(self, mode, checked):
        """Handle simulation mode change"""
        if checked:
            params = {"mode": mode}
            self.simulationParamsChanged.emit(params)

    def onWeatherChanged(self, weather):
        """Handle weather condition change"""
        params = {"weather": weather}
        self.simulationParamsChanged.emit(params)

    def onCommModelChanged(self, model):
        """Handle communication model change"""
        params = {"comm_model": model}
        self.simulationParamsChanged.emit(params)

    def get_vehicle_type(self):
        """Get selected vehicle type"""
        type_name = self.vehicle_type_combo.currentText()
        for vtype in VehicleType:
            if vtype.value == type_name:
                return vtype
        return VehicleType.CAR

    def get_infra_type(self):
        """Get selected infrastructure type"""
        type_name = self.infra_type_combo.currentText()
        for itype in InfrastructureType:
            if itype.value == type_name:
                return itype
        return InfrastructureType.TRAFFIC_LIGHT


class StatisticsPanel(QWidget):
    """Panel for displaying simulation statistics with Plotly visualizations"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        main_layout = QVBoxLayout()

        # Create tab widget for different statistics views
        tab_widget = QTabWidget()

        # Tab 1: Basic Statistics
        basic_tab = QWidget()
        basic_layout = QVBoxLayout()

        # General statistics group
        stats_group = QGroupBox("Simulation Statistics")
        stats_layout = QFormLayout()

        self.vehicle_count_label = QLabel("0")
        stats_layout.addRow("Vehicle Count:", self.vehicle_count_label)

        self.infra_count_label = QLabel("0")
        stats_layout.addRow("Infrastructure Count:", self.infra_count_label)

        self.messages_sent_label = QLabel("0")
        stats_layout.addRow("Messages Sent:", self.messages_sent_label)

        self.messages_received_label = QLabel("0")
        stats_layout.addRow("Messages Received:", self.messages_received_label)

        self.avg_speed_label = QLabel("0 km/h")
        stats_layout.addRow("Average Speed:", self.avg_speed_label)

        self.collision_count_label = QLabel("0")
        stats_layout.addRow("Collisions:", self.collision_count_label)

        self.sim_time_label = QLabel("0:00")
        stats_layout.addRow("Simulation Time:", self.sim_time_label)

        # Performance indicators
        self.traffic_efficiency_label = QLabel("100%")
        stats_layout.addRow("Traffic Efficiency:", self.traffic_efficiency_label)

        self.comm_reliability_label = QLabel("100%")
        stats_layout.addRow("Comm. Reliability:", self.comm_reliability_label)

        self.network_load_label = QLabel("0%")
        stats_layout.addRow("Network Load:", self.network_load_label)

        self.success_rate_label = QLabel("100%")
        stats_layout.addRow("Msg. Success Rate:", self.success_rate_label)

        stats_group.setLayout(stats_layout)
        basic_layout.addWidget(stats_group)

        # Add performance indicators chart
        self.perf_indicators_chart = PerformanceIndicatorsChart()
        basic_layout.addWidget(self.perf_indicators_chart)

        basic_tab.setLayout(basic_layout)
        tab_widget.addTab(basic_tab, "Overview")

        # Tab 2: Communication Statistics
        comm_tab = QWidget()
        comm_layout = QVBoxLayout()

        # Message type distribution chart
        self.message_chart = MessageDistributionChart()
        comm_layout.addWidget(self.message_chart)

        # Communications table
        comm_stats_group = QGroupBox("Communication Statistics")
        comm_stats_layout = QFormLayout()

        self.msg_types_label = QLabel("None")
        comm_stats_layout.addRow("Top Message Types:", self.msg_types_label)

        self.comm_efficiency_label = QLabel("0%")
        comm_stats_layout.addRow(
            "Communication Efficiency:", self.comm_efficiency_label
        )

        self.packets_lost_label = QLabel("0")
        comm_stats_layout.addRow("Packets Lost:", self.packets_lost_label)

        comm_stats_group.setLayout(comm_stats_layout)
        comm_layout.addWidget(comm_stats_group)

        # Network load chart
        self.network_load_chart = NetworkLoadChart()
        comm_layout.addWidget(self.network_load_chart)

        comm_tab.setLayout(comm_layout)
        tab_widget.addTab(comm_tab, "Communication")

        # Tab 3: Traffic Analysis
        traffic_tab = QWidget()
        traffic_layout = QVBoxLayout()

        # Traffic flow chart
        self.traffic_chart = TrafficFlowChart()
        traffic_layout.addWidget(self.traffic_chart)

        # Vehicle type distribution
        self.vehicle_chart = VehicleDistributionChart()
        traffic_layout.addWidget(self.vehicle_chart)

        traffic_tab.setLayout(traffic_layout)
        tab_widget.addTab(traffic_tab, "Traffic")

        main_layout.addWidget(tab_widget)
        self.setLayout(main_layout)

    def update_statistics(self, stats):
        """Update statistics display with new data"""
        # Update basic statistics
        self.vehicle_count_label.setText(str(stats["vehicle_count"]))
        self.infra_count_label.setText(str(stats["infrastructure_count"]))
        self.messages_sent_label.setText(str(stats["messages_sent"]))
        self.messages_received_label.setText(str(stats["messages_received"]))
        self.avg_speed_label.setText(f"{stats['average_speed']:.1f} km/h")
        self.collision_count_label.setText(str(stats["collisions"]))

        # Format simulation time
        minutes = int(stats["time"] // 60)
        seconds = int(stats["time"] % 60)
        self.sim_time_label.setText(f"{minutes}:{seconds:02d}")

        # Update performance indicators
        self.traffic_efficiency_label.setText(f"{stats['traffic_efficiency']*100:.1f}%")
        self.comm_reliability_label.setText(
            f"{stats['communication_reliability']*100:.1f}%"
        )
        self.network_load_label.setText(f"{stats['network_load']*100:.1f}%")
        self.success_rate_label.setText(f"{stats['message_success_rate']:.1f}%")

        # Calculate communication efficiency
        if stats["messages_sent"] > 0:
            efficiency = (stats["messages_received"] / stats["messages_sent"]) * 100
            self.comm_efficiency_label.setText(f"{efficiency:.1f}%")

            # Calculate packets lost
            packets_lost = stats["messages_sent"] - stats["messages_received"]
            self.packets_lost_label.setText(str(packets_lost))
        else:
            self.comm_efficiency_label.setText("0%")
            self.packets_lost_label.setText("0")

        # Update message distribution chart
        if "message_types" in stats and stats["message_types"]:
            self.message_chart.update_data(stats["message_types"])

            # Update message type text
            msg_types_text = ""
            total = sum(stats["message_types"].values())
            for msg_type, count in sorted(
                stats["message_types"].items(), key=lambda x: x[1], reverse=True
            )[:3]:
                percentage = (count / total) * 100
                msg_types_text += f"{msg_type}: {percentage:.1f}%\n"
            self.msg_types_label.setText(msg_types_text)

        # Update traffic flow chart
        if "history" in stats and stats["history"].get("time_points"):
            self.traffic_chart.update_data(stats["history"])
            self.network_load_chart.update_data(stats["history"])

        # Update vehicle distribution chart
        if "vehicles_by_type" in stats and stats["vehicles_by_type"]:
            self.vehicle_chart.update_data(stats["vehicles_by_type"])

        # Update performance indicators chart
        self.perf_indicators_chart.update_data(stats)


class V2XSimulator(QMainWindow):
    """Main application window for V2X simulation"""

    def __init__(self):
        super().__init__()
        self.initUI()
        self.initSimulation()

    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("Enhanced V2X Simulation Application")
        self.setGeometry(100, 100, 1400, 900)

        # Create central widget with splitter
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Create graphics scene and view
        self.scene = QGraphicsScene()
        self.view = OptimizedGraphicsView(self.scene)
        self.view.scaleChanged.connect(self.onViewScaleChanged)

        # Create control and statistics panels
        right_panel = QSplitter(Qt.Orientation.Vertical)

        self.control_panel = SimulationControlPanel()
        self.control_panel.simulationParamsChanged.connect(
            self.onSimulationParamsChanged
        )
        self.control_panel.play_button.clicked.connect(self.startSimulation)
        self.control_panel.pause_button.clicked.connect(self.pauseSimulation)
        self.control_panel.reset_button.clicked.connect(self.resetSimulation)
        self.control_panel.add_vehicle_button.clicked.connect(self.addRandomVehicle)
        self.control_panel.add_infra_button.clicked.connect(
            self.addRandomInfrastructure
        )
        self.control_panel.emergency_button.clicked.connect(
            self.triggerEmergencyVehicle
        )
        self.control_panel.brake_test_button.clicked.connect(self.runBrakeTest)
        self.control_panel.platooning_button.clicked.connect(self.formVehiclePlatoon)
        self.control_panel.save_button.clicked.connect(self.saveSimulation)
        self.control_panel.load_button.clicked.connect(self.loadSimulation)

        self.stats_panel = StatisticsPanel()

        # Add widgets to the right panel
        right_panel.addWidget(self.control_panel)
        right_panel.addWidget(self.stats_panel)

        # Set size hints
        right_panel.setSizes([400, 600])

        # Add widgets to main layout
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.addWidget(self.view)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([1000, 400])

        main_layout.addWidget(main_splitter)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Simulation ready. Press Play to start.")

        # Create toolbar for quick actions
        self.createToolbar()

        # Create simulation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.updateSimulation)
        self.update_interval = Config.UPDATE_INTERVAL
        self.delta_time = self.update_interval / 1000.0  # seconds
        self.simulation_speed = 1.0

        # For measuring performance
        self.fps_timer = QElapsedTimer()
        self.fps_timer.start()
        self.frame_times = deque(maxlen=60)

        # Active communication visualizations
        self.active_message_animations = []

        # Current view scale
        self.current_view_scale = 1.0

        # Show grid
        self.grid_visible = True
        self.grid_item = None

    def createToolbar(self):
        """Create toolbar with quick access buttons"""
        toolbar = QToolBar("Simulation Toolbar")
        self.addToolBar(toolbar)

        # Create actions
        play_action = QAction(QIcon.fromTheme("media-playback-start"), "Play", self)
        play_action.triggered.connect(self.startSimulation)
        play_action.setShortcut(QKeySequence("F5"))

        pause_action = QAction(QIcon.fromTheme("media-playback-pause"), "Pause", self)
        pause_action.triggered.connect(self.pauseSimulation)
        pause_action.setShortcut(QKeySequence("F6"))

        reset_action = QAction(QIcon.fromTheme("edit-undo"), "Reset", self)
        reset_action.triggered.connect(self.resetSimulation)
        reset_action.setShortcut(QKeySequence("F7"))

        add_vehicle_action = QAction(QIcon.fromTheme("list-add"), "Add Vehicle", self)
        add_vehicle_action.triggered.connect(self.addRandomVehicle)

        add_infra_action = QAction(
            QIcon.fromTheme("list-add"), "Add Infrastructure", self
        )
        add_infra_action.triggered.connect(self.addRandomInfrastructure)

        zoom_in_action = QAction(QIcon.fromTheme("zoom-in"), "Zoom In", self)
        zoom_in_action.triggered.connect(lambda: self.view.zoom(self.view.zoom_factor))
        zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)

        zoom_out_action = QAction(QIcon.fromTheme("zoom-out"), "Zoom Out", self)
        zoom_out_action.triggered.connect(
            lambda: self.view.zoom(1 / self.view.zoom_factor)
        )
        zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)

        fit_view_action = QAction(QIcon.fromTheme("zoom-fit-best"), "Fit View", self)
        fit_view_action.triggered.connect(
            lambda: self.view.fitInView(
                self.scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
        )
        fit_view_action.setShortcut(QKeySequence("Ctrl+0"))

        # Add actions to toolbar
        toolbar.addAction(play_action)
        toolbar.addAction(pause_action)
        toolbar.addAction(reset_action)
        toolbar.addSeparator()
        toolbar.addAction(add_vehicle_action)
        toolbar.addAction(add_infra_action)
        toolbar.addSeparator()
        toolbar.addAction(zoom_in_action)
        toolbar.addAction(zoom_out_action)
        toolbar.addAction(fit_view_action)

    def initSimulation(self):
        """Initialize the simulation environment"""
        # Create event manager
        self.event_manager = EventManager()

        # Create simulation environment with event manager
        self.simulation = SimulationEnvironment(self.event_manager)

        # Connect event signals
        self.event_manager.statistics_updated.connect(self.onStatisticsUpdated)
        self.event_manager.collision_detected.connect(self.onCollisionDetected)
        self.event_manager.simulation_reset.connect(self.onSimulationReset)

        # Create background grid
        self.createGrid()

        # Create initial road network
        self.createDefaultRoadNetwork()

        # Add initial vehicles
        for _ in range(20):
            self.addRandomVehicle()

        # Add initial infrastructure
        for _ in range(5):
            self.addRandomInfrastructure()

    def createGrid(self):
        """Create background grid"""
        if self.grid_item:
            self.scene.removeItem(self.grid_item)

        grid_rect = QRectF(-2000, -2000, 4000, 4000)
        self.grid_item = GridGraphicsItem(grid_rect)
        self.scene.addItem(self.grid_item)
        self.grid_item.setVisible(self.grid_visible)

    def createDefaultRoadNetwork(self):
        """Create a default road network for the simulation"""
        # Create a grid road network
        grid_size = 500
        center_x = 0
        center_y = 0

        # Horizontal roads
        for i in range(-2, 3):
            y = center_y + i * grid_size
            road = Road(
                center_x - 2 * grid_size,
                y,
                center_x + 2 * grid_size,
                y,
                lanes=2,
                bidirectional=True,
                name=f"H Road {i+3}",
            )
            self.simulation.add_road(road)
            self.scene.addItem(RoadGraphicsItem(road))

        # Vertical roads
        for i in range(-2, 3):
            x = center_x + i * grid_size
            road = Road(
                x,
                center_y - 2 * grid_size,
                x,
                center_y + 2 * grid_size,
                lanes=2,
                bidirectional=True,
                name=f"V Road {i+3}",
            )
            self.simulation.add_road(road)
            self.scene.addItem(RoadGraphicsItem(road))

        # Add a few diagonal roads for variety
        road = Road(
            center_x - grid_size,
            center_y - grid_size,
            center_x + grid_size,
            center_y + grid_size,
            lanes=2,
            bidirectional=True,
            name="Diagonal 1",
        )
        self.simulation.add_road(road)
        self.scene.addItem(RoadGraphicsItem(road))

        road = Road(
            center_x - grid_size,
            center_y + grid_size,
            center_x + grid_size,
            center_y - grid_size,
            lanes=2,
            bidirectional=True,
            name="Diagonal 2",
        )
        self.simulation.add_road(road)
        self.scene.addItem(RoadGraphicsItem(road))

        # Set view to show the entire road network
        self.view.setSceneRect(
            QRectF(
                center_x - 2 * grid_size - 100,
                center_y - 2 * grid_size - 100,
                4 * grid_size + 200,
                4 * grid_size + 200,
            )
        )
        self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def addRandomVehicle(self):
        """Add a random vehicle to the simulation"""
        if not self.simulation.roads:
            return

        # Choose a random road to place the vehicle on
        road = random.choice(self.simulation.roads)

        # Choose a random position along the road
        distance = random.uniform(0, road.length)
        lane = random.randint(0, road.lanes - 1)
        direction = 1 if random.random() > 0.5 or not road.bidirectional else -1

        # Get position
        pos = road.get_lane_position(distance, lane, direction)

        # Get speed range from control panel
        min_speed = self.control_panel.min_speed.value()
        max_speed = self.control_panel.max_speed.value()

        # Create vehicle with selected type
        vehicle_type = self.control_panel.get_vehicle_type()
        vehicle = Vehicle(
            pos.x(),
            pos.y(),
            vehicle_type,
            speed=random.uniform(min_speed, max_speed),
            direction=road.angle if direction == 1 else road.angle + math.pi,
        )

        # Adjust driver behavior based on control panel
        caution_level = self.control_panel.driver_caution.value() / 10.0
        vehicle.driver_behavior["caution"] = caution_level
        vehicle.driver_behavior["aggression"] = 1.0 - caution_level

        # Add to simulation
        self.simulation.add_vehicle(vehicle)

        # Create visual representation
        vehicle_item = VehicleGraphicsItem(vehicle)
        self.scene.addItem(vehicle_item)

        # Generate a route for the vehicle
        self.generateVehicleRoute(vehicle)

        return vehicle

    def addRandomInfrastructure(self):
        """Add a random infrastructure element to the simulation"""
        if not self.simulation.roads:
            return

        # Choose a random road to place infrastructure near
        road = random.choice(self.simulation.roads)

        # Choose a random position along the road
        distance = random.uniform(0, road.length)
        side = random.choice([-1, 1])

        # Calculate position (offset from the road)
        base_pos = road.get_lane_position(distance, 0, 1)
        perp_angle = road.angle + math.pi / 2
        offset = road.width / 2 + 20  # 20 pixels away from the road edge

        x = base_pos.x() + side * offset * math.cos(perp_angle)
        y = base_pos.y() + side * offset * math.sin(perp_angle)

        # Create infrastructure with selected type
        infra_type = self.control_panel.get_infra_type()
        infrastructure = Infrastructure(x, y, infra_type)

        # Add to simulation
        self.simulation.add_infrastructure(infrastructure)

        # Create visual representation
        infra_item = InfrastructureGraphicsItem(infrastructure)
        self.scene.addItem(infra_item)

        return infrastructure

    def generateVehicleRoute(self, vehicle):
        """Generate a random route for a vehicle through the road network"""
        if not self.simulation.roads:
            return

        # Create a route: go to 3-5 random points
        route_length = random.randint(3, 5)
        route = []

        # Pick random roads for waypoints
        roads = random.sample(
            self.simulation.roads, min(route_length, len(self.simulation.roads))
        )

        for road in roads:
            distance = random.uniform(0, road.length)
            lane = random.randint(0, road.lanes - 1)
            direction = random.choice([1, -1]) if road.bidirectional else 1

            pos = road.get_lane_position(distance, lane, direction)
            route.append(pos)

        vehicle.set_route(route)

    def startSimulation(self):
        """Start the simulation"""
        self.timer.start(self.update_interval)
        self.statusBar.showMessage("Simulation running")

    def pauseSimulation(self):
        """Pause the simulation"""
        self.timer.stop()
        self.statusBar.showMessage("Simulation paused")

    def resetSimulation(self):
        """Reset the simulation to initial state"""
        self.timer.stop()

        # Clear the scene
        self.scene.clear()

        # Reset simulation object ID counter
        SimulationObject._next_id = 1

        # Create grid
        self.createGrid()

        # Reinitialize the simulation
        self.initSimulation()

        # Notify listeners
        if self.event_manager:
            self.event_manager.simulation_reset.emit()

        self.statusBar.showMessage("Simulation reset")

    def updateSimulation(self):
        """Update the simulation for one time step"""
        # Measure frame time
        frame_start = self.fps_timer.elapsed()

        # Update the simulation with adjusted time step based on simulation speed
        self.simulation.update(self.delta_time * self.simulation_speed)

        # Update all vehicle graphics
        for vehicle in self.simulation.vehicles:
            if vehicle.item:
                vehicle.item.update_graphics(self.current_view_scale)

        # Update all infrastructure graphics
        for infra in self.simulation.infrastructure:
            if infra.item:
                infra.item.update_graphics(self.current_view_scale)

        # Update all road graphics
        for road in self.simulation.roads:
            if road.item:
                road.item.update_graphics(self.current_view_scale)

        # Visualize communications if enabled
        if self.control_panel.show_comm_checkbox.isChecked():
            self.visualizeCommunications()

        # Update message animations
        self.updateMessageAnimations()

        # Update frame time measurement
        frame_end = self.fps_timer.elapsed()
        frame_time = frame_end - frame_start
        self.frame_times.append(frame_time)

        # Calculate FPS
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1000 / avg_frame_time if avg_frame_time > 0 else 0
            self.statusBar.showMessage(f"Simulation running - FPS: {fps:.1f}")

    def visualizeCommunications(self):
        """Visualize communication between entities"""
        # Process recent messages in the simulation's message history
        recent_messages = list(self.simulation.message_history)[
            -10:
        ]  # Only show most recent messages

        # Limit number of active animations for performance
        if len(self.active_message_animations) >= Config.ANIMATION_LIMIT:
            return

        # Check if we should visualize this particular message (random sampling)
        if not recent_messages or random.random() > Config.COMM_VISUAL_CHANCE:
            return

        # Pick a random recent message to visualize
        message = random.choice(recent_messages)
        sender = message.sender

        # Find receivers in range
        receivers = []
        for entity in self.simulation.vehicles + self.simulation.infrastructure:
            if entity != sender:
                if distance(sender.position(), entity.position()) <= sender.comm_range:
                    receivers.append(entity)

        # Create communication visualization
        if receivers:
            comm_item = CommunicationVisualizerItem(sender, receivers, message.type)
            self.scene.addItem(comm_item)
            self.active_message_animations.append(comm_item)

    def updateMessageAnimations(self):
        """Update all active message animations"""
        # Update animations and remove expired ones
        i = 0
        while i < len(self.active_message_animations):
            anim_item = self.active_message_animations[i]

            # Update the animation
            expired = anim_item.update_animation()

            if expired:
                # Remove the item
                self.scene.removeItem(anim_item)

                # Remove from active animations
                self.active_message_animations.pop(i)
            else:
                i += 1

    def onViewScaleChanged(self, scale):
        """Handle view scale change"""
        self.current_view_scale = scale

        # Update all graphics with new scale factor
        for vehicle in self.simulation.vehicles:
            if vehicle.item:
                vehicle.item.update_graphics(scale)

        for infra in self.simulation.infrastructure:
            if infra.item:
                infra.item.update_graphics(scale)

        for road in self.simulation.roads:
            if road.item:
                road.item.update_graphics(scale)

    def onSimulationParamsChanged(self, params):
        """Handle changes to simulation parameters"""
        if "sim_speed" in params:
            self.simulation_speed = params["sim_speed"]

        if "comm_range" in params:
            # Update communication range for all entities
            new_range = params["comm_range"]
            for vehicle in self.simulation.vehicles:
                vehicle.comm_range = new_range
                # Update visualization if range circles are visible
                if vehicle.item and vehicle.item.range_circle:
                    vehicle.item.range_circle.update_range(new_range)

            for infra in self.simulation.infrastructure:
                infra.comm_range = new_range
                # Update visualization if range circles are visible
                if infra.item and infra.item.range_circle:
                    infra.item.range_circle.update_range(new_range)

        if "show_ranges" in params:
            if params["show_ranges"]:
                self.showCommunicationRanges()
            else:
                self.hideCommunicationRanges()

        if "show_grid" in params:
            self.grid_visible = params["show_grid"]
            if self.grid_item:
                self.grid_item.setVisible(self.grid_visible)

        if "packet_loss" in params:
            # Apply packet loss to communication model
            pass  # Would be implemented in realistic communication model

        if "mode" in params:
            # Change simulation mode
            self.simulation.set_mode(params["mode"])

        if "weather" in params:
            # Change weather conditions
            self.simulation.set_weather(params["weather"])

        if "comm_model" in params:
            # Change communication model
            self.simulation.communication_model = params["comm_model"]

    def onStatisticsUpdated(self, stats):
        """Handle statistics update event"""
        # Update statistics panel
        self.stats_panel.update_statistics(stats)

    def onCollisionDetected(self, vehicle1, vehicle2):
        """Handle collision between vehicles"""
        # Visual effect for collision
        position = QPointF((vehicle1.x + vehicle2.x) / 2, (vehicle1.y + vehicle2.y) / 2)

        # Create a simple collision effect
        collision_item = QGraphicsEllipseItem(-20, -20, 40, 40)
        collision_item.setBrush(QBrush(QColor(255, 0, 0, 150)))
        collision_item.setPen(QPen(Qt.GlobalColor.red, 2))
        collision_item.setPos(position)
        collision_item.setZValue(30)  # Above everything
        self.scene.addItem(collision_item)

        # Create fade-out animation
        def remove_item():
            if collision_item in self.scene.items():
                self.scene.removeItem(collision_item)

        # Remove after 1 second
        QTimer.singleShot(1000, remove_item)

        # Status message
        self.statusBar.showMessage(
            f"Collision detected between vehicles {vehicle1.id} and {vehicle2.id}"
        )

    def onSimulationReset(self):
        """Handle simulation reset event"""
        # Clear active animations
        for anim in self.active_message_animations:
            self.scene.removeItem(anim)
        self.active_message_animations.clear()

    def showCommunicationRanges(self):
        """Show communication range circles for all entities"""
        for vehicle in self.simulation.vehicles:
            if vehicle.item and not vehicle.item.range_circle:
                range_circle = CommunicationRangeCircle(vehicle)
                self.scene.addItem(range_circle)
                vehicle.item.range_circle = range_circle
                vehicle.item.show_range = True

        for infra in self.simulation.infrastructure:
            if infra.item and not infra.item.range_circle:
                range_circle = CommunicationRangeCircle(infra)
                self.scene.addItem(range_circle)
                infra.item.range_circle = range_circle
                infra.item.show_range = True

    def hideCommunicationRanges(self):
        """Hide communication range circles for all entities"""
        for vehicle in self.simulation.vehicles:
            if vehicle.item and vehicle.item.range_circle:
                self.scene.removeItem(vehicle.item.range_circle)
                vehicle.item.range_circle = None
                vehicle.item.show_range = False

        for infra in self.simulation.infrastructure:
            if infra.item and infra.item.range_circle:
                self.scene.removeItem(infra.item.range_circle)
                infra.item.range_circle = None
                infra.item.show_range = False

    def triggerEmergencyVehicle(self):
        """Create an emergency vehicle scenario"""
        # Find a good road for emergency vehicle
        if not self.simulation.roads:
            return

        # Choose a long road
        suitable_roads = sorted(
            self.simulation.roads, key=lambda r: r.length, reverse=True
        )
        road = (
            suitable_roads[0]
            if suitable_roads
            else random.choice(self.simulation.roads)
        )

        # Place at the start of the road
        pos = road.get_lane_position(0, 0, 1)

        # Create emergency vehicle
        vehicle = Vehicle(
            pos.x(), pos.y(), VehicleType.EMERGENCY, speed=120, direction=road.angle
        )

        # Create destination at other end of road
        dest_pos = road.get_lane_position(road.length, 0, 1)
        vehicle.set_route([dest_pos])

        # Add to simulation
        self.simulation.add_vehicle(vehicle)

        # Create visual representation
        vehicle_item = VehicleGraphicsItem(vehicle)
        self.scene.addItem(vehicle_item)

        # Switch to emergency response mode
        self.simulation.set_mode(SimulationMode.EMERGENCY_RESPONSE)

        # Update UI
        for mode, button in self.control_panel.mode_buttons.items():
            if mode == SimulationMode.EMERGENCY_RESPONSE:
                button.setChecked(True)

        self.statusBar.showMessage("Emergency vehicle dispatched!")

    def runBrakeTest(self):
        """Run a brake test scenario"""
        # Find a good road for brake test
        if not self.simulation.roads:
            return

        road = random.choice(self.simulation.roads)

        # Create a vehicle in front that will brake
        front_pos = road.get_lane_position(road.length / 2, 0, 1)
        front_vehicle = Vehicle(
            front_pos.x(),
            front_pos.y(),
            VehicleType.CAR,
            speed=80,
            direction=road.angle,
        )
        self.simulation.add_vehicle(front_vehicle)
        self.scene.addItem(VehicleGraphicsItem(front_vehicle))

        # Create a following vehicle
        back_pos = road.get_lane_position(road.length / 2 - 100, 0, 1)
        back_vehicle = Vehicle(
            back_pos.x(),
            back_pos.y(),
            VehicleType.AUTONOMOUS,
            speed=100,
            direction=road.angle,
        )
        self.simulation.add_vehicle(back_vehicle)
        self.scene.addItem(VehicleGraphicsItem(back_vehicle))

        # Schedule the brake event
        def apply_brake():
            if front_vehicle in self.simulation.vehicles:
                front_vehicle.apply_brake(0.8)
                self.statusBar.showMessage("Brake test initiated!")

        QTimer.singleShot(2000, apply_brake)

        self.statusBar.showMessage("Brake test scenario set up...")

    def formVehiclePlatoon(self):
        """Form a platoon of vehicles"""
        # Find a good road for platoon
        if not self.simulation.roads:
            return

        road = random.choice(self.simulation.roads)

        # Lead vehicle is autonomous
        lead_pos = road.get_lane_position(100, 0, 1)
        lead_vehicle = Vehicle(
            lead_pos.x(),
            lead_pos.y(),
            VehicleType.AUTONOMOUS,
            speed=80,
            direction=road.angle,
        )
        self.simulation.add_vehicle(lead_vehicle)
        self.scene.addItem(VehicleGraphicsItem(lead_vehicle))

        # Add following vehicles
        platoon_size = 5
        spacing = 30  # Distance between vehicles

        for i in range(1, platoon_size):
            pos = road.get_lane_position(100 - i * spacing, 0, 1)
            vehicle = Vehicle(
                pos.x(), pos.y(), VehicleType.AUTONOMOUS, speed=80, direction=road.angle
            )
            # Link to lead vehicle's target speed
            vehicle.target_speed = lead_vehicle.target_speed
            self.simulation.add_vehicle(vehicle)
            self.scene.addItem(VehicleGraphicsItem(vehicle))

        # Create a destination route for the platoon leader
        end_pos = road.get_lane_position(road.length, 0, 1)
        lead_vehicle.set_route([end_pos])

        self.statusBar.showMessage(
            f"Created vehicle platoon with {platoon_size} vehicles"
        )

    def saveSimulation(self):
        """Save the current simulation state"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Simulation", "", "Simulation Files (*.v2x);;All Files (*)"
        )
        if filename:
            try:
                # Get simulation state
                state = self.simulation.get_state_dict()

                # Save to file
                with open(filename, "w") as f:
                    json.dump(state, f, indent=2)

                self.statusBar.showMessage(f"Simulation saved to {filename}")
            except Exception as e:
                self.statusBar.showMessage(f"Error saving simulation: {str(e)}")

    def loadSimulation(self):
        """Load a saved simulation state"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Load Simulation", "", "Simulation Files (*.v2x);;All Files (*)"
        )
        if filename:
            try:
                # Load from file
                with open(filename, "r") as f:
                    state = json.load(f)

                # Reset simulation
                self.resetSimulation()

                # TODO: Implement reconstruction of simulation from saved state

                self.statusBar.showMessage(f"Simulation loaded from {filename}")
            except Exception as e:
                self.statusBar.showMessage(f"Error loading simulation: {str(e)}")

    def closeEvent(self, event):
        """Handle window close event"""
        self.timer.stop()
        super().closeEvent(event)


# Main application execution
def main():
    app = QApplication(sys.argv)
    simulator = V2XSimulator()
    simulator.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
