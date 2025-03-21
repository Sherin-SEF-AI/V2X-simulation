#!/usr/bin/env python3
# V2X Simulation Application

import sys
import random
import math
import time
from collections import deque
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
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
)
from PyQt6.QtGui import QPen, QBrush, QColor, QPainterPath, QPainter, QFont, QIcon
from PyQt6.QtCore import Qt, QTimer, QPointF, QRectF, QLineF, pyqtSignal, QObject

# Constants
ROAD_WIDTH = 20
VEHICLE_SIZE = 16
INFRASTRUCTURE_SIZE = 24
MAX_SPEED = 150  # km/h
MAX_RANGE = 300  # meters (communication range)
ROAD_COLOR = QColor(50, 50, 50)
BACKGROUND_COLOR = QColor(230, 230, 230)
UPDATE_INTERVAL = 50  # milliseconds (20 fps)

# Vehicle types and colors
VEHICLE_TYPES = {
    "Car": QColor(30, 144, 255),
    "Truck": QColor(0, 128, 0),
    "Bus": QColor(255, 140, 0),
    "Emergency": QColor(255, 0, 0),
    "Autonomous": QColor(138, 43, 226),
}

# Infrastructure types and colors
INFRASTRUCTURE_TYPES = {
    "Traffic Light": QColor(255, 215, 0),
    "Road Sign": QColor(255, 192, 203),
    "Base Station": QColor(0, 255, 255),
    "Sensor": QColor(255, 105, 180),
}

# Communication types
COMMUNICATION_TYPES = [
    "Basic Safety Message",
    "Signal Phase and Timing",
    "Map Data",
    "Emergency Vehicle Alert",
    "Traveler Information",
    "Roadside Alert",
]


# Utility Functions
def distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return math.sqrt((p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2)


def angle_between_points(p1, p2):
    """Calculate angle in radians between two points"""
    return math.atan2(p2.y() - p1.y(), p2.x() - p1.x())


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
        self.item = None  # Reference to graphics item

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


class Vehicle(SimulationObject):
    """Represents a vehicle in the simulation"""

    def __init__(self, x, y, vehicle_type, speed=0, direction=0):
        super().__init__(x, y, vehicle_type)
        self.speed = speed  # km/h
        self.direction = direction  # radians
        self.size = VEHICLE_SIZE
        self.comm_range = MAX_RANGE
        self.is_braking = False
        self.is_autonomous = vehicle_type == "Autonomous"
        self.route = []  # List of waypoints
        self.current_waypoint = None
        self.target_speed = speed

    def update(self, delta_time):
        """Update vehicle position based on speed and direction"""
        # Convert km/h to m/s, then to pixels per time step
        speed_ms = self.speed * 0.277778  # km/h to m/s
        distance_per_step = (
            speed_ms * delta_time * 0.05
        )  # Scale factor for visualization

        # Gradually adjust speed toward target speed
        if abs(self.speed - self.target_speed) > 1.0:
            if self.speed < self.target_speed:
                self.speed += min(5.0, self.target_speed - self.speed)  # Accelerate
            else:
                self.speed -= min(
                    8.0, self.speed - self.target_speed
                )  # Decelerate faster
        else:
            self.speed = self.target_speed  # Reached target

        # Move vehicle
        self.x += math.cos(self.direction) * distance_per_step
        self.y += math.sin(self.direction) * distance_per_step

        # Update graphics position
        self.update_graphics_position()

        # Check if we reached a waypoint
        if (
            self.current_waypoint
            and distance(self.position(), self.current_waypoint) < 10
        ):
            if self.route:
                self.current_waypoint = self.route.pop(0)
                new_direction = angle_between_points(
                    self.position(), self.current_waypoint
                )

                # Smooth direction change
                angle_diff = (new_direction - self.direction + math.pi) % (
                    2 * math.pi
                ) - math.pi
                if abs(angle_diff) > 0.5:  # If turning significantly, slow down
                    self.target_speed = max(20, self.target_speed * 0.7)
                else:
                    self.target_speed = min(MAX_SPEED, self.target_speed * 1.2)

                self.direction = new_direction
            else:
                self.current_waypoint = None

        # Check for boundary conditions - wrap around screen
        boundary = 1500  # Screen boundary
        if self.x < -boundary:
            self.x = boundary
        if self.x > boundary:
            self.x = -boundary
        if self.y < -boundary:
            self.y = boundary
        if self.y > boundary:
            self.y = -boundary

    def apply_brake(self, intensity=0.3):
        """Apply brakes to slow down the vehicle"""
        self.target_speed *= 1 - intensity
        self.is_braking = True

    def accelerate(self, amount=5):
        """Increase vehicle speed"""
        self.target_speed = min(self.target_speed + amount, MAX_SPEED)
        self.is_braking = False

    def set_route(self, waypoints):
        """Set a route for the vehicle to follow"""
        self.route = waypoints.copy()
        if self.route:
            self.current_waypoint = self.route.pop(0)
            self.direction = angle_between_points(
                self.position(), self.current_waypoint
            )


class Infrastructure(SimulationObject):
    """Represents infrastructure elements like traffic lights, road signs, etc."""

    def __init__(self, x, y, infra_type, state="idle"):
        super().__init__(x, y, infra_type)
        self.state = state
        self.size = INFRASTRUCTURE_SIZE
        self.comm_range = MAX_RANGE
        self.connected_vehicles = set()

        # Traffic light specific properties
        self.cycle_times = {"red": 30, "yellow": 5, "green": 30}
        self.current_phase = "red"
        self.phase_time = 0

        # Initialize state based on type
        if infra_type == "Traffic Light":
            phases = ["red", "green", "yellow"]
            self.current_phase = random.choice(phases)
            self.phase_time = random.uniform(0, self.cycle_times[self.current_phase])

    def update(self, delta_time):
        """Update infrastructure state"""
        if self.type == "Traffic Light":
            self.phase_time += delta_time

            # Check if we need to change the traffic light phase
            if (
                self.current_phase == "red"
                and self.phase_time >= self.cycle_times["red"]
            ):
                self.current_phase = "green"
                self.phase_time = 0
            elif (
                self.current_phase == "green"
                and self.phase_time >= self.cycle_times["green"]
            ):
                self.current_phase = "yellow"
                self.phase_time = 0
            elif (
                self.current_phase == "yellow"
                and self.phase_time >= self.cycle_times["yellow"]
            ):
                self.current_phase = "red"
                self.phase_time = 0

            # Update color of traffic light in visualization
            if self.item:
                if self.current_phase == "red":
                    self.item.setBrush(QBrush(QColor(255, 0, 0)))
                elif self.current_phase == "yellow":
                    self.item.setBrush(QBrush(QColor(255, 255, 0)))
                elif self.current_phase == "green":
                    self.item.setBrush(QBrush(QColor(0, 255, 0)))

        # Road signs might update state based on nearby vehicles
        elif self.type == "Road Sign":
            nearby_count = 0
            for vehicle in self.connected_vehicles:
                if distance(self.position(), vehicle.position()) <= self.comm_range:
                    nearby_count += 1

            # Update display based on vehicle count
            if nearby_count > 3:
                self.state = "congested"
            elif nearby_count > 0:
                self.state = "active"
            else:
                self.state = "idle"


class Road:
    """Represents a road segment in the simulation"""

    def __init__(self, start_x, start_y, end_x, end_y, lanes=2, bidirectional=True):
        self.start = QPointF(start_x, start_y)
        self.end = QPointF(end_x, end_y)
        self.lanes = lanes
        self.bidirectional = bidirectional
        self.length = distance(self.start, self.end)
        self.angle = angle_between_points(self.start, self.end)
        self.width = ROAD_WIDTH * lanes * (2 if bidirectional else 1)
        self.item = None  # Reference to graphics item

    def get_lane_position(self, distance_along, lane=0, direction=1):
        """Get position at specified distance along the road in a specific lane"""
        lane_offset = (lane + 0.5) * ROAD_WIDTH
        if direction == -1 and self.bidirectional:
            lane_offset += self.lanes * ROAD_WIDTH

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


class Message:
    """Represents a V2X message"""

    def __init__(self, sender, message_type, content=None):
        self.sender = sender
        self.type = message_type
        self.content = content or {}
        self.timestamp = time.time()

    def __str__(self):
        return f"Message({self.type}) from {self.sender.type} {self.sender.id}"


class SimulationEnvironment:
    """Manages the entire simulation environment"""

    def __init__(self):
        self.vehicles = []
        self.infrastructure = []
        self.roads = []
        self.messages = []
        self.active_message_graphics = []
        self.time = 0
        self.statistics = {
            "messages_sent": 0,
            "messages_received": 0,
            "vehicle_count": 0,
            "infrastructure_count": 0,
            "average_speed": 0,
            "collisions": 0,
            "message_types": {},
            "time": 0,
        }

        # Message history - for keeping a limited buffer
        self.message_history = deque(maxlen=100)

    def add_vehicle(self, vehicle):
        """Add a vehicle to the simulation"""
        self.vehicles.append(vehicle)
        self.statistics["vehicle_count"] += 1
        return vehicle

    def add_infrastructure(self, infrastructure):
        """Add an infrastructure element to the simulation"""
        self.infrastructure.append(infrastructure)
        self.statistics["infrastructure_count"] += 1
        return infrastructure

    def add_road(self, road):
        """Add a road to the simulation"""
        self.roads.append(road)
        return road

    def update(self, delta_time):
        """Update the entire simulation for one time step"""
        self.time += delta_time

        # Update all vehicles
        for vehicle in self.vehicles:
            vehicle.update(delta_time)

        # Update all infrastructure
        for infra in self.infrastructure:
            infra.update(delta_time)

        # Simulate V2X communication
        self.simulate_communication()

        # Update statistics
        self.update_statistics()

        return self.active_message_graphics

    def simulate_communication(self):
        """Simulate V2X communication between entities"""
        # Different types of entities send different message types with different frequencies
        for vehicle in self.vehicles:
            # Each entity has a chance to send a message, weighted by type
            if random.random() < 0.1:  # 10% chance per update
                # Autonomous vehicles communicate more frequently
                if vehicle.is_autonomous:
                    message_type = "Basic Safety Message"
                elif vehicle.is_braking:
                    message_type = "Emergency Vehicle Alert"
                else:
                    message_type = random.choice(
                        ["Basic Safety Message", "Traveler Information"]
                    )

                self.transmit_message(vehicle, message_type)

        # Infrastructure communications
        for infra in self.infrastructure:
            if random.random() < 0.15:  # 15% chance per update
                if infra.type == "Traffic Light":
                    message_type = "Signal Phase and Timing"
                elif infra.type == "Road Sign":
                    message_type = "Roadside Alert"
                elif infra.type == "Base Station":
                    message_type = "Map Data"
                else:
                    message_type = random.choice(COMMUNICATION_TYPES)

                self.transmit_message(infra, message_type)

    def transmit_message(self, sender, message_type):
        """Transmit a message from sender to all receivers in range"""
        message = Message(sender, message_type)
        self.message_history.append(message)
        self.statistics["messages_sent"] += 1

        # Track message types for statistics
        if message_type in self.statistics["message_types"]:
            self.statistics["message_types"][message_type] += 1
        else:
            self.statistics["message_types"][message_type] = 1

        # Find all receivers in range
        receivers = []
        sender_pos = sender.position()

        for receiver in self.vehicles + self.infrastructure:
            if receiver != sender:
                if distance(sender_pos, receiver.position()) <= sender.comm_range:
                    receivers.append(receiver)
                    receiver.receive_message(message)
                    self.statistics["messages_received"] += 1

        return receivers

    def update_statistics(self):
        """Update simulation statistics"""
        if self.vehicles:
            self.statistics["average_speed"] = sum(
                v.speed for v in self.vehicles
            ) / len(self.vehicles)
        else:
            self.statistics["average_speed"] = 0

        # Check for collisions (simplified)
        for i, v1 in enumerate(self.vehicles):
            for v2 in self.vehicles[i + 1 :]:
                if distance(v1.position(), v2.position()) < (v1.size + v2.size) / 2:
                    self.statistics["collisions"] += 1
                    # Simulate collision response
                    v1.speed = 0
                    v2.speed = 0
                    v1.target_speed = 0
                    v2.target_speed = 0

        # Update time in statistics
        self.statistics["time"] = self.time


# Graphics Classes
class VehicleGraphicsItem(QGraphicsEllipseItem):
    """Visual representation of a vehicle"""

    def __init__(self, vehicle):
        super().__init__(
            -vehicle.size / 2, -vehicle.size / 2, vehicle.size, vehicle.size
        )
        self.vehicle = vehicle
        self.vehicle.item = self

        # Set visual appearance
        self.setBrush(QBrush(VEHICLE_TYPES.get(vehicle.type, QColor(100, 100, 100))))
        self.setPen(QPen(Qt.GlobalColor.black, 1))

        # Show vehicle direction
        self.direction_line = QGraphicsLineItem(
            0,
            0,
            vehicle.size / 2 * math.cos(vehicle.direction),
            vehicle.size / 2 * math.sin(vehicle.direction),
            parent=self,
        )
        self.direction_line.setPen(QPen(Qt.GlobalColor.white, 2))

        # Position in the scene
        self.setPos(vehicle.x, vehicle.y)

        # For showing communication range
        self.range_circle = None

        # Add small label with id
        self.id_label = QGraphicsSimpleTextItem(str(vehicle.id), parent=self)
        self.id_label.setPos(-3, -3)
        self.id_label.setBrush(QBrush(Qt.GlobalColor.white))

    def update_graphics(self):
        """Update graphics based on vehicle state"""
        self.setPos(self.vehicle.x, self.vehicle.y)

        # Update direction indicator
        self.direction_line.setLine(
            0,
            0,
            self.vehicle.size / 2 * math.cos(self.vehicle.direction),
            self.vehicle.size / 2 * math.sin(self.vehicle.direction),
        )

        # Show braking state
        if self.vehicle.is_braking:
            self.setBrush(QBrush(QColor(255, 0, 0)))
        else:
            self.setBrush(
                QBrush(VEHICLE_TYPES.get(self.vehicle.type, QColor(100, 100, 100)))
            )


class InfrastructureGraphicsItem(QGraphicsRectItem):
    """Visual representation of infrastructure elements"""

    def __init__(self, infrastructure):
        size = infrastructure.size
        super().__init__(-size / 2, -size / 2, size, size)
        self.infrastructure = infrastructure
        self.infrastructure.item = self

        # Set visual appearance
        color = INFRASTRUCTURE_TYPES.get(infrastructure.type, QColor(150, 150, 150))
        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.GlobalColor.black, 1))

        # For traffic lights, update based on phase
        if infrastructure.type == "Traffic Light":
            if infrastructure.current_phase == "red":
                self.setBrush(QBrush(QColor(255, 0, 0)))
            elif infrastructure.current_phase == "yellow":
                self.setBrush(QBrush(QColor(255, 255, 0)))
            elif infrastructure.current_phase == "green":
                self.setBrush(QBrush(QColor(0, 255, 0)))

        # Position in the scene
        self.setPos(infrastructure.x, infrastructure.y)

        # Add text label
        self.label = QGraphicsSimpleTextItem(infrastructure.type, parent=self)
        self.label.setPos(-self.label.boundingRect().width() / 2, size / 2 + 5)
        self.label.setBrush(QBrush(Qt.GlobalColor.black))

        # For showing communication range
        self.range_circle = None


class RoadGraphicsItem(QGraphicsPathItem):
    """Visual representation of a road"""

    def __init__(self, road):
        super().__init__()
        self.road = road
        self.road.item = self

        # Create road path
        path = QPainterPath()

        # Calculate road outline
        angle = road.angle
        perp_angle = angle + math.pi / 2
        half_width = road.width / 2

        # Road corners
        p1 = QPointF(
            road.start.x() + half_width * math.cos(perp_angle),
            road.start.y() + half_width * math.sin(perp_angle),
        )
        p2 = QPointF(
            road.start.x() - half_width * math.cos(perp_angle),
            road.start.y() - half_width * math.sin(perp_angle),
        )
        p3 = QPointF(
            road.end.x() - half_width * math.cos(perp_angle),
            road.end.y() - half_width * math.sin(perp_angle),
        )
        p4 = QPointF(
            road.end.x() + half_width * math.cos(perp_angle),
            road.end.y() + half_width * math.sin(perp_angle),
        )

        # Create road polygon
        path.moveTo(p1)
        path.lineTo(p2)
        path.lineTo(p3)
        path.lineTo(p4)
        path.closeSubpath()

        self.setPath(path)
        self.setBrush(QBrush(ROAD_COLOR))
        self.setPen(QPen(Qt.GlobalColor.black, 1))

        # Add lane markings
        if road.lanes > 1:
            self.add_lane_markings()

    def add_lane_markings(self):
        """Add visual lane markings to the road"""
        lanes = self.road.lanes
        start = self.road.start
        end = self.road.end
        angle = self.road.angle
        length = self.road.length

        # Add center line if bidirectional
        if self.road.bidirectional:
            center_line = QGraphicsLineItem(parent=self)
            center_line.setPen(QPen(Qt.GlobalColor.yellow, 2, Qt.PenStyle.DashLine))
            center_line.setLine(QLineF(start, end))

        # Add lane markings
        for i in range(1, lanes):
            lane_pos_start = self.road.get_lane_position(0, i, 1)
            lane_pos_end = self.road.get_lane_position(length, i, 1)

            lane_line = QGraphicsLineItem(parent=self)
            lane_line.setPen(QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine))
            lane_line.setLine(QLineF(lane_pos_start, lane_pos_end))

            if self.road.bidirectional:
                lane_pos_start = self.road.get_lane_position(0, i, -1)
                lane_pos_end = self.road.get_lane_position(length, i, -1)

                lane_line = QGraphicsLineItem(parent=self)
                lane_line.setPen(QPen(Qt.GlobalColor.white, 1, Qt.PenStyle.DashLine))
                lane_line.setLine(QLineF(lane_pos_start, lane_pos_end))


class MessageGraphicsItem(QGraphicsEllipseItem):
    """Visual representation of a message being transmitted"""

    def __init__(self, sender_pos, receivers, message_type):
        super().__init__(0, 0, 10, 10)
        self.sender_pos = sender_pos
        self.receivers = receivers
        self.message_type = message_type
        self.time_to_live = 10  # Animation frames

        # Set color based on message type
        if "Safety" in message_type:
            color = QColor(255, 0, 0, 150)  # Red for safety
        elif "Emergency" in message_type:
            color = QColor(255, 165, 0, 150)  # Orange for emergency
        elif "Signal" in message_type:
            color = QColor(0, 255, 0, 150)  # Green for traffic signals
        else:
            color = QColor(255, 255, 255, 150)  # White for others

        self.setBrush(QBrush(color))
        self.setPen(QPen(Qt.GlobalColor.transparent))
        self.setPos(sender_pos)

        # Create lines to receivers
        self.lines = []
        for receiver in receivers:
            line = QGraphicsLineItem(
                sender_pos.x(), sender_pos.y(), receiver.x, receiver.y
            )
            # Use the same color for the communication lines
            line.setPen(QPen(color, 1, Qt.PenStyle.DotLine))
            self.lines.append(line)

    def update_animation(self):
        """Update animation state"""
        self.time_to_live -= 1
        # Make message fade out
        opacity = self.time_to_live / 10

        brush = self.brush()
        color = brush.color()
        color.setAlpha(int(150 * opacity))
        self.setBrush(QBrush(color))

        for line in self.lines:
            pen = line.pen()
            color = pen.color()
            color.setAlpha(int(150 * opacity))
            pen.setColor(color)
            line.setPen(pen)

        return self.time_to_live <= 0


# GUI Classes
class SimulationView(QGraphicsView):
    """Main visualization view for the simulation"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Set background color
        self.setBackgroundBrush(QBrush(BACKGROUND_COLOR))

        # Zoom factor
        self.zoom_factor = 1.15

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        if event.angleDelta().y() > 0:
            self.scale(self.zoom_factor, self.zoom_factor)
        else:
            self.scale(1 / self.zoom_factor, 1 / self.zoom_factor)


class SimulationControlPanel(QWidget):
    """Control panel for simulation parameters"""

    simulationParamsChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Simulation control group
        sim_group = QGroupBox("Simulation Control")
        sim_layout = QVBoxLayout()

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
        sim_layout.addLayout(speed_layout)

        # Play/pause buttons
        buttons_layout = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")
        self.reset_button = QPushButton("Reset")
        buttons_layout.addWidget(self.play_button)
        buttons_layout.addWidget(self.pause_button)
        buttons_layout.addWidget(self.reset_button)
        sim_layout.addLayout(buttons_layout)

        sim_group.setLayout(sim_layout)
        layout.addWidget(sim_group)

        # Vehicle control group
        vehicle_group = QGroupBox("Vehicle Settings")
        vehicle_layout = QFormLayout()

        # Vehicle number control
        self.vehicle_count = QSpinBox()
        self.vehicle_count.setMinimum(0)
        self.vehicle_count.setMaximum(500)
        self.vehicle_count.setValue(20)
        vehicle_layout.addRow("Number of Vehicles:", self.vehicle_count)

        # Vehicle type distribution
        self.vehicle_type_combo = QComboBox()
        for vtype in VEHICLE_TYPES:
            self.vehicle_type_combo.addItem(vtype)
        vehicle_layout.addRow("Add Vehicle Type:", self.vehicle_type_combo)

        # Add vehicle button
        self.add_vehicle_button = QPushButton("Add Vehicle")
        vehicle_layout.addRow("", self.add_vehicle_button)

        vehicle_group.setLayout(vehicle_layout)
        layout.addWidget(vehicle_group)

        # Infrastructure control group
        infra_group = QGroupBox("Infrastructure Settings")
        infra_layout = QFormLayout()

        # Infrastructure type selection
        self.infra_type_combo = QComboBox()
        for itype in INFRASTRUCTURE_TYPES:
            self.infra_type_combo.addItem(itype)
        infra_layout.addRow("Infrastructure Type:", self.infra_type_combo)

        # Add infrastructure button
        self.add_infra_button = QPushButton("Add Infrastructure")
        infra_layout.addRow("", self.add_infra_button)

        infra_group.setLayout(infra_layout)
        layout.addWidget(infra_group)

        # Communication settings group
        comm_group = QGroupBox("Communication Settings")
        comm_layout = QFormLayout()

        # Communication range control
        self.comm_range_slider = QSlider(Qt.Orientation.Horizontal)
        self.comm_range_slider.setMinimum(50)
        self.comm_range_slider.setMaximum(500)
        self.comm_range_slider.setValue(MAX_RANGE)
        self.comm_range_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.comm_range_slider.setTickInterval(50)
        self.comm_range_label = QLabel(f"Communication Range ({MAX_RANGE}m):")
        comm_layout.addRow(self.comm_range_label, self.comm_range_slider)

        # Show communication visualization
        self.show_comm_checkbox = QCheckBox("Show Communication")
        self.show_comm_checkbox.setChecked(True)
        comm_layout.addRow("", self.show_comm_checkbox)

        # Show vehicle ranges
        self.show_ranges_checkbox = QCheckBox("Show Communication Ranges")
        self.show_ranges_checkbox.setChecked(False)
        comm_layout.addRow("", self.show_ranges_checkbox)

        comm_group.setLayout(comm_layout)
        layout.addWidget(comm_group)

        # Connect signals
        self.speed_slider.valueChanged.connect(self.updateSimulationSpeed)
        self.vehicle_count.valueChanged.connect(self.updateSimulationParams)
        self.comm_range_slider.valueChanged.connect(self.updateCommRange)
        self.show_ranges_checkbox.toggled.connect(self.updateSimulationParams)

        self.setLayout(layout)

    def updateSimulationSpeed(self, value):
        """Update simulation speed label and emit signal"""
        self.speed_label.setText(f"{value}%")
        params = {"sim_speed": value / 100.0}
        self.simulationParamsChanged.emit(params)

    def updateSimulationParams(self):
        """Emit signal with updated simulation parameters"""
        params = {
            "vehicle_count": self.vehicle_count.value(),
            "show_ranges": self.show_ranges_checkbox.isChecked(),
        }
        self.simulationParamsChanged.emit(params)

    def updateCommRange(self, value):
        """Update communication range label and emit signal"""
        self.comm_range_label.setText(f"Communication Range ({value}m):")
        params = {"comm_range": value}
        self.simulationParamsChanged.emit(params)


class StatisticsPanel(QWidget):
    """Panel for displaying simulation statistics"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

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

        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)

        # Communication statistics
        comm_group = QGroupBox("Communication Statistics")
        comm_layout = QFormLayout()

        self.msg_types_label = QLabel("None")
        comm_layout.addRow("Message Types:", self.msg_types_label)

        self.comm_efficiency_label = QLabel("0%")
        comm_layout.addRow("Communication Efficiency:", self.comm_efficiency_label)

        comm_group.setLayout(comm_layout)
        layout.addWidget(comm_group)

        self.setLayout(layout)

    def update_statistics(self, stats):
        """Update statistics display"""
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

        # Calculate communication efficiency
        if stats["messages_sent"] > 0:
            efficiency = (stats["messages_received"] / stats["messages_sent"]) * 100
            self.comm_efficiency_label.setText(f"{efficiency:.1f}%")
        else:
            self.comm_efficiency_label.setText("0%")

        # Show message type distribution
        if stats["message_types"]:
            msg_types_text = ""
            total = sum(stats["message_types"].values())
            for msg_type, count in sorted(
                stats["message_types"].items(), key=lambda x: x[1], reverse=True
            )[:3]:
                percentage = (count / total) * 100
                msg_types_text += f"{msg_type}: {percentage:.1f}%\n"
            self.msg_types_label.setText(msg_types_text)
        else:
            self.msg_types_label.setText("None")


class V2XSimulator(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.initUI()
        self.initSimulation()

    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("V2X Simulation Application")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()

        # Create simulation view
        self.scene = QGraphicsScene()
        self.view = SimulationView(self.scene)

        # Create control panels in a tab widget
        panel_widget = QWidget()
        panel_layout = QVBoxLayout()

        self.control_panel = SimulationControlPanel()
        self.control_panel.simulationParamsChanged.connect(
            self.on_simulation_params_changed
        )
        self.control_panel.play_button.clicked.connect(self.start_simulation)
        self.control_panel.pause_button.clicked.connect(self.pause_simulation)
        self.control_panel.reset_button.clicked.connect(self.reset_simulation)
        self.control_panel.add_vehicle_button.clicked.connect(self.add_random_vehicle)
        self.control_panel.add_infra_button.clicked.connect(
            self.add_random_infrastructure
        )

        self.stats_panel = StatisticsPanel()

        panel_layout.addWidget(self.control_panel)
        panel_layout.addWidget(self.stats_panel)
        panel_widget.setLayout(panel_layout)

        # Set fixed width for panel
        panel_widget.setFixedWidth(350)

        # Add widgets to main layout
        main_layout.addWidget(self.view)
        main_layout.addWidget(panel_widget)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Simulation ready. Press Play to start.")

        # Create simulation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.delta_time = UPDATE_INTERVAL / 1000.0  # seconds
        self.simulation_speed = 1.0

        # For message animations
        self.active_message_animations = []

    def initSimulation(self):
        """Initialize the simulation environment"""
        self.simulation = SimulationEnvironment()

        # Create some initial roads
        self.create_default_road_network()

        # Add some initial vehicles
        for _ in range(20):
            self.add_random_vehicle()

        # Add some infrastructure
        for _ in range(5):
            self.add_random_infrastructure()

    def create_default_road_network(self):
        """Create a default road network for the simulation"""
        # Create a simple grid road network
        grid_size = 500
        center_x = 0
        center_y = 0

        # Horizontal roads
        for i in range(-1, 2):
            y = center_y + i * grid_size
            road = Road(
                center_x - grid_size,
                y,
                center_x + grid_size,
                y,
                lanes=2,
                bidirectional=True,
            )
            self.simulation.add_road(road)
            self.scene.addItem(RoadGraphicsItem(road))

        # Vertical roads
        for i in range(-1, 2):
            x = center_x + i * grid_size
            road = Road(
                x,
                center_y - grid_size,
                x,
                center_y + grid_size,
                lanes=2,
                bidirectional=True,
            )
            self.simulation.add_road(road)
            self.scene.addItem(RoadGraphicsItem(road))

        # Add a diagonal road for variety
        road = Road(
            center_x - grid_size,
            center_y - grid_size,
            center_x + grid_size,
            center_y + grid_size,
            lanes=2,
            bidirectional=True,
        )
        self.simulation.add_road(road)
        self.scene.addItem(RoadGraphicsItem(road))

        # Set view to show the entire road network
        self.view.setSceneRect(
            QRectF(
                center_x - grid_size - 100,
                center_y - grid_size - 100,
                grid_size * 2 + 200,
                grid_size * 2 + 200,
            )
        )
        self.view.fitInView(self.view.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def add_random_vehicle(self):
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

        # Create vehicle
        vehicle_type = self.control_panel.vehicle_type_combo.currentText()
        vehicle = Vehicle(
            pos.x(),
            pos.y(),
            vehicle_type,
            speed=random.uniform(30, 90),  # Random speed between 30-90 km/h
            direction=road.angle if direction == 1 else road.angle + math.pi,
        )

        # Add to simulation
        self.simulation.add_vehicle(vehicle)

        # Create visual representation
        vehicle_item = VehicleGraphicsItem(vehicle)
        self.scene.addItem(vehicle_item)

        # Generate a route for the vehicle
        self.generate_vehicle_route(vehicle)

        return vehicle

    def add_random_infrastructure(self):
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

        # Create infrastructure
        infra_type = self.control_panel.infra_type_combo.currentText()
        infrastructure = Infrastructure(x, y, infra_type)

        # Add to simulation
        self.simulation.add_infrastructure(infrastructure)

        # Create visual representation
        infra_item = InfrastructureGraphicsItem(infrastructure)
        self.scene.addItem(infra_item)

        return infrastructure

    def generate_vehicle_route(self, vehicle):
        """Generate a random route for a vehicle through the road network"""
        if not self.simulation.roads:
            return

        # Create a simple route: go to 3-5 random points
        route_length = random.randint(3, 5)
        route = []

        for _ in range(route_length):
            road = random.choice(self.simulation.roads)
            distance = random.uniform(0, road.length)
            lane = random.randint(0, road.lanes - 1)
            direction = random.choice([1, -1]) if road.bidirectional else 1

            pos = road.get_lane_position(distance, lane, direction)
            route.append(pos)

        vehicle.set_route(route)

    def start_simulation(self):
        """Start the simulation"""
        self.timer.start(UPDATE_INTERVAL)
        self.statusBar.showMessage("Simulation running")

    def pause_simulation(self):
        """Pause the simulation"""
        self.timer.stop()
        self.statusBar.showMessage("Simulation paused")

    def reset_simulation(self):
        """Reset the simulation to initial state"""
        self.timer.stop()

        # Clear the scene
        self.scene.clear()

        # Reinitialize the simulation
        self.initSimulation()

        self.statusBar.showMessage("Simulation reset")

    def update_simulation(self):
        """Update the simulation for one time step"""
        # Update the simulation with adjusted time step based on simulation speed
        self.simulation.update(self.delta_time * self.simulation_speed)

        # Update all vehicle graphics
        for vehicle in self.simulation.vehicles:
            if vehicle.item:
                vehicle.item.update_graphics()

        # Visualize communications
        if self.control_panel.show_comm_checkbox.isChecked():
            self.visualize_communications()

        # Show/hide communication ranges
        if self.control_panel.show_ranges_checkbox.isChecked():
            self.show_communication_ranges()
        else:
            self.hide_communication_ranges()

        # Update statistics panel
        stats = self.simulation.statistics.copy()
        stats["time"] = self.simulation.time
        self.stats_panel.update_statistics(stats)

        # Update message animations
        self.update_message_animations()

    def visualize_communications(self):
        """Visualize communication between entities"""
        # Process recent messages in the simulation's message history
        recent_messages = list(self.simulation.message_history)[
            -5:
        ]  # Only show most recent messages

        for message in recent_messages:
            sender = message.sender

            # Find receivers in range
            receivers = []
            for entity in self.simulation.vehicles + self.simulation.infrastructure:
                if entity != sender:
                    if (
                        distance(sender.position(), entity.position())
                        <= sender.comm_range
                    ):
                        receivers.append(entity)

            # Create message visualization if not already created
            if (
                receivers and random.random() < 0.3
            ):  # Only visualize some messages to avoid cluttering
                msg_item = MessageGraphicsItem(
                    sender.position(), receivers, message.type
                )
                self.scene.addItem(msg_item)

                # Add lines to receivers
                for line in msg_item.lines:
                    self.scene.addItem(line)

                # Store for animation updates
                self.active_message_animations.append(msg_item)

    def update_message_animations(self):
        """Update all active message animations"""
        # Update animations and remove expired ones
        i = 0
        while i < len(self.active_message_animations):
            msg_item = self.active_message_animations[i]

            # Update the animation
            expired = msg_item.update_animation()

            if expired:
                # Remove the lines first
                for line in msg_item.lines:
                    self.scene.removeItem(line)

                # Remove the message item
                self.scene.removeItem(msg_item)

                # Remove from active animations
                self.active_message_animations.pop(i)
            else:
                i += 1

    def show_communication_ranges(self):
        """Show communication range circles for all entities"""
        for vehicle in self.simulation.vehicles:
            if vehicle.item and not vehicle.item.range_circle:
                # Create a circle for the communication range
                range_circle = self.scene.addEllipse(
                    -vehicle.comm_range,
                    -vehicle.comm_range,
                    vehicle.comm_range * 2,
                    vehicle.comm_range * 2,
                    QPen(QColor(0, 255, 0, 50), 1, Qt.PenStyle.DashLine),
                    QBrush(QColor(0, 255, 0, 20)),
                )
                range_circle.setPos(vehicle.x, vehicle.y)
                vehicle.item.range_circle = range_circle

        for infra in self.simulation.infrastructure:
            if infra.item and not infra.item.range_circle:
                # Create a circle for the communication range
                range_circle = self.scene.addEllipse(
                    -infra.comm_range,
                    -infra.comm_range,
                    infra.comm_range * 2,
                    infra.comm_range * 2,
                    QPen(QColor(0, 100, 255, 50), 1, Qt.PenStyle.DashLine),
                    QBrush(QColor(0, 100, 255, 20)),
                )
                range_circle.setPos(infra.x, infra.y)
                infra.item.range_circle = range_circle

    def hide_communication_ranges(self):
        """Hide communication range circles for all entities"""
        for vehicle in self.simulation.vehicles:
            if vehicle.item and vehicle.item.range_circle:
                self.scene.removeItem(vehicle.item.range_circle)
                vehicle.item.range_circle = None

        for infra in self.simulation.infrastructure:
            if infra.item and infra.item.range_circle:
                self.scene.removeItem(infra.item.range_circle)
                infra.item.range_circle = None

    def on_simulation_params_changed(self, params):
        """Handle changes to simulation parameters"""
        if "sim_speed" in params:
            self.simulation_speed = params["sim_speed"]

        if "comm_range" in params:
            # Update communication range for all entities
            new_range = params["comm_range"]
            for vehicle in self.simulation.vehicles:
                vehicle.comm_range = new_range
            for infra in self.simulation.infrastructure:
                infra.comm_range = new_range

            # Update visualization if ranges are shown
            if self.control_panel.show_ranges_checkbox.isChecked():
                self.hide_communication_ranges()
                self.show_communication_ranges()

        if "show_ranges" in params:
            if params["show_ranges"]:
                self.show_communication_ranges()
            else:
                self.hide_communication_ranges()

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
