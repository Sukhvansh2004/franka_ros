#!/usr/bin/env python

import rospy
import tf.transformations
import numpy as np

from interactive_markers.interactive_marker_server import \
    InteractiveMarkerServer, InteractiveMarkerFeedback
from visualization_msgs.msg import InteractiveMarker, \
    InteractiveMarkerControl
from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState
from tf.transformations import quaternion_matrix, translation_matrix, inverse_matrix
from scipy.linalg import logm
import copy

prev_marker_pose = None
marker_pose = PoseStamped()
pose_pub = None
# [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
position_limits = [[-0.6, 0.6], [-0.6, 0.6], [0.05, 0.9]]

SE3_TOLERANCE = 0.001  

def msg_to_matrix(pose_msg):
    """Converts geometry_msgs/Pose to a 4x4 numpy transformation matrix."""
    t = [pose_msg.position.x, pose_msg.position.y, pose_msg.position.z]
    q = [pose_msg.orientation.x, pose_msg.orientation.y, pose_msg.orientation.z, pose_msg.orientation.w]
    
    T_trans = translation_matrix(t)
    T_rot = quaternion_matrix(q)
    
    # Combine rotation and translation
    return np.dot(T_trans, T_rot)

def get_se3_error_norm(T_prev, T_curr):
    """
    Calculates the norm of the Lie Algebra error (twist magnitude).
    T_err = T_prev^-1 * T_curr
    error = || log(T_err) ||
    """
    # 1. Calculate relative transform
    T_prev_inv = inverse_matrix(T_prev)
    T_rel = np.dot(T_prev_inv, T_curr)
    
    # 2. Compute the Matrix Logarithm to get to se(3)
    # logm returns a complex matrix usually, but for SE(3) real inputs 
    # the result should be real (ignoring numerical noise)
    se3_matrix = np.real(logm(T_rel))
    
    # 3. Vee operator: Extract the 6D twist vector from the 4x4 skew-symmetric matrix
    # The structure of se(3) matrix is:
    # [ [0, -wz, wy, vx],
    #   [wz, 0, -wx, vy],
    #   [-wy, wx, 0, vz],
    #   [0,  0,  0,  1 ] ]
    
    vx = se3_matrix[0, 3]
    vy = se3_matrix[1, 3]
    vz = se3_matrix[2, 3]
    wx = se3_matrix[2, 1]
    wy = se3_matrix[0, 2]
    wz = se3_matrix[1, 0]
    
    twist = np.array([vx, vy, vz, wx, wy, wz])
    
    # 4. Return the Euclidean norm of the twist
    return np.linalg.norm(twist)

def publisher_callback(msg, link_name):
    global prev_marker_pose, marker_pose

    marker_pose.header.frame_id = link_name
    marker_pose.header.stamp = rospy.Time(0)

    # Convert ROS messages to 4x4 Matrices
    T_curr = msg_to_matrix(marker_pose.pose)
    T_prev = msg_to_matrix(prev_marker_pose.pose)

    # Calculate SE(3) error on the Lie Algebra
    error_magnitude = get_se3_error_norm(T_prev, T_curr)

    # Check tolerance and publish
    if error_magnitude > SE3_TOLERANCE:
        pose_pub.publish(marker_pose)
        prev_marker_pose = copy.deepcopy(marker_pose)

def process_feedback(feedback):
    if feedback.event_type == InteractiveMarkerFeedback.POSE_UPDATE:
        marker_pose.pose.position.x = max([min([feedback.pose.position.x,
                                          position_limits[0][1]]),
                                          position_limits[0][0]])
        marker_pose.pose.position.y = max([min([feedback.pose.position.y,
                                          position_limits[1][1]]),
                                          position_limits[1][0]])
        marker_pose.pose.position.z = max([min([feedback.pose.position.z,
                                          position_limits[2][1]]),
                                          position_limits[2][0]])
        marker_pose.pose.orientation = feedback.pose.orientation
    server.applyChanges()


def wait_for_initial_pose():
    global marker_pose, prev_marker_pose

    msg = rospy.wait_for_message("franka_state_controller/franka_states",
                                 FrankaState)  # type: FrankaState

    initial_quaternion = \
        tf.transformations.quaternion_from_matrix(
            np.transpose(np.reshape(msg.O_T_EE,
                                    (4, 4))))
    initial_quaternion = initial_quaternion / \
        np.linalg.norm(initial_quaternion)
    marker_pose.pose.orientation.x = initial_quaternion[0]
    marker_pose.pose.orientation.y = initial_quaternion[1]
    marker_pose.pose.orientation.z = initial_quaternion[2]
    marker_pose.pose.orientation.w = initial_quaternion[3]
    marker_pose.pose.position.x = msg.O_T_EE[12]
    marker_pose.pose.position.y = msg.O_T_EE[13]
    marker_pose.pose.position.z = msg.O_T_EE[14]

    prev_marker_pose = copy.deepcopy(marker_pose)

if __name__ == "__main__":
    rospy.init_node("equilibrium_pose_node")
    listener = tf.TransformListener()
    link_name = rospy.get_param("~link_name")

    wait_for_initial_pose()

    pose_pub = rospy.Publisher(
        "equilibrium_pose", PoseStamped, queue_size=10)
    server = InteractiveMarkerServer("equilibrium_pose_marker")
    int_marker = InteractiveMarker()
    int_marker.header.frame_id = link_name
    int_marker.scale = 0.3
    int_marker.name = "equilibrium_pose"
    int_marker.description = ("Equilibrium Pose\nBE CAREFUL! "
                              "If you move the \nequilibrium "
                              "pose the robot will follow it\n"
                              "so be aware of potential collisions")
    int_marker.pose = marker_pose.pose
    # run pose publisher
    rospy.Timer(rospy.Duration(0.005),
                lambda msg: publisher_callback(msg, link_name))

    # insert a box
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 1
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "rotate_x"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    int_marker.controls.append(control)

    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 1
    control.orientation.y = 0
    control.orientation.z = 0
    control.name = "move_x"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0
    control.name = "rotate_y"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    int_marker.controls.append(control)
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 1
    control.orientation.z = 0
    control.name = "move_y"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 0
    control.orientation.z = 1
    control.name = "rotate_z"
    control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
    int_marker.controls.append(control)
    control = InteractiveMarkerControl()
    control.orientation.w = 1
    control.orientation.x = 0
    control.orientation.y = 0
    control.orientation.z = 1
    control.name = "move_z"
    control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
    int_marker.controls.append(control)
    server.insert(int_marker, process_feedback)

    server.applyChanges()

    rospy.spin()
