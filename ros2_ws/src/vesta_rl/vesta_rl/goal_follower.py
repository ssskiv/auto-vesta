"""Drive to goals clicked in RViz ("2D Goal Pose" -> /goal_pose).

For every goal: planner_server ComputePathToPose -> controller_server FollowPath
(vesta_neural_controller). A new goal cancels the one being driven. Started by
nav2.launch.py; stands in for bt_navigator, whose recovery behaviours (spin in
place, back up) an Ackermann car can't do.
"""
import collections
import math

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, FollowPath
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import String

STATUS = {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED', GoalStatus.STATUS_CANCELED: 'CANCELED',
          GoalStatus.STATUS_ABORTED: 'ABORTED'}


class GoalFollower(Node):
    def __init__(self):
        super().__init__('goal_follower')
        self.planner_id = self.declare_parameter('planner_id', 'GridBased').value
        self.controller_id = self.declare_parameter('controller_id', 'FollowPath').value
        self.goal_checker_id = self.declare_parameter('goal_checker_id', 'general_goal_checker').value
        self.planner = ActionClient(self, ComputePathToPose, 'compute_path_to_pose')
        self.follower = ActionClient(self, FollowPath, 'follow_path')
        self.create_subscription(PoseStamped, 'goal_pose', self.on_goal, 10)
        self.create_subscription(String, 'neural_controller/source', lambda m: self.sources.update([m.data]), 10)
        self.sources = collections.Counter()
        self.goal_id = 0          # increments with every click; stale callbacks are ignored
        self.follow_handle = None
        self.get_logger().info('waiting for goals on /goal_pose (RViz: "2D Goal Pose")')

    def on_goal(self, goal):
        self.goal_id += 1
        gid = self.goal_id
        q = goal.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.get_logger().info(f'goal #{gid}: ({goal.pose.position.x:.1f}, {goal.pose.position.y:.1f}, '
                               f'yaw {yaw:.2f}) in {goal.header.frame_id}')
        if self.follow_handle is not None:
            self.follow_handle.cancel_goal_async()
            self.follow_handle = None
        if not self.planner.wait_for_server(timeout_sec=2.0) or not self.follower.wait_for_server(timeout_sec=2.0):
            self.get_logger().error('planner_server / controller_server not available')
            return
        self.planner.send_goal_async(ComputePathToPose.Goal(goal=goal, planner_id=self.planner_id)) \
            .add_done_callback(lambda f: self.on_plan_accepted(f, gid))

    def on_plan_accepted(self, future, gid):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error(f'goal #{gid}: planner rejected the goal')
            return
        handle.get_result_async().add_done_callback(lambda f: self.on_path(f, gid))

    def on_path(self, future, gid):
        if gid != self.goal_id:
            return  # a newer goal was clicked meanwhile
        path = future.result().result.path
        if not path.poses:
            self.get_logger().error(f'goal #{gid}: no path (goal in an obstacle or unreachable?)')
            return
        length = sum(math.hypot(b.pose.position.x - a.pose.position.x, b.pose.position.y - a.pose.position.y)
                     for a, b in zip(path.poses[:-1], path.poses[1:]))
        self.get_logger().info(f'goal #{gid}: path {length:.1f} m, driving')
        self.sources.clear()
        self.follower.send_goal_async(FollowPath.Goal(path=path, controller_id=self.controller_id,
                                                      goal_checker_id=self.goal_checker_id)) \
            .add_done_callback(lambda f: self.on_follow_accepted(f, gid))

    def on_follow_accepted(self, future, gid):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().error(f'goal #{gid}: controller rejected the path')
            return
        if gid != self.goal_id:
            handle.cancel_goal_async()
            return
        self.follow_handle = handle
        handle.get_result_async().add_done_callback(lambda f: self.on_done(f, gid))

    def on_done(self, future, gid):
        status = STATUS.get(future.result().status, str(future.result().status))
        n = sum(self.sources.values()) or 1
        self.get_logger().info(
            f'goal #{gid}: {status}; control share NN {self.sources["NN"] / n:.0%}, '
            f'RPP {self.sources["RPP"] / n:.0%}, STOP {self.sources["STOP"] / n:.0%}')
        if gid == self.goal_id:
            self.follow_handle = None


def main():
    rclpy.init()
    node = GoalFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
