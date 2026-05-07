import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import json
import base64
import io
import numpy as np
from PIL import Image, ImageDraw
import habitat_sim
import quaternion
import magnum as mn
from transformers import pipeline
import random

STATE_FILE = "/root/agent_state.json"

# ========== 全局变量 ==========
output_frames = []
status_message = ""
trajectory = []
frame_count = [0]
occ_map = None
visited_frontiers = set()
sim = None
agent = None
detector = None

# ========== 初始化 ==========
def init():
    global sim, agent, detector, occ_map, visited_frontiers, trajectory, frame_count

    scene_path = "/root/habitat-data/scene_datasets/hm3d/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = scene_path
    backend_cfg.enable_physics = False
    backend_cfg.gpu_device_id = 0 

    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "color_sensor"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [480, 640]
    rgb_sensor.position = [0.0, 0.88, 0.0]
    rgb_sensor.orientation = mn.Vector3(-0.3, 0, 0)
    rgb_sensor.hfov = mn.Deg(90)

    depth_sensor = habitat_sim.CameraSensorSpec()
    depth_sensor.uuid = "depth_sensor"
    depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor.resolution = [480, 640]
    depth_sensor.position = [0.0, 0.88, 0.0]
    depth_sensor.orientation = mn.Vector3(-0.3, 0, 0)
    depth_sensor.hfov = mn.Deg(90)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    cfg = habitat_sim.Configuration(backend_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    agent = sim.initialize_agent(0)

    # ========== 读取上次位置 ==========
    state = habitat_sim.AgentState()
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r") as f:
                saved = json.load(f)
            state.position = np.array(saved["position"])
            q = saved["rotation"]
            state.rotation = quaternion.quaternion(q[0], q[1], q[2], q[3])
            agent.set_state(state)
            print(f"[状态恢复] 从上次位置出发: {saved['position']}", file=sys.stderr)
        except Exception as e:
            print(f"[状态恢复失败，使用默认起点] {e}", file=sys.stderr)
            state.position = np.array([-4.896151, 3.163378, -6.3560014])
            agent.set_state(state)
            for _ in range(6):
                sim.step("turn_left")
    else:
        print("[首次启动] 使用默认起点", file=sys.stderr)
        state.position = np.array([-4.896151, 3.163378, -6.3560014])
        agent.set_state(state)
        for _ in range(6):
            sim.step("turn_left")

    detector = pipeline(
        "zero-shot-object-detection",
        model="/root/owlvit-large-patch14",
        device=0
    )

    occ_map = OccupancyMap(resolution=0.25)
    visited_frontiers = set()
    trajectory = []
    frame_count = [0]

# ========== 保存当前位置 ==========
def save_state():
    s = agent.get_state()
    q = s.rotation
    data = {
        "position": s.position.tolist(),
        "rotation": [q.w, q.x, q.y, q.z]
    }
    with open(STATE_FILE, "w") as f:
        json.dump(data, f)
    print(f"[状态保存] 当前位置: {s.position.tolist()}", file=sys.stderr)

# ========== 工具函数 ==========
def draw_robot_marker(rgb, agent_pos, agent_rot, all_positions=None):
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    map_size = 120
    map_x0 = rgb.shape[1] - map_size - 10
    map_y0 = rgb.shape[0] - map_size - 10
    draw.rectangle([map_x0, map_y0, map_x0+map_size, map_y0+map_size],
                   fill=(30,30,30), outline=(255,255,255), width=2)
    x_min, x_max = -12.0, 4.0
    z_min, z_max = -10.0, 2.0
    def world_to_map(x, z):
        mx = int((x - x_min) / (x_max - x_min) * map_size) + map_x0
        mz = int((z - z_min) / (z_max - z_min) * map_size) + map_y0
        return mx, mz
    if all_positions and len(all_positions) > 1:
        for i in range(1, len(all_positions)):
            p1 = world_to_map(all_positions[i-1][0], all_positions[i-1][2])
            p2 = world_to_map(all_positions[i][0], all_positions[i][2])
            draw.line([p1, p2], fill=(100,100,255), width=1)
    rx, rz = world_to_map(agent_pos[0], agent_pos[2])
    draw.ellipse([rx-4, rz-4, rx+4, rz+4], fill=(0,120,255), outline=(255,255,255))
    rot_matrix = quaternion.as_rotation_matrix(agent_rot)
    forward = rot_matrix @ np.array([0, 0, -1])
    ax = rx + int(forward[0] * 10)
    az = rz + int(forward[2] * 10)
    draw.line([rx, rz, ax, az], fill=(0,255,0), width=2)
    draw.text((map_x0+2, map_y0+2),
              f"({agent_pos[0]:.1f},{agent_pos[2]:.1f})", fill=(255,255,0))
    return np.array(img)

def show_rgb(tag=""):
    global output_frames
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"][:, :, :3]
    pos = agent.get_state().position
    trajectory.append(np.array(pos))
    rgb_marked = draw_robot_marker(rgb, np.array(pos),
                                   agent.get_state().rotation, trajectory)
    buf = io.BytesIO()
    Image.fromarray(rgb_marked).save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode()
    output_frames.append(b64)
    return rgb

# ========== OccupancyMap ==========
class OccupancyMap:
    def __init__(self, resolution=0.25):
        self.resolution = resolution
        self.grid = {}

    def world_to_grid(self, x, z):
        return (round(x / self.resolution), round(z / self.resolution))

    def grid_to_world(self, ix, iz, y):
        return np.array([ix * self.resolution, y, iz * self.resolution])

    def update(self, agent_pos, depth_obs, rot_matrix):
        H, W = depth_obs.shape
        fov = np.deg2rad(90)
        fx = (W / 2) / np.tan(fov / 2)
        forward = rot_matrix @ np.array([0, 0, -1])
        right   = rot_matrix @ np.array([1, 0, 0])
        forward[1] = 0
        right[1]   = 0
        ax, az = self.world_to_grid(agent_pos[0], agent_pos[2])
        for dx in range(-3, 4):
            for dz in range(-3, 4):
                key = (ax+dx, az+dz)
                if key not in self.grid:
                    self.grid[key] = "free"
        for col in range(0, W, 15):
            depth = depth_obs[H//2, col]
            if depth <= 0.1 or depth > 6.0:
                continue
            x_cam = (col - W/2) / fx * depth
            world_pt = agent_pos + forward * depth + right * x_cam
            ix, iz = self.world_to_grid(world_pt[0], world_pt[2])
            self.grid[(ix, iz)] = "obstacle"

    def get_frontiers(self, agent_pos):
        frontiers = []
        seen = set()
        for (ix, iz), status in self.grid.items():
            if status != "free":
                continue
            for dx, dz in [(1,0),(-1,0),(0,1),(0,-1)]:
                neighbor = (ix+dx, iz+dz)
                if neighbor not in self.grid:
                    if (ix, iz) not in seen:
                        seen.add((ix, iz))
                        world_pt = self.grid_to_world(ix, iz, agent_pos[1])
                        dist = np.linalg.norm(
                            np.array([world_pt[0]-agent_pos[0],
                                      world_pt[2]-agent_pos[2]])
                        )
                        frontiers.append((world_pt, dist))
                    break
        frontiers.sort(key=lambda x: x[1])
        return frontiers

# ========== 导航函数 ==========
def turn_to_face(target_pos, tolerance=15, max_turns=12):
    target = np.array(target_pos)
    for i in range(max_turns):
        agent_pos = np.array(agent.get_state().position)
        rot_matrix = quaternion.as_rotation_matrix(agent.get_state().rotation)
        forward = rot_matrix @ np.array([0, 0, -1])
        fw = np.array([forward[0], forward[2]])
        fw /= np.linalg.norm(fw) + 1e-8
        to_t = np.array([target[0]-agent_pos[0], target[2]-agent_pos[2]])
        dist = np.linalg.norm(to_t)
        if dist < 0.01:
            break
        to_t /= dist
        cos_a = np.clip(np.dot(fw, to_t), -1, 1)
        angle = np.degrees(np.arccos(cos_a))
        if angle < tolerance:
            break
        right = rot_matrix @ np.array([1, 0, 0])
        rw = np.array([right[0], right[2]])
        is_right = np.dot(rw, to_t) > 0
        sim.step("turn_right" if is_right else "turn_left")

def is_valid_observation():
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"][:, :, :3]
    return np.mean(rgb) > 20

def force_navigate(target_pos):
    global occ_map
    target_pos = np.array(target_pos)
    agent_pos = np.array(agent.get_state().position)
    path = habitat_sim.ShortestPath()
    path.requested_start = agent_pos
    path.requested_end = target_pos
    found = sim.pathfinder.find_path(path)
    if not found:
        return False
    waypoints = [np.array(p) for p in path.points]
    step_count = 0
    for wp_idx, wp in enumerate(waypoints[1:], 1):
        for step in range(100):
            pos = np.array(agent.get_state().position)
            dist = np.linalg.norm(pos[[0,2]] - wp[[0,2]])
            if dist < 0.35:
                break
            if not is_valid_observation():
                for _ in range(4):
                    sim.step("turn_right")
                for _ in range(3):
                    sim.step("move_forward")
                break
            rot = quaternion.as_rotation_matrix(agent.get_state().rotation)
            fw = rot @ np.array([0,0,-1])
            fw2 = np.array([fw[0], fw[2]])
            fw2 /= np.linalg.norm(fw2) + 1e-8
            to_wp = wp[[0,2]] - pos[[0,2]]
            to_wp /= np.linalg.norm(to_wp) + 1e-8
            cos_a = np.clip(np.dot(fw2, to_wp), -1, 1)
            angle = np.degrees(np.arccos(cos_a))
            right = rot @ np.array([1,0,0])
            rw = np.array([right[0], right[2]])
            is_right = np.dot(rw, to_wp) > 0
            if angle > 15:
                sim.step("turn_right" if is_right else "turn_left")
            else:
                prev_pos = np.array(agent.get_state().position)
                sim.step("move_forward")
                new_pos = np.array(agent.get_state().position)
                obs = sim.get_sensor_observations()
                rot_matrix = quaternion.as_rotation_matrix(agent.get_state().rotation)
                occ_map.update(new_pos, obs["depth_sensor"], rot_matrix)
                if np.linalg.norm(new_pos - prev_pos) < 0.01:
                    sim.step("turn_left")
                    sim.step("move_forward")
                    sim.step("turn_right")
                    sim.step("move_forward")
            step_count += 1
            if step_count % 8 == 0:
                show_rgb("nav")
    final_pos = np.array(agent.get_state().position)
    final_dist = np.linalg.norm(final_pos[[0,2]] - target_pos[[0,2]])
    return final_dist < 1.0

def get_nav_target(box, depth_obs):
    H, W = depth_obs.shape
    cx = int((box["xmin"] + box["xmax"]) / 2)
    margin = 10
    cy_bottom = int(box["ymax"]) - margin
    cx1 = max(0, cx - margin)
    cx2 = min(W, cx + margin)
    cy1 = max(0, cy_bottom - margin)
    cy2 = min(H, cy_bottom + margin)
    depth_vals = depth_obs[cy1:cy2, cx1:cx2]
    depth_vals = depth_vals[(depth_vals > 0.1) & (depth_vals < 8.0)]
    depth = float(np.percentile(depth_vals, 20)) if len(depth_vals) > 0 else 2.0
    depth = np.clip(depth, 0.5, 6.0)
    fov = np.deg2rad(90)
    fx = (W / 2) / np.tan(fov / 2)
    x_cam = (cx - W / 2) / fx * depth
    agent_state = agent.get_state()
    agent_pos = np.array(agent_state.position)
    rot_matrix = quaternion.as_rotation_matrix(agent_state.rotation)
    forward = rot_matrix @ np.array([0, 0, -1])
    right   = rot_matrix @ np.array([1, 0, 0])
    forward[1] = 0
    right[1]   = 0
    object_pos = agent_pos + forward * depth + right * x_cam
    object_pos[1] = agent_pos[1]
    to_agent = agent_pos - object_pos
    to_agent[1] = 0
    dist_to_obj = np.linalg.norm(to_agent)
    to_agent_norm = to_agent / dist_to_obj if dist_to_obj > 0 else forward
    for offset in [0.8, 1.0, 1.2, 1.5, 2.0]:
        candidate = object_pos + to_agent_norm * offset
        candidate[1] = agent_pos[1]
        snapped = sim.pathfinder.snap_point(candidate)
        snap_dist = np.linalg.norm(np.array(snapped) - candidate)
        y_diff = abs(snapped[1] - agent_pos[1])
        if snap_dist < 0.5 and sim.pathfinder.is_navigable(snapped) and y_diff < 0.3:
            return np.array(snapped), object_pos
    fallback = agent_pos + forward * max(0.5, depth - 0.8)
    fallback[1] = agent_pos[1]
    snapped = sim.pathfinder.snap_point(fallback)
    return np.array(snapped), object_pos

def look_around_and_detect(target_labels, threshold=0.5):
    global occ_map
    best_result = None
    best_depth  = None
    best_step   = 0
    for i in range(12):
        obs = sim.get_sensor_observations()
        rgb = obs["color_sensor"][:, :, :3]
        depth = obs["depth_sensor"]
        rot_matrix = quaternion.as_rotation_matrix(agent.get_state().rotation)
        occ_map.update(np.array(agent.get_state().position), depth, rot_matrix)
        results = detector(Image.fromarray(rgb), candidate_labels=target_labels)
        for r in results:
            if r["label"] not in target_labels or r["score"] < threshold:
                continue
            if best_result is None or r["score"] > best_result["score"]:
                best_result = r
                best_depth  = depth.copy()
                best_step   = i
        sim.step("turn_right")
    if best_result is not None:
        for _ in range(best_step):
            sim.step("turn_right")
    return best_result, best_depth

def get_frontier_point():
    global visited_frontiers
    agent_pos = np.array(agent.get_state().position)
    frontiers = occ_map.get_frontiers(agent_pos)
    if len(frontiers) < 5:
        for _ in range(100):
            pt = sim.pathfinder.get_random_navigable_point()
            if pt[1] < 2.0:
                continue
            dist = np.linalg.norm(np.array(pt)[[0,2]] - agent_pos[[0,2]])
            frontiers.append((np.array(pt), dist))
    valid = []
    for frontier_pt, dist in frontiers:
        if dist < 1.0 or dist > 10.0:
            continue
        snapped = sim.pathfinder.snap_point(frontier_pt)
        key = (round(snapped[0], 1), round(snapped[2], 1))
        if key in visited_frontiers:
            continue
        y_diff = abs(snapped[1] - agent_pos[1])
        if y_diff > 0.5:
            continue
        path = habitat_sim.ShortestPath()
        path.requested_start = agent_pos
        path.requested_end = snapped
        if sim.pathfinder.find_path(path):
            valid.append((np.array(snapped), dist))
    if not valid:
        return None
    valid.sort(key=lambda x: -x[1])
    pt = valid[0][0]
    visited_frontiers.add((round(pt[0], 1), round(pt[2], 1)))
    return pt

def explore_and_find(target_labels, max_steps=20):
    global occ_map, visited_frontiers
    for step in range(max_steps):
        best, depth = look_around_and_detect(target_labels)
        if best:
            return best, depth
        frontier = get_frontier_point()
        if frontier is None:
            break
        force_navigate(frontier)
        for _ in range(random.randint(1, 4)):
            sim.step("turn_right")
        show_rgb(f"frontier_{step}")
    return None, None

GOAL_LABELS = {
    "沙发": ["sofa", "couch"],
    "sofa": ["sofa", "couch"],
    "床":   ["bed"],
    "bed":  ["bed"],
    "椅子": ["chair"],
    "chair":["chair"],
}

def parse_goal(text):
    for key, labels in GOAL_LABELS.items():
        if key in text:
            return labels
    return None

def run_navigation(user_input):
    global visited_frontiers, status_message, output_frames
    target_labels = parse_goal(user_input)
    if target_labels is None:
        status_message = "不支持该目标，支持：沙发、床、椅子"
        return False
    visited_frontiers = set()
    best_result, best_depth = explore_and_find(target_labels, max_steps=60)
    if best_result is None:
        status_message = "未找到目标"
        return False
    obs = sim.get_sensor_observations()
    rgb = obs["color_sensor"][:, :, :3]
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    box = best_result["box"]
    draw.rectangle([box["xmin"], box["ymin"], box["xmax"], box["ymax"]],
                   outline="red", width=3)
    draw.text((box["xmin"], max(0, box["ymin"]-20)),
              f"{best_result['label']}: {best_result['score']:.2f}", fill="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    output_frames.append(base64.b64encode(buf.getvalue()).decode())
    nav_target, object_pos = get_nav_target(best_result["box"], best_depth)
    success = force_navigate(nav_target)
    if success:
        turn_to_face(object_pos)
        show_rgb("arrived")
        status_message = f"已到达{user_input}旁边！还需要什么？"
        return True
    else:
        status_message = "导航失败"
        return False

if __name__ == "__main__":
    user_input = sys.argv[1]
    init()
    success = run_navigation(user_input)
    # 导航结束后保存当前位置
    save_state()

    # 显式关闭仿真器释放GPU资源
    try:
        sim.close()
    except:
        pass

    result = {
        "success": success,
        "message": status_message,
        "frames": output_frames
    }
    print(json.dumps(result), flush=True)