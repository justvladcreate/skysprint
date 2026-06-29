import json
import math
from config import (
    ROTATION_UP, ROTATION_RETURN, TARGET_ANGLE,
    THRUST_FORCE, GLIDE_FORCE, GRAVITY, DRAG, MAX_SPEED,
    RAY_COUNT, RAY_ANGLES, RAY_MAX_DIST
)

class SkySprintEnv:
    def __init__(self, level_path: str, use_rays=True, training_mode=True):
        with open(level_path, 'r') as f:
            data = json.load(f)
        if 'start' not in data:
            raise KeyError(f"Level file {level_path} missing 'start' field")
        if 'checkpoints' not in data:
            raise KeyError(f"Level file {level_path} missing 'checkpoints' field")

        self.start_pos = data['start']
        self.checkpoints = data['checkpoints']
        self.obstacles = data.get('obstacles', [])
        self.width = data.get('width', 4000)
        self.height = data.get('height', 1080)
        self.use_rays = use_rays
        self.training_mode = training_mode
        self.reset()

    def reset(self):
        self.player1 = self._new_player_state(self.start_pos)
        self.player2 = None
        self.time = 0.0
        return self.get_state(1)

    def add_second_player(self):
        self.player2 = self._new_player_state(self.start_pos)

    def _new_player_state(self, pos):
        return {
            'x': pos[0], 'y': pos[1],
            'vx': 0.0, 'vy': 0.0,
            'angle': TARGET_ANGLE,
            'next_cp': 0,
            'alive': True,
            'deaths': 0          # счётчик смертей
        }

    def get_state(self, player_id: int):
        p = self.player1 if player_id == 1 else self.player2
        if not p:
            return None
        cp = None
        if p['next_cp'] < len(self.checkpoints):
            cp = self.checkpoints[p['next_cp']]
        state = {
            'x': p['x'], 'y': p['y'],
            'vx': p['vx'], 'vy': p['vy'],
            'angle': p['angle'],
            'next_cp': p['next_cp'],
            'cp_x': cp['x'] if cp else None,
            'cp_y': cp['y'] if cp else None,
            'alive': p['alive'],
            'deaths': p['deaths']     # <-- теперь здесь
        }
        if self.use_rays:
            state['rays'] = self._cast_rays(p)
        return state

    def _cast_rays(self, p):
        rays = []
        base_angle = p['angle']
        for da in RAY_ANGLES:
            angle = base_angle + da
            dx = math.cos(angle)
            dy = -math.sin(angle)
            dist = RAY_MAX_DIST
            for obs in self.obstacles:
                t = self._ray_intersects_rect(p['x'], p['y'], dx, dy, obs)
                if t is not None and t < dist:
                    dist = t
            rays.append(dist / RAY_MAX_DIST)
        return rays

    def _ray_intersects_rect(self, x, y, dx, dy, rect):
        rx, ry, rw, rh = rect['x'], rect['y'], rect['w'], rect['h']
        tx1 = (rx - x) / dx if dx != 0 else float('inf')
        tx2 = (rx + rw - x) / dx if dx != 0 else float('inf')
        ty1 = (ry - y) / dy if dy != 0 else float('inf')
        ty2 = (ry + rh - y) / dy if dy != 0 else float('inf')
        t_min = max(min(tx1, tx2), min(ty1, ty2))
        t_max = min(max(tx1, tx2), max(ty1, ty2))
        if t_min <= t_max and t_min >= 0:
            return t_min
        return None

    def step(self, action1: int, action2: int = None):
        self._update_player(self.player1, action1)
        if self.player2 and action2 is not None:
            self._update_player(self.player2, action2)
        self.time += 1 / 60

        done1 = self.player1['next_cp'] >= len(self.checkpoints)
        if self.training_mode and not self.player1['alive']:
            done1 = True

        done2 = False
        if self.player2:
            done2 = self.player2['next_cp'] >= len(self.checkpoints)
            if self.training_mode and not self.player2['alive']:
                done2 = True

        return self.get_state(1), self.get_state(2) if self.player2 else None, done1, done2

    def _update_player(self, p, action):
        if not p['alive']:
            if self.training_mode:
                return
            # В игровом режиме респавн (воскрешение)
            self._respawn_player(p)
            return

        # Вращение
        target = TARGET_ANGLE
        diff = (target - p['angle']) % (2 * math.pi)
        if diff > math.pi:
            diff -= 2 * math.pi

        if action == 1:
            if 0 < diff < math.pi:
                speed = ROTATION_UP + ROTATION_RETURN
            else:
                speed = ROTATION_UP
            p['angle'] += speed
        else:
            if diff < 0:
                speed = ROTATION_RETURN + ROTATION_UP
            else:
                speed = ROTATION_RETURN
            if abs(diff) <= speed:
                p['angle'] = target
            else:
                p['angle'] += math.copysign(speed, diff)

        # Силы
        if action == 1:
            thrust_x = THRUST_FORCE * math.cos(p['angle'])
            thrust_y = -THRUST_FORCE * math.sin(p['angle'])
        else:
            thrust_x = 0.0
            thrust_y = 0.0

        if action == 0:
            glide_x = GLIDE_FORCE * math.cos(p['angle'])
        else:
            glide_x = 0.0

        drag_x = DRAG * p['vx']
        drag_y = DRAG * p['vy']

        ax = thrust_x + glide_x - drag_x
        ay = thrust_y + GRAVITY - drag_y

        p['vx'] += ax
        p['vy'] += ay

        speed = math.hypot(p['vx'], p['vy'])
        if speed > MAX_SPEED:
            p['vx'] = p['vx'] / speed * MAX_SPEED
            p['vy'] = p['vy'] / speed * MAX_SPEED

        p['x'] += p['vx']
        p['y'] += p['vy']

        # Столкновение
        if self._collides_obstacle(p['x'], p['y']):
            if self.training_mode:
                p['alive'] = False
                p['vx'] = 0.0
                p['vy'] = 0.0
            else:
                # Игровой режим: увеличиваем счётчик смертей, затем респавн
                p['deaths'] += 1
                self._respawn_player(p)
            return

        # Чекпоинты
        if p['next_cp'] < len(self.checkpoints):
            cp = self.checkpoints[p['next_cp']]
            if math.hypot(p['x'] - cp['x'], p['y'] - cp['y']) < cp.get('radius', 40):
                p['next_cp'] += 1

    def _respawn_player(self, p):
        if p['next_cp'] > 0:
            cp = self.checkpoints[p['next_cp'] - 1]
            p['x'], p['y'] = cp['x'], cp['y']
        else:
            p['x'], p['y'] = self.start_pos
        p['vx'] = p['vy'] = 0.0
        p['angle'] = TARGET_ANGLE
        p['alive'] = True

    def _collides_obstacle(self, x, y):
        for obs in self.obstacles:
            if obs['x'] <= x <= obs['x'] + obs['w'] and obs['y'] <= y <= obs['y'] + obs['h']:
                return True
        return False