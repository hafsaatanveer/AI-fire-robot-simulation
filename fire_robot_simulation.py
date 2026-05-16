"""
=============================================================================
  AI FIRE DETECTION & EXTINGUISHING ROBOT - SMART CITY SIMULATION
=============================================================================
  Project: Autonomous Fire Response Robot Simulation
  Tech: Python + Pygame
  Features:
    - 2D Smart-City Map with roads, buildings, obstacles
    - Autonomous robot with A* pathfinding
    - Manual & Automatic modes
    - Fire spawning, detection, extinguishing
    - Water level management & refill station
    - Animated fire, water spray, robot
    - Score & status dashboard

  HOW TO RUN (VS Code):
    1. Install Python 3.8+ from python.org
    2. Open terminal in VS Code (Ctrl + `)
    3. Run: pip install pygame
    4. Run: python fire_robot_simulation.py

  CONTROLS:
    M         - Toggle Manual / Automatic mode
    W / Up    - Move Forward (manual)
    S / Down  - Move Backward (manual)
    A / Left  - Turn Left (manual)
    D / Right - Turn Right (manual)
    SPACE     - Spray Water (manual)
    R         - Refill water (when near station, manual)
    ESC       - Quit
=============================================================================
"""

import pygame
import math
import random
import heapq
import sys
from collections import deque

# ─────────────────────────────────────────────
#  CONSTANTS & CONFIGURATION
# ─────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1280, 780
MAP_W, MAP_H       = 960, 720           # left game area
PANEL_W            = SCREEN_W - MAP_W   # right dashboard
FPS                = 60
CELL               = 40                 # grid cell size (pixels)
COLS               = MAP_W // CELL      # 24
ROWS               = MAP_H // CELL      # 18

# Colours
C_BG          = (15,  17,  26)
C_ROAD        = (35,  40,  55)
C_ROAD_MARK   = (60,  65,  80)
C_GRASS       = (28,  48,  30)
C_BUILDING    = (50,  55,  80)
C_BUILDING2   = (65,  45,  70)
C_BUILDING3   = (40,  60,  80)
C_ROOF        = (80,  85, 110)
C_PANEL       = (12,  14,  22)
C_PANEL_BORDER= (40,  80, 140)
C_TEXT        = (200, 210, 230)
C_TEXT_DIM    = (100, 110, 140)
C_ACCENT      = ( 60, 180, 255)
C_GREEN       = ( 50, 220, 100)
C_RED         = (255,  60,  60)
C_ORANGE      = (255, 140,  30)
C_YELLOW      = (255, 230,  50)
C_WATER       = ( 40, 160, 255)
C_WATER_LOW   = (255, 140,  30)
C_STATION     = ( 30, 180, 200)
C_WHITE       = (255, 255, 255)
C_ROBOT_BODY  = ( 70, 200, 255)
C_ROBOT_WHEEL = ( 40,  50,  70)
C_FIRE1       = (255,  80,  20)
C_FIRE2       = (255, 200,  30)
C_FIRE3       = (255, 255, 120)
C_SMOKE       = ( 80,  80,  90)

# Grid cell types
ROAD     = 0
BUILDING = 1
GRASS    = 2
OBSTACLE = 3
WATER_ST = 4


# ─────────────────────────────────────────────
#  MAP LAYOUT  (0=road, 1=building, 2=grass,
#               3=obstacle, 4=water station)
# ─────────────────────────────────────────────
def build_map():
    """
    Build a 24×18 city grid by hand with roads forming a grid network.
    Buildings fill blocks between roads.
    """
    grid = [[BUILDING] * COLS for _ in range(ROWS)]

    # Horizontal road rows
    h_roads = [0, 1, 5, 6, 10, 11, 15, 16, ROWS-2, ROWS-1]
    # Vertical road columns
    v_roads = [0, 1, 6, 7, 12, 13, 18, 19, COLS-2, COLS-1]

    for r in range(ROWS):
        for c in range(COLS):
            if r in h_roads or c in v_roads:
                grid[r][c] = ROAD

    # Grass patches inside some blocks
    grass_blocks = [
        (2, 2, 4, 5),   (2, 8, 4, 11),  (2, 14, 4, 17),
        (7, 2, 9, 5),   (7, 14, 9, 17),
        (12, 8, 14, 11),(12, 14, 14, 17),
    ]
    for (r1, c1, r2, c2) in grass_blocks:
        for r in range(r1, r2+1):
            for c in range(c1, c2+1):
                grid[r][c] = GRASS

    # Obstacles (barrels, barriers)
    obstacles = [
        (3, 3),(3, 4),(8, 3),(8, 4),(13, 3),(13, 4),
        (3, 9),(3, 10),(8, 16),(8, 17),
        (12, 2),(12, 4),(4, 15),(4, 16),
    ]
    for (r, c) in obstacles:
        grid[r][c] = OBSTACLE

    # Water refill station
    grid[2][20] = WATER_ST
    grid[2][21] = WATER_ST

    return grid


# ─────────────────────────────────────────────
#  A* PATHFINDING
# ─────────────────────────────────────────────
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    """
    A* on the grid. Passable cells = ROAD or GRASS or WATER_ST.
    Returns list of (row, col) cells or empty list if no path.
    """
    passable = {ROAD, GRASS, WATER_ST}
    rows, cols = len(grid), len(grid[0])
    open_heap = []
    heapq.heappush(open_heap, (0, start))
    came_from = {start: None}
    g_score   = {start: 0}

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (r+dr, c+dc)
            nr, nc = nb
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr][nc] in passable:
                    ng = g_score[current] + 1
                    if nb not in g_score or ng < g_score[nb]:
                        g_score[nb] = ng
                        f = ng + heuristic(nb, goal)
                        heapq.heappush(open_heap, (f, nb))
                        came_from[nb] = current
    return []   # no path found


# ─────────────────────────────────────────────
#  PARTICLE SYSTEMS
# ─────────────────────────────────────────────
class Particle:
    def __init__(self, x, y, vx, vy, color, life, size, gravity=0):
        self.x, self.y  = x, y
        self.vx, self.vy = vx, vy
        self.color      = color
        self.life       = life
        self.max_life   = life
        self.size       = size
        self.gravity    = gravity

    def update(self):
        self.x  += self.vx
        self.y  += self.vy
        self.vy += self.gravity
        self.life -= 1
        return self.life > 0

    def draw(self, surf):
        alpha = int(255 * (self.life / self.max_life))
        s = max(1, int(self.size * self.life / self.max_life))
        r, g, b = self.color
        c = (min(255,r), min(255,g), min(255,b))
        pygame.draw.circle(surf, c, (int(self.x), int(self.y)), s)


class ParticleSystem:
    def __init__(self):
        self.particles = []

    def emit(self, x, y, vx, vy, color, life, size, gravity=0, count=1):
        for _ in range(count):
            jx = vx + random.uniform(-0.5, 0.5)
            jy = vy + random.uniform(-0.5, 0.5)
            self.particles.append(Particle(x, y, jx, jy, color, life, size, gravity))

    def update(self):
        self.particles = [p for p in self.particles if p.update()]

    def draw(self, surf):
        for p in self.particles:
            p.draw(surf)


# ─────────────────────────────────────────────
#  FIRE CLASS
# ─────────────────────────────────────────────
class Fire:
    def __init__(self, row, col):
        self.row, self.col = row, col
        self.x = col * CELL + CELL // 2
        self.y = row * CELL + CELL // 2
        self.intensity = 100   # 100 = full fire, 0 = extinguished
        self.flicker   = 0
        self.particles = ParticleSystem()
        self.smoke_timer = 0
        self.alive     = True

    def update(self):
        self.flicker += 1
        # Emit fire particles
        if self.flicker % 2 == 0 and self.intensity > 0:
            scale = self.intensity / 100
            # Core flame
            self.particles.emit(
                self.x + random.uniform(-6,6)*scale,
                self.y + random.uniform(-4,4)*scale,
                random.uniform(-0.4,0.4), random.uniform(-2.5,-0.8)*scale,
                C_FIRE1, int(22*scale)+5, int(8*scale)+3, gravity=-0.05
            )
            # Bright core
            self.particles.emit(
                self.x + random.uniform(-3,3)*scale,
                self.y + random.uniform(-2,2)*scale,
                random.uniform(-0.2,0.2), random.uniform(-2.0,-1.0)*scale,
                C_FIRE2, int(15*scale)+3, int(5*scale)+2, gravity=-0.03
            )
            # Tip
            self.particles.emit(
                self.x, self.y - 8*scale,
                random.uniform(-0.2,0.2), random.uniform(-1.5,-0.5)*scale,
                C_FIRE3, int(10*scale)+2, int(3*scale)+1
            )
        # Smoke
        self.smoke_timer += 1
        if self.smoke_timer % 6 == 0 and self.intensity > 0:
            scale = self.intensity / 100
            self.particles.emit(
                self.x + random.uniform(-5,5),
                self.y - 20*scale,
                random.uniform(-0.3,0.3), random.uniform(-0.8,-0.2),
                C_SMOKE, 40, int(6*scale)+4, gravity=-0.01
            )
        self.particles.update()
        if self.intensity <= 0:
            self.alive = False

    def draw(self, surf):
        self.particles.draw(surf)
        # Glow underneath
        if self.intensity > 0:
            glow_s = pygame.Surface((CELL*2, CELL*2), pygame.SRCALPHA)
            alpha  = int(80 * self.intensity / 100)
            pygame.draw.circle(glow_s, (*C_FIRE1, alpha), (CELL, CELL), CELL)
            surf.blit(glow_s, (self.x - CELL, self.y - CELL))

    def extinguish(self, amount):
        self.intensity = max(0, self.intensity - amount)


# ─────────────────────────────────────────────
#  ROBOT CLASS
# ─────────────────────────────────────────────
class Robot:
    SPEED     = 3.0   # px per frame
    TURN_SPD  = 4.0   # degrees per frame
    SPRAY_RANGE = CELL * 2.2
    SAFE_DIST   = CELL * 1.8

    STATUS_SEARCH    = "Searching"
    STATUS_DETECTED  = "Fire Detected!"
    STATUS_MOVING    = "Moving to Fire"
    STATUS_EXTINGUISH= "Extinguishing"
    STATUS_REFILL    = "Going to Refill"
    STATUS_MANUAL    = "Manual Control"

    def __init__(self, row, col):
        self.x       = col * CELL + CELL // 2
        self.y       = row * CELL + CELL // 2
        self.angle   = 0        # degrees, 0=right
        self.water   = 100.0    # percent
        self.status  = self.STATUS_SEARCH
        self.path    = []
        self.target_fire   = None
        self.target_refill = None
        self.spray_active  = False
        self.spray_timer   = 0
        self.wheel_anim    = 0
        self.particles     = ParticleSystem()
      # patrol
        self.patrol_pts    = []
        self.patrol_idx    = 0
        self.stuck_timer   = 0
        self.prev_pos      = (self.x, self.y)
        # stuck-detection (NEW)
        self.last_cell     = (-1, -1)
        self.path_retry    = 0

    def cell(self):
        return (int(self.y // CELL), int(self.x // CELL))

    def move_toward_path(self):
        """Move one step along current path."""
        if not self.path:
            return
        tr, tc = self.path[0]
        tx = tc * CELL + CELL // 2
        ty = tr * CELL + CELL // 2
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < self.SPEED + 1:
            self.x, self.y = tx, ty
            self.path.pop(0)
        else:
            target_angle = math.degrees(math.atan2(-dy, dx))
            da = (target_angle - self.angle + 540) % 360 - 180
            if abs(da) > 3:
                self.angle += max(-self.TURN_SPD*2, min(self.TURN_SPD*2, da))
            else:
                self.angle = target_angle
                self.x += (dx/dist)*self.SPEED
                self.y += (dy/dist)*self.SPEED
        self.wheel_anim += 3

    def dist_to(self, other_x, other_y):
        return math.hypot(self.x - other_x, self.y - other_y)

    def spray_water(self, fires):
        if self.water <= 0:
            return
        self.spray_active = True
        self.spray_timer  = 15
        self.water = max(0, self.water - 0.5)
        # shoot spray particles
        angle_r = math.radians(self.angle)
        for _ in range(4):
            sp = random.uniform(3, 6)
            sa = angle_r + random.uniform(-0.3, 0.3)
            self.particles.emit(
                self.x + math.cos(angle_r)*18,
                self.y - math.sin(angle_r)*18,
                math.cos(sa)*sp, -math.sin(sa)*sp,
                C_WATER, 20, 4, gravity=0.15
            )
        # extinguish nearby fires
        for f in fires:
            if self.dist_to(f.x, f.y) < self.SPRAY_RANGE:
                f.extinguish(1.5)

    def draw(self, surf):
        # Draw water spray particles
        self.particles.update()
        self.particles.draw(surf)

        cx, cy = int(self.x), int(self.y)
        angle  = self.angle

        # Shadow
        shadow_s = pygame.Surface((52, 28), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_s, (0,0,0,80), (0,0,52,28))
        surf.blit(shadow_s, (cx-26, cy+14))

        # Body (rotated rectangle) — draw manually
        body_pts = self._rotated_rect(cx, cy, 34, 22, angle)
        pygame.draw.polygon(surf, C_ROBOT_BODY, body_pts)
        pygame.draw.polygon(surf, C_ACCENT, body_pts, 2)

        # Wheels (4 corners)
        for wx, wy in [(-14,-12),(-14,12),(14,-12),(14,12)]:
            rwx = cx + wx*math.cos(math.radians(angle)) - wy*math.sin(math.radians(angle))
            rwy = cy - wx*math.sin(math.radians(angle)) - wy*math.cos(math.radians(angle))
            wheel_pts = self._rotated_rect(int(rwx), int(rwy), 10, 6, angle)
            pygame.draw.polygon(surf, C_ROBOT_WHEEL, wheel_pts)

        # Nozzle direction indicator
        nx = cx + math.cos(math.radians(angle)) * 20
        ny = cy - math.sin(math.radians(angle)) * 20
        pygame.draw.line(surf, C_WATER, (cx,cy), (int(nx),int(ny)), 3)
        pygame.draw.circle(surf, C_WATER, (int(nx),int(ny)), 4)

        # Status dot
        status_colors = {
            self.STATUS_SEARCH:    C_GREEN,
            self.STATUS_DETECTED:  C_YELLOW,
            self.STATUS_MOVING:    C_ACCENT,
            self.STATUS_EXTINGUISH:C_ORANGE,
            self.STATUS_REFILL:    C_STATION,
            self.STATUS_MANUAL:    C_WHITE,
        }
        dot_c = status_colors.get(self.status, C_WHITE)
        pygame.draw.circle(surf, dot_c, (cx, cy-16), 5)
        pygame.draw.circle(surf, C_WHITE, (cx, cy-16), 5, 1)

        # Spray flash
        if self.spray_timer > 0:
            self.spray_timer -= 1
            for i in range(5):
                sa = math.radians(angle) + random.uniform(-0.4,0.4)
                dist_s = random.uniform(20, 45)
                sx = cx + math.cos(sa)*dist_s
                sy = cy - math.sin(sa)*dist_s
                pygame.draw.circle(surf, (*C_WATER, 180), (int(sx),int(sy)), random.randint(2,5))

    def _rotated_rect(self, cx, cy, w, h, angle_deg):
        """Return 4 corner points of a rotated rectangle."""
        a = math.radians(angle_deg)
        cos_a, sin_a = math.cos(a), math.sin(a)
        corners = [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
        return [
            (int(cx + x*cos_a - y*sin_a),
             int(cy - x*sin_a - y*cos_a))   # note: y-axis flipped
            for x, y in corners
        ]


# ─────────────────────────────────────────────
#  BUILDING DRAW HELPER
# ─────────────────────────────────────────────
BUILDING_COLORS = [C_BUILDING, C_BUILDING2, C_BUILDING3]
building_color_map = {}

def get_building_color(r, c):
    if (r, c) not in building_color_map:
        building_color_map[(r,c)] = random.choice(BUILDING_COLORS)
    return building_color_map[(r,c)]


def draw_map(surf, grid):
    for r in range(ROWS):
        for c in range(COLS):
            x, y = c*CELL, r*CELL
            cell_type = grid[r][c]

            if cell_type == ROAD:
                pygame.draw.rect(surf, C_ROAD, (x, y, CELL, CELL))
                # Road markings
                if r % 5 in (0,1) or r % 5 in (3,4):
                    pygame.draw.line(surf, C_ROAD_MARK, (x+CELL//2, y), (x+CELL//2, y+CELL), 1)
                if c % 6 in (0,1) or c % 6 in (4,5):
                    pygame.draw.line(surf, C_ROAD_MARK, (x, y+CELL//2), (x+CELL, y+CELL//2), 1)

            elif cell_type == GRASS:
                pygame.draw.rect(surf, C_GRASS, (x, y, CELL, CELL))
                # Grass texture dots
                for _ in range(3):
                    gx = x + random.randint(4, CELL-4)
                    gy = y + random.randint(4, CELL-4)
                    pygame.draw.circle(surf, (35,58,35), (gx,gy), 2)

            elif cell_type == BUILDING:
                bc = get_building_color(r, c)
                pygame.draw.rect(surf, bc, (x, y, CELL, CELL))
                # Roof accent
                pygame.draw.rect(surf, C_ROOF, (x+4, y+4, CELL-8, CELL-8))
                # Windows
                for wy in range(y+6, y+CELL-6, 9):
                    for wx in range(x+6, x+CELL-6, 9):
                        wc = (200,220,255) if random.random() > 0.3 else (60,70,90)
                        pygame.draw.rect(surf, wc, (wx, wy, 4, 4))
                pygame.draw.rect(surf, (20,20,35), (x,y,CELL,CELL), 1)

            elif cell_type == OBSTACLE:
                pygame.draw.rect(surf, C_ROAD, (x, y, CELL, CELL))
                # Barrel/barrier visual
                pygame.draw.rect(surf, (180,120,40), (x+8, y+8, CELL-16, CELL-16))
                pygame.draw.rect(surf, (220,160,60), (x+10, y+12, CELL-20, 6))
                pygame.draw.rect(surf, (220,160,60), (x+10, y+22, CELL-20, 6))
                pygame.draw.rect(surf, (100,60,20), (x+8, y+8, CELL-16, CELL-16), 2)

            elif cell_type == WATER_ST:
                pygame.draw.rect(surf, C_ROAD, (x, y, CELL, CELL))
                # Water station box
                pygame.draw.rect(surf, C_STATION, (x+4, y+4, CELL-8, CELL-8))
                pygame.draw.rect(surf, (20,220,240), (x+4, y+4, CELL-8, CELL-8), 2)
                # Water drop icon
                mid_x, mid_y = x+CELL//2, y+CELL//2
                pygame.draw.circle(surf, (200,240,255), (mid_x, mid_y+2), 7)
                pygame.draw.polygon(surf, (200,240,255), [
                    (mid_x, mid_y-10),(mid_x-6, mid_y),(mid_x+6, mid_y)
                ])


# ─────────────────────────────────────────────
#  DASHBOARD / PANEL DRAW
# ─────────────────────────────────────────────
class Dashboard:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.font_big   = None
        self.font_med   = None
        self.font_sm    = None
        self.font_title = None

    def init_fonts(self):
        self.font_title = pygame.font.SysFont("consolas", 20, bold=True)
        self.font_big   = pygame.font.SysFont("consolas", 28, bold=True)
        self.font_med   = pygame.font.SysFont("consolas", 15, bold=True)
        self.font_sm    = pygame.font.SysFont("consolas", 13)

    def draw(self, surf, robot, fires, score, mode, fire_count, elapsed):
        px, py = self.x, self.y

        # Panel background
        pygame.draw.rect(surf, C_PANEL, (px, py, self.w, self.h))
        pygame.draw.rect(surf, C_PANEL_BORDER, (px, py, self.w, self.h), 2)
        pygame.draw.line(surf, C_PANEL_BORDER, (px+2, py+2), (px+self.w-3, py+2), 1)

        y = py + 16

        # Title
        title = self.font_title.render("🤖 FIRE ROBOT AI", True, C_ACCENT)
        surf.blit(title, (px + self.w//2 - title.get_width()//2, y))
        y += 30
        sub = self.font_sm.render("Smart City Response System", True, C_TEXT_DIM)
        surf.blit(sub, (px + self.w//2 - sub.get_width()//2, y))
        y += 24

        pygame.draw.line(surf, C_PANEL_BORDER, (px+10, y), (px+self.w-10, y), 1)
        y += 12

        # Mode
        mode_text = "AUTOMATIC" if mode == "auto" else "MANUAL"
        mode_color = C_GREEN if mode == "auto" else C_YELLOW
        self._label(surf, "MODE", px, y)
        mc = self.font_big.render(mode_text, True, mode_color)
        surf.blit(mc, (px + self.w//2 - mc.get_width()//2, y+16))
        y += 52

        pygame.draw.line(surf, C_PANEL_BORDER, (px+10, y), (px+self.w-10, y), 1)
        y += 12

        # Status
        self._label(surf, "STATUS", px, y)
        y += 18
        st_color = {
            Robot.STATUS_SEARCH:    C_GREEN,
            Robot.STATUS_DETECTED:  C_YELLOW,
            Robot.STATUS_MOVING:    C_ACCENT,
            Robot.STATUS_EXTINGUISH:C_ORANGE,
            Robot.STATUS_REFILL:    C_STATION,
            Robot.STATUS_MANUAL:    C_WHITE,
        }.get(robot.status, C_WHITE)
        st = self.font_med.render(robot.status, True, st_color)
        surf.blit(st, (px + self.w//2 - st.get_width()//2, y))
        y += 32

        # Score
        pygame.draw.line(surf, C_PANEL_BORDER, (px+10, y), (px+self.w-10, y), 1)
        y += 10
        self._label(surf, "SCORE", px, y)
        sc = self.font_big.render(str(score), True, C_YELLOW)
        surf.blit(sc, (px + self.w//2 - sc.get_width()//2, y+14))
        y += 50

        # Fires extinguished
        self._stat_row(surf, "Fires Extinguished", str(score // 10), px, y, C_ORANGE)
        y += 26

        # Active fires
        fc_color = C_RED if fire_count > 2 else (C_YELLOW if fire_count > 0 else C_GREEN)
        self._stat_row(surf, "Active Fires", str(fire_count), px, y, fc_color)
        y += 26

        # Time elapsed
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        self._stat_row(surf, "Time", f"{mins:02d}:{secs:02d}", px, y, C_TEXT)
        y += 36

        # Water level bar
        pygame.draw.line(surf, C_PANEL_BORDER, (px+10, y), (px+self.w-10, y), 1)
        y += 12
        self._label(surf, "WATER LEVEL", px, y)
        y += 18
        bar_w = self.w - 30
        bar_h = 18
        bx = px + 15
        pygame.draw.rect(surf, (20,30,50), (bx, y, bar_w, bar_h), border_radius=5)
        fill = int(bar_w * robot.water / 100)
        wc = C_WATER if robot.water > 30 else C_WATER_LOW
        if fill > 0:
            pygame.draw.rect(surf, wc, (bx, y, fill, bar_h), border_radius=5)
        pygame.draw.rect(surf, C_PANEL_BORDER, (bx, y, bar_w, bar_h), 1, border_radius=5)
        wt = self.font_sm.render(f"{int(robot.water)}%", True, C_WHITE)
        surf.blit(wt, (bx + bar_w//2 - wt.get_width()//2, y+2))
        y += 36

        # Robot position
        r, c = robot.cell()
        self._stat_row(surf, "Position", f"({c},{r})", px, y, C_TEXT_DIM)
        y += 26

        pygame.draw.line(surf, C_PANEL_BORDER, (px+10, y), (px+self.w-10, y), 1)
        y += 14

        # Controls cheatsheet
        self._label(surf, "CONTROLS", px, y)
        y += 18
        controls = [
            ("M",     "Auto/Manual"),
            ("WASD",  "Move (manual)"),
            ("SPACE", "Spray water"),
            ("R",     "Refill water"),
            ("ESC",   "Quit"),
        ]
        for key, desc in controls:
            k  = self.font_sm.render(f"[{key}]", True, C_ACCENT)
            dv = self.font_sm.render(desc, True, C_TEXT_DIM)
            surf.blit(k,  (px+14, y))
            surf.blit(dv, (px+14+k.get_width()+6, y))
            y += 17

        y += 8
        pygame.draw.line(surf, C_PANEL_BORDER, (px+10, y), (px+self.w-10, y), 1)
        y += 10

        # Warning if water low
        if robot.water < 20:
            warn = self.font_med.render("⚠ LOW WATER!", True, C_RED)
            surf.blit(warn, (px + self.w//2 - warn.get_width()//2, y))
            y += 22
        if fire_count >= 3:
            warn2 = self.font_med.render("🔥 MULTIPLE FIRES!", True, C_ORANGE)
            surf.blit(warn2, (px + self.w//2 - warn2.get_width()//2, y))

    def _label(self, surf, text, px, y):
        t = self.font_sm.render(text, True, C_TEXT_DIM)
        surf.blit(t, (px + self.w//2 - t.get_width()//2, y))

    def _stat_row(self, surf, label, value, px, y, val_color):
        l = self.font_sm.render(label + ":", True, C_TEXT_DIM)
        v = self.font_med.render(value, True, val_color)
        surf.blit(l, (px + 14, y))
        surf.blit(v, (px + self.w - 14 - v.get_width(), y))


# ─────────────────────────────────────────────
#  MAIN GAME CLASS
# ─────────────────────────────────────────────
class FireRobotSim:
    FIRE_SPAWN_INTERVAL = 8    # seconds between new fires
    MAX_FIRES           = 5

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("AI Fire Detection & Extinguishing Robot — Smart City Simulation")
        self.clock  = pygame.time.Clock()

        # Pre-seed random for reproducible building colors
        random.seed(42)
        self.grid       = build_map()
        random.seed()   # restore true randomness

        self.robot      = Robot(8, 1)
        self.fires      = []
        self.score      = 0
        self.mode       = "auto"
        self.dashboard  = Dashboard(MAP_W, 0, PANEL_W, SCREEN_H)
        self.dashboard.init_fonts()

        self.fire_timer = 0
        self.start_time = pygame.time.get_ticks()
        self.font_notif = pygame.font.SysFont("consolas", 16, bold=True)
        self.notifications = []   # [(text, color, timer)]

        # Pre-build patrol path for auto mode
        self._build_patrol_points()

        # Spawn initial fires
        for _ in range(2):
            self._spawn_fire()

        # Map surface (static, redraw only when needed)
        self.map_surf = pygame.Surface((MAP_W, MAP_H))
        self._redraw_map()

    def _redraw_map(self):
        self.map_surf.fill(C_BG)
        draw_map(self.map_surf, self.grid)

    def _build_patrol_points(self):
        """Pick a set of road cells as patrol waypoints."""
        road_cells = [(r, c) for r in range(ROWS) for c in range(COLS)
                      if self.grid[r][c] == ROAD]
        # Sample evenly spaced points
        random.shuffle(road_cells)
        self.robot.patrol_pts = road_cells[:12]
        self.robot.patrol_idx = 0

    def _spawn_fire(self):
        if len(self.fires) >= self.MAX_FIRES:
            return
        # Find a valid spawn cell (grass or obstacle area, not road, not near robot)
        attempts = 0
        while attempts < 100:
            r = random.randint(2, ROWS-3)
            c = random.randint(2, COLS-3)
            cell_t = self.grid[r][c]
            if cell_t in (GRASS, OBSTACLE, BUILDING):
                fx = c*CELL + CELL//2
                fy = r*CELL + CELL//2
                # Not too close to robot
                if math.hypot(fx-self.robot.x, fy-self.robot.y) > CELL*3:
                    # Not duplicate
                    dup = any(f.row==r and f.col==c for f in self.fires)
                    if not dup:
                        self.fires.append(Fire(r, c))
                        self.notify(f"🔥 Fire detected at ({c},{r})!", C_RED)
                        return
            attempts += 1

    def notify(self, text, color=C_TEXT):
        self.notifications.append([text, color, 180])  # 3 sec at 60fps

    def _find_road_near(self, row, col):
        """BFS to find nearest road cell to (row,col)."""
        visited = set()
        q = deque([(row, col, 0)])
        while q:
            r, c, d = q.popleft()
            if (r,c) in visited:
                continue
            visited.add((r,c))
            if self.grid[r][c] == ROAD and d > 0:
                return (r, c)
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < ROWS and 0 <= nc < COLS and (nr,nc) not in visited:
                    q.append((nr, nc, d+1))
        return None

    # ── AUTO LOGIC ──────────────────────────────
    def auto_update(self):
        bot   = self.robot
        fires = [f for f in self.fires if f.alive]

        # ── Tick path-retry cooldown ──────────────────────────
        if bot.path_retry > 0:
            bot.path_retry -= 1

        # If the robot hasn't moved to a new grid cell in 90 frames
        # (~1.5 sec), wipe the path and force a re-plan.
        cur_cell = bot.cell()
        if cur_cell == bot.last_cell:
            bot.stuck_timer += 1
        else:
            bot.stuck_timer = 0
            bot.last_cell   = cur_cell

        if bot.stuck_timer > 90 and bot.status not in (bot.STATUS_EXTINGUISH, bot.STATUS_MANUAL):
            bot.path        = []
            bot.stuck_timer = 0
            bot.path_retry  = 30          # wait 30 frames before re-planning
            if bot.status != bot.STATUS_REFILL:
                bot.status      = bot.STATUS_SEARCH
                bot.target_fire = None

        # ── Low water → go refill ─────────────────────────────
        if bot.water < 15 and bot.status != bot.STATUS_REFILL:
            bot.status      = bot.STATUS_REFILL
            bot.target_fire = None
            bot.path        = []
            best_path = []
            for r in range(ROWS):
                for c in range(COLS):
                    if self.grid[r][c] == WATER_ST:
                        nr = self._find_road_near(r, c)
                        if nr:
                            p = astar(self.grid, bot.cell(), nr)
                            if p and (not best_path or len(p) < len(best_path)):
                                best_path = p
            bot.path = best_path

        if bot.status == bot.STATUS_REFILL:
            if bot.path:
                bot.move_toward_path()
            else:
                # Check if close enough to refill
                refilled = False
                for r in range(ROWS):
                    for c in range(COLS):
                        if self.grid[r][c] == WATER_ST:
                            sx = c * CELL + CELL // 2
                            sy = r * CELL + CELL // 2
                            if bot.dist_to(sx, sy) < CELL * 2.5:
                                bot.water  = 100
                                bot.status = bot.STATUS_SEARCH
                                self.notify("💧 Water refilled!", C_WATER)
                                refilled   = True
                                break
                    if refilled:
                        break
                # Path gone but not close enough → retry after cooldown
                if not refilled and bot.path_retry == 0:
                    for r in range(ROWS):
                        for c in range(COLS):
                            if self.grid[r][c] == WATER_ST:
                                nr = self._find_road_near(r, c)
                                if nr:
                                    bot.path = astar(self.grid, bot.cell(), nr)
                                    if bot.path:
                                        break
                        if bot.path:
                            break
                    bot.path_retry = 60
            return

        # ── Fire pursuit ──────────────────────────────────────
        if fires:
            # Pick / re-pick nearest fire if we have no valid target
            if (bot.status == bot.STATUS_SEARCH or
                    bot.target_fire is None or
                    not bot.target_fire.alive):
                bot.target_fire = min(fires, key=lambda f: bot.dist_to(f.x, f.y))
                bot.status      = bot.STATUS_DETECTED
                bot.path        = []          # force re-plan for new target
                self.notify(f"🎯 Targeting fire at ({bot.target_fire.col},{bot.target_fire.row})", C_YELLOW)

            if bot.target_fire and bot.target_fire.alive:
                dist = bot.dist_to(bot.target_fire.x, bot.target_fire.y)

                if dist > bot.SAFE_DIST:
                    # Need to get closer — re-plan if no path and cooldown done
                    if not bot.path and bot.path_retry == 0:
                        nr = self._find_road_near(bot.target_fire.row, bot.target_fire.col)
                        if nr:
                            new_path = astar(self.grid, bot.cell(), nr)
                            if new_path:
                                bot.path = new_path
                            else:
                                # A* failed — search wider area around fire
                                for sr in range(max(0, bot.target_fire.row - 3),
                                               min(ROWS, bot.target_fire.row + 4)):
                                    for sc in range(max(0, bot.target_fire.col - 3),
                                                    min(COLS, bot.target_fire.col + 4)):
                                        if self.grid[sr][sc] == ROAD:
                                            p = astar(self.grid, bot.cell(), (sr, sc))
                                            if p:
                                                bot.path = p
                                                break
                                    if bot.path:
                                        break
                                bot.path_retry = 45   # back-off
                        bot.status = bot.STATUS_MOVING

                    if bot.path:
                        bot.move_toward_path()
                    else:
                        # No path reachable — spray from extended range if possible
                        if dist < bot.SPRAY_RANGE * 1.5:
                            bot.status = bot.STATUS_EXTINGUISH
                            dx = bot.target_fire.x - bot.x
                            dy = bot.target_fire.y - bot.y
                            bot.angle = math.degrees(math.atan2(-dy, dx))
                            bot.spray_water(self.fires)
                else:
                    # Close enough — stop and spray
                    bot.path   = []
                    bot.status = bot.STATUS_EXTINGUISH
                    dx = bot.target_fire.x - bot.x
                    dy = bot.target_fire.y - bot.y
                    bot.angle  = math.degrees(math.atan2(-dy, dx))
                    bot.spray_water(self.fires)
            else:
                bot.target_fire = None
                bot.status      = bot.STATUS_SEARCH
                bot.path        = []

        else:
            # ── Patrol (no fires) ─────────────────────────────
            bot.status = bot.STATUS_SEARCH
            if not bot.path and bot.path_retry == 0 and bot.patrol_pts:
                pt = bot.patrol_pts[bot.patrol_idx % len(bot.patrol_pts)]
                new_path = astar(self.grid, bot.cell(), pt)
                if new_path:
                    bot.path = new_path
                else:
                    bot.patrol_idx += 1   # skip unreachable waypoint
                    bot.path_retry  = 20
                bot.patrol_idx += 1
            bot.move_toward_path()

    # ── MANUAL LOGIC ────────────────────────────
    def manual_update(self, keys):
        bot = self.robot
        bot.status = bot.STATUS_MANUAL
        spd = bot.SPEED
        turn = bot.TURN_SPD

        if keys[pygame.K_LEFT]  or keys[pygame.K_a]:
            bot.angle += turn
            bot.wheel_anim += 2
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            bot.angle -= turn
            bot.wheel_anim += 2
        if keys[pygame.K_UP]    or keys[pygame.K_w]:
            nx = bot.x + math.cos(math.radians(bot.angle))*spd
            ny = bot.y - math.sin(math.radians(bot.angle))*spd
            if self._valid_pos(nx, ny):
                bot.x, bot.y = nx, ny
            bot.wheel_anim += 3
        if keys[pygame.K_DOWN]  or keys[pygame.K_s]:
            nx = bot.x - math.cos(math.radians(bot.angle))*spd
            ny = bot.y + math.sin(math.radians(bot.angle))*spd
            if self._valid_pos(nx, ny):
                bot.x, bot.y = nx, ny
            bot.wheel_anim += 3
        if keys[pygame.K_SPACE]:
            bot.spray_water(self.fires)
        if keys[pygame.K_r]:
            # Manual refill near station
            for r in range(ROWS):
                for c in range(COLS):
                    if self.grid[r][c] == WATER_ST:
                        sx = c*CELL + CELL//2
                        sy = r*CELL + CELL//2
                        if bot.dist_to(sx, sy) < CELL*2:
                            bot.water = 100
                            self.notify("💧 Water refilled!", C_WATER)

    def _valid_pos(self, x, y):
        """Check if pixel position is on a passable cell."""
        c = int(x // CELL)
        r = int(y // CELL)
        if 0 <= r < ROWS and 0 <= c < COLS:
            return self.grid[r][c] in {ROAD, GRASS, WATER_ST}
        return False

    # ── MAIN LOOP ────────────────────────────────
    def run(self):
        running = True
        while running:
            dt = self.clock.tick(FPS) / 1000.0
            elapsed = (pygame.time.get_ticks() - self.start_time) / 1000.0

            # ── Events ──
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_m:
                        self.mode = "manual" if self.mode == "auto" else "auto"
                        self.robot.path = []
                        self.robot.target_fire = None
                        self.notify(f"Mode: {'AUTOMATIC' if self.mode=='auto' else 'MANUAL'}", C_ACCENT)

            keys = pygame.key.get_pressed()

            # ── Fire timer ──
            self.fire_timer += dt
            if self.fire_timer >= self.FIRE_SPAWN_INTERVAL:
                self.fire_timer = 0
                self._spawn_fire()

            # ── Update fires ──
            for f in self.fires:
                f.update()

            # ── Remove extinguished fires ──
            newly_dead = [f for f in self.fires if not f.alive]
            for f in newly_dead:
                self.score += 10
                self.notify(f"✅ Fire extinguished! +10 pts", C_GREEN)
            self.fires = [f for f in self.fires if f.alive]

            # ── Robot update ──
            if self.mode == "auto":
                self.auto_update()
            else:
                self.manual_update(keys)

            # ── Update notifications ──
            self.notifications = [[t, c, n-1] for t,c,n in self.notifications if n > 1]

            # ── Draw ──
            self.screen.fill(C_BG)

            # Map layer
            self.screen.blit(self.map_surf, (0, 0))

            # Grid overlay (subtle)
            for r in range(ROWS+1):
                pygame.draw.line(self.screen, (25,28,40), (0, r*CELL), (MAP_W, r*CELL), 1)
            for c in range(COLS+1):
                pygame.draw.line(self.screen, (25,28,40), (c*CELL, 0), (c*CELL, MAP_H), 1)

            # Fires
            for f in self.fires:
                f.draw(self.screen)

            # Robot
            self.robot.draw(self.screen)

            # Water station glow
            for r in range(ROWS):
                for c in range(COLS):
                    if self.grid[r][c] == WATER_ST:
                        gx, gy = c*CELL+CELL//2, r*CELL+CELL//2
                        pulse = 0.5 + 0.5*math.sin(elapsed*3)
                        g_surf = pygame.Surface((CELL*3, CELL*3), pygame.SRCALPHA)
                        pygame.draw.circle(g_surf, (*C_STATION, int(40*pulse)), (CELL*3//2, CELL*3//2), CELL)
                        self.screen.blit(g_surf, (gx-CELL*3//2, gy-CELL*3//2))

            # Notifications
            ny_start = MAP_H - 30
            for txt, col, timer in reversed(self.notifications[-4:]):
                alpha = min(255, timer*2)
                n_surf = self.font_notif.render(txt, True, col)
                self.screen.blit(n_surf, (10, ny_start))
                ny_start -= 22

            # Dashboard
            active_fires = len([f for f in self.fires if f.alive])
            self.dashboard.draw(
                self.screen, self.robot, self.fires,
                self.score, self.mode, active_fires, elapsed
            )

            # Map border
            pygame.draw.rect(self.screen, C_PANEL_BORDER, (0, 0, MAP_W, MAP_H), 2)

            pygame.display.flip()

        pygame.quit()
        sys.exit()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    sim = FireRobotSim()
    sim.run()