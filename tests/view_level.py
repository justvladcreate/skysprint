import pygame
import json
import sys
import os

def load_level(path):
    with open(path, 'r') as f:
        return json.load(f)

def main():

    level_path = os.path.join("../levels/9.json")

    if not os.path.exists(level_path):
        print(f"Файл '{level_path}' не найден.")
        return

    data = load_level(level_path)
    width = data.get('width', 4000)
    height = data.get('height', 1080)
    start = data['start']
    checkpoints = data.get('checkpoints', [])
    obstacles = data.get('obstacles', [])

    pygame.init()
    info = pygame.display.Info()
    screen_w, screen_h = info.current_w, info.current_h
    # Ограничим размер окна 90% экрана, но не меньше 800x600
    screen_w = max(800, int(screen_w * 0.9))
    screen_h = max(600, int(screen_h * 0.9))
    screen = pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Level Viewer")

    # Масштаб, чтобы весь уровень влез с отступами
    margin = 40
    scale_x = (screen_w - 2 * margin) / width
    scale_y = (screen_h - 2 * margin) / height
    scale = min(scale_x, scale_y)

    offset_x = (screen_w - width * scale) / 2
    offset_y = (screen_h - height * scale) / 2

    def to_screen(x, y):
        return (offset_x + x * scale, offset_y + y * scale)

    clock = pygame.time.Clock()
    running = True
    # Автозакрытие через 5 секунд (можно убрать)
    close_timer = 5000  # мс
    start_ticks = pygame.time.get_ticks()

    font = pygame.font.Font(None, 24)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.fill((30, 30, 30))

        # Границы уровня
        border_rect = pygame.Rect(
            offset_x, offset_y,
            width * scale, height * scale
        )
        pygame.draw.rect(screen, (80, 80, 80), border_rect, 2)

        # Препятствия
        for obs in obstacles:
            rx, ry = to_screen(obs['x'], obs['y'])
            rw = obs['w'] * scale
            rh = obs['h'] * scale
            pygame.draw.rect(screen, (200, 50, 50), (rx, ry, rw, rh))

        # Чекпоинты
        for i, cp in enumerate(checkpoints):
            cx, cy = to_screen(cp['x'], cp['y'])
            radius = max(3, int(cp.get('radius', 40) * scale))
            color = (255, 215, 0) if i == 0 else (180, 180, 180)
            pygame.draw.circle(screen, color, (int(cx), int(cy)), radius)
            # номер
            num_surf = font.render(str(i+1), True, (255,255,255))
            screen.blit(num_surf, (cx - num_surf.get_width()//2, cy - num_surf.get_height()//2))

        # Старт
        sx, sy = to_screen(start[0], start[1])
        pygame.draw.circle(screen, (0, 255, 0), (int(sx), int(sy)), 5)
        start_text = font.render("S", True, (0,255,0))
        screen.blit(start_text, (sx + 8, sy - 12))

        pygame.display.flip()

        # Автозакрытие
        if pygame.time.get_ticks() - start_ticks > close_timer:
            running = False

        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    main()