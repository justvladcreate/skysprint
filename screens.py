import pygame
import os
import math
import numpy as np
import torch
from env import SkySprintEnv
from agent import NeuroevoAgent
from trainer import PopulationTrainer
from config import (
    BORDERLESS_FULLSCREEN, FPS, TRAINING_FPS,
    LEVELS_FOLDER, MODELS_FOLDER
)
import records

import os
print("Текущая папка:", os.getcwd())
print("Существует plane.png:", os.path.exists("assets/plane.png"))

# ---------- Вспомогательные функции ----------
def draw_text(screen, text, x, y, font, color=(255,255,255)):
    surf = font.render(text, True, color)
    screen.blit(surf, (x, y))
    return surf.get_height()

# ---------- Загрузка спрайтов (с заглушками) ----------
def load_image(path, default_size, fallback_color=(0,0,255)):
    if os.path.exists(path):
        try:
            print("TEST")
            img = pygame.image.load(path)
            # Масштабируем до нужного размера
            img = pygame.transform.smoothscale(img, default_size)
            return img
        except Exception as e:
            print("TEST FAILED", e)
            pass
    # Заглушка, если файла нет или он битый
    surf = pygame.Surface(default_size, pygame.SRCALPHA)
    if 'plane' in path.lower():
        w, h = default_size
        pts = [(w//2, 0), (0, h), (w, h)]
        pygame.draw.polygon(surf, fallback_color, pts)
    else:
        r = min(default_size) // 2
        pygame.draw.circle(surf, (255,255,255,128), (r, r), r)
    return surf

PLANE_IMG = load_image("assets/plane.png", (120, 100), (0,0,255))
SMOKE_IMG = load_image("assets/smoke.png", (40, 40), (255,255,255))

# ---------- Button ----------
class Button:
    def __init__(self, x, y, w, h, text, callback=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.base_color = (100, 100, 100)
        self.hover_color = (150, 150, 150)
        self.current_color = self.base_color
        self.font = pygame.font.Font(None, 36)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.current_color = self.hover_color if self.rect.collidepoint(event.pos) else self.base_color
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and self.callback:
                self.callback()

    def draw(self, screen):
        pygame.draw.rect(screen, self.current_color, self.rect)
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        screen.blit(text_surf, (self.rect.x + 10, self.rect.y + 10))

# ---------- InputBox ----------
class InputBox:
    def __init__(self, x, y, w, h, text='', font=None, numeric=True, callback=None):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font or pygame.font.Font(None, 28)
        self.numeric = numeric
        self.callback = callback
        self.active = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if self.active and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                self.active = False
                if self.callback:
                    self.callback(self.text)
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                if self.numeric:
                    if event.unicode.isdigit() or event.unicode == '.':
                        self.text += event.unicode
                else:
                    self.text += event.unicode

    def draw(self, screen):
        color = (0, 255, 0) if self.active else (255, 255, 255)
        pygame.draw.rect(screen, color, self.rect, 2)
        text_surf = self.font.render(self.text, True, (255, 255, 255))
        screen.blit(text_surf, (self.rect.x + 5, self.rect.y + 5))

# ---------- Slider ----------
class Slider:
    def __init__(self, x, y, w, min_val, max_val, step, initial_val, callback=None):
        self.rect = pygame.Rect(x, y, w, 20)
        self.min = min_val
        self.max = max_val
        self.step = step
        self.value = initial_val
        self.callback = callback
        self.grabbed = False
        self.knob_radius = 10

    def _value_to_x(self):
        ratio = (self.value - self.min) / (self.max - self.min)
        return self.rect.x + int(ratio * self.rect.width)

    def _set_value_from_pos(self, x):
        x = max(self.rect.x, min(x, self.rect.x + self.rect.width))
        ratio = (x - self.rect.x) / self.rect.width
        new_val = self.min + ratio * (self.max - self.min)
        new_val = round(new_val / self.step) * self.step
        new_val = max(self.min, min(self.max, new_val))
        if new_val != self.value:
            self.value = new_val
            if self.callback:
                self.callback(self.value)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.grabbed = True
                self._set_value_from_pos(event.pos[0])
            else:
                knob_x = self._value_to_x()
                knob_rect = pygame.Rect(knob_x - self.knob_radius, self.rect.centery - self.knob_radius,
                                        self.knob_radius*2, self.knob_radius*2)
                if knob_rect.collidepoint(event.pos):
                    self.grabbed = True
                    self._set_value_from_pos(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.grabbed = False
        elif event.type == pygame.MOUSEMOTION and self.grabbed:
            self._set_value_from_pos(event.pos[0])

    def draw(self, screen):
        pygame.draw.line(screen, (200,200,200), (self.rect.x, self.rect.centery), (self.rect.x+self.rect.width, self.rect.centery), 4)
        knob_x = self._value_to_x()
        pygame.draw.circle(screen, (0,200,0), (knob_x, self.rect.centery), self.knob_radius)
        font = pygame.font.Font(None, 28)
        text = font.render(str(int(self.value)), True, (255,255,255))
        screen.blit(text, (knob_x - text.get_width()//2, self.rect.y - 25))

# ---------- SaveDialog (без изменений) ----------
class SaveDialog:
    def __init__(self, game, level_name, best_time, on_save, on_cancel):
        self.game = game
        self.level_name = level_name
        self.best_time = best_time
        self.on_save = on_save
        self.on_cancel = on_cancel
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)

        self.model_base = MODELS_FOLDER
        self.level_dirs = [d for d in os.listdir(self.model_base) if d.startswith("level_") and os.path.isdir(os.path.join(self.model_base, d))]
        self.selected_dir = None
        self.dir_buttons = []
        self._build_dir_buttons()

        self.input_text = ""
        self.active = False
        self.save_btn = Button(300, 500, 100, 40, "Save", self._save)
        self.cancel_btn = Button(420, 500, 100, 40, "Cancel", self._cancel)
        self.sw, self.sh = game.screen_width, game.screen_height
        self.input_rect = pygame.Rect(100, 240, 250, 40)   # поле ввода

    def _build_dir_buttons(self):
        self.dir_buttons.clear()
        y = 150
        for d in self.level_dirs:
            btn = Button(100, y, 250, 40, d, lambda dir=d: self._select_dir(dir))
            self.dir_buttons.append(btn)
            y += 50

    def _select_dir(self, dir_name):
        self.selected_dir = dir_name
        self.active = True
        self.input_text = ""                 # сброс текста при выборе папки

    def _save(self):
        if self.selected_dir and self.input_text.strip():
            self.on_save(self.selected_dir, self.input_text.strip())
        self.on_cancel()

    def _cancel(self):
        self.on_cancel()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._cancel()
            return

        # Кнопки папок – только когда папка ещё не выбрана
        if not self.selected_dir:
            for btn in self.dir_buttons:
                btn.handle_event(event)
        else:
            # Обработка клика по полю ввода
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self.active = self.input_rect.collidepoint(event.pos)
            if self.active and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.input_text = self.input_text[:-1]
                elif event.key == pygame.K_RETURN:
                    self._save()
                else:
                    self.input_text += event.unicode

        self.save_btn.handle_event(event)
        self.cancel_btn.handle_event(event)

    def draw(self, screen):
        overlay = pygame.Surface((self.sw, self.sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))

        title = self.font.render("Save Best Model", True, (255, 255, 255))
        screen.blit(title, (100, 50))
        time_text = self.small_font.render(f"Best time: {self.best_time:.2f}s", True, (200, 200, 200))
        screen.blit(time_text, (100, 80))

        if not self.selected_dir:
            prompt = self.font.render("Select level folder:", True, (255, 255, 255))
            screen.blit(prompt, (100, 120))
            for btn in self.dir_buttons:
                btn.draw(screen)
        else:
            current = self.font.render(f"Folder: {self.selected_dir}", True, (255, 255, 255))
            screen.blit(current, (100, 120))
            name_prompt = self.font.render("Model name:", True, (255, 255, 255))
            screen.blit(name_prompt, (100, 200))
            pygame.draw.rect(screen, (255, 255, 255), self.input_rect, 2)
            text_surf = self.font.render(self.input_text, True, (255, 255, 255))
            screen.blit(text_surf, (self.input_rect.x + 5, self.input_rect.y + 5))
            self.save_btn.draw(screen)
            self.cancel_btn.draw(screen)

# ---------- LoadModelDialog ----------
class LoadModelDialog:
    def __init__(self, game, level_name, on_load, on_cancel):
        self.game = game
        self.level_name = level_name
        self.on_load = on_load
        self.on_cancel = on_cancel
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)

        # Ищем папку уровня
        self.model_dir = os.path.join(MODELS_FOLDER, f"level_{level_name}")
        self.models = []
        self.model_buttons = []
        if os.path.exists(self.model_dir):
            self.models = [f for f in os.listdir(self.model_dir) if f.endswith('.pkl')]
        self._build_model_buttons()
        self.sw, self.sh = game.screen_width, game.screen_height

    def _build_model_buttons(self):
        self.model_buttons.clear()
        y = 150
        for m in self.models:
            path = os.path.join(self.model_dir, m)
            try:
                agent = NeuroevoAgent.load(path, device='cpu')
                pb_time = getattr(agent, 'pb_time', None)
                label = f"{m}"
                if pb_time is not None:
                    label += f" - {pb_time:.2f}s"
            except:
                label = f"{m} (invalid)"
            btn = Button(100, y, 400, 40, label, lambda p=path: self._select_model(p))
            self.model_buttons.append(btn)
            y += 50

    def _select_model(self, path):
        self.on_load(path)
        self.on_cancel()

    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.on_cancel()
            return
        for btn in self.model_buttons:
            btn.handle_event(event)

    def draw(self, screen):
        overlay = pygame.Surface((self.sw, self.sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        screen.blit(overlay, (0, 0))
        title = self.font.render("Load Model", True, (255, 255, 255))
        screen.blit(title, (100, 50))
        prompt = self.font.render("Select model to continue training:", True, (255, 255, 255))
        screen.blit(prompt, (100, 100))
        for btn in self.model_buttons:
            btn.draw(screen)

# ---------- MainMenu (без изменений) ----------
# (класс MainMenu оставлен как в предыдущей версии, здесь не дублируется)

# ---------- LevelSelect (без изменений) ----------
# (класс LevelSelect оставлен как в предыдущей версии)

# ---------- Gameplay (без изменений, кроме stochastic fix) ----------


# ---------- MainMenu ----------
class MainMenu:
    def __init__(self, game):
        self.game = game
        sw, sh = game.screen_width, game.screen_height
        self.buttons = [
            Button(sw//2 - 160, sh//2 - 30, 320, 60, "Select Level", lambda: game.change_state("LEVEL_SELECT")),
            Button(sw//2 - 160, sh//2 + 60, 320, 60, "Exit", self._confirm_exit)
        ]
        self.confirm_exit = False
        self.exit_buttons = []

    def _confirm_exit(self):
        self.confirm_exit = True
        sw = self.game.screen_width
        sh = self.game.screen_height
        self.exit_buttons = [
            Button(sw//2 - 110, sh//2 + 20, 100, 50, "OK", self._exit_game),
            Button(sw//2 + 10, sh//2 + 20, 100, 50, "Cancel", self._cancel_exit)
        ]

    def _exit_game(self):
        self.game.running = False

    def _cancel_exit(self):
        self.confirm_exit = False

    def handle_event(self, event):
        if self.confirm_exit:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self._exit_game()
                elif event.key == pygame.K_ESCAPE:
                    self._cancel_exit()
            for btn in self.exit_buttons:
                btn.handle_event(event)
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.confirm_exit = True
            return
        for btn in self.buttons:
            btn.handle_event(event)

    def update(self): pass

    def draw(self, screen):
        screen.fill((30, 30, 30))
        sw, sh = self.game.screen_width, self.game.screen_height
        font = pygame.font.Font(None, 72)
        title = font.render("Sky Sprint", True, (255, 255, 255))
        screen.blit(title, (sw//2 - 120, sh//4 - 50))
        for btn in self.buttons:
            btn.draw(screen)

        if self.confirm_exit:
            overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            screen.blit(overlay, (0, 0))
            dialog_rect = pygame.Rect(sw//2 - 150, sh//2 - 40, 300, 120)
            pygame.draw.rect(screen, (50, 50, 50), dialog_rect)
            pygame.draw.rect(screen, (255, 255, 255), dialog_rect, 2)
            prompt = pygame.font.Font(None, 36).render("Exit game?", True, (255, 255, 255))
            screen.blit(prompt, (sw//2 - 60, sh//2 - 30))
            for btn in self.exit_buttons:
                btn.draw(screen)

# ---------- LevelSelect ----------
class LevelSelect:
    def __init__(self, game):
        self.game = game
        sw, sh = game.screen_width, game.screen_height
        self.selected_level_index = None
        self.level_files = sorted([
            f for f in os.listdir(LEVELS_FOLDER)
            if f.endswith('.json') and f != 'records.json'
        ])
        self.num_levels = len(self.level_files)

        self.level_buttons = []
        cols = 4
        btn_w, btn_h = 150, 100
        start_x = 80
        start_y = 200
        for i in range(self.num_levels):
            x = start_x + (i % cols) * (btn_w + 20)
            y = start_y + (i // cols) * (btn_h + 20)
            btn = Button(x, y, btn_w, btn_h, str(i+1), lambda idx=i: self._select_level(idx))
            self.level_buttons.append(btn)

        panel_x = sw - 520
        self.solo_btn = Button(panel_x, 300, 300, 60, "Play Solo", self._start_solo)
        self.vs_ai_btn = Button(panel_x, 400, 300, 60, "Play vs AI", self._open_ai_select)
        self.train_btn = Button(panel_x, 500, 300, 60, "Train Model", self._start_training)
        self.back_btn = Button(50, 50, 150, 50, "Back", lambda: game.change_state("MAIN_MENU"))

        self.selected_model = None
        self.pb_text = ""
        self.ai_models = []
        self.ai_buttons = []
        self.show_ai_select = False

        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 28)

    def _select_level(self, idx):
        self.selected_level_index = idx
        self.pb_text = ""
        self.show_ai_select = False
        level_name = os.path.splitext(self.level_files[idx])[0]
        pb = records.get_pb(level_name + ".json")
        if pb:
            self.pb_text = f"PB: {pb[0]:.2f}s  Deaths: {pb[1]}"
        else:
            self.pb_text = "PB: N/A"

        level_dir = os.path.join(MODELS_FOLDER, f"level_{level_name}")
        self.ai_models = []
        if os.path.exists(level_dir):
            self.ai_models = [f for f in os.listdir(level_dir) if f.endswith('.pkl')]
        else:
            self.ai_models = []

    def _open_ai_select(self):
        if self.selected_level_index is None:
            return
        self.show_ai_select = True
        self._build_ai_buttons()

    def _build_ai_buttons(self):
        self.ai_buttons.clear()
        y = 200
        for m in self.ai_models:
            path = os.path.join(MODELS_FOLDER, f"level_{os.path.splitext(self.level_files[self.selected_level_index])[0]}", m)
            try:
                agent = NeuroevoAgent.load(path, device='cpu')
                pb_time = getattr(agent, 'pb_time', None)
                deaths = getattr(agent, 'deaths', 0)
                if pb_time is not None:
                    label = f"{m} - {pb_time:.2f}s  Deaths: {deaths}"
                else:
                    label = f"{m} - Time N/A"
            except:
                label = f"{m} (invalid)"
            btn = Button(100, y, 500, 40, label, lambda path=path: self._start_vs_ai(path))
            self.ai_buttons.append(btn)
            y += 50

    def _start_vs_ai(self, model_path):
        self.selected_model = NeuroevoAgent.load(model_path, device='cpu')
        fname = self.level_files[self.selected_level_index]
        self.game.change_state("GAMEPLAY",
                               level_file=os.path.join(LEVELS_FOLDER, fname),
                               mode="vs_ai",
                               model=self.selected_model)

    def _start_solo(self):
        if self.selected_level_index is not None:
            fname = self.level_files[self.selected_level_index]
            self.game.change_state("GAMEPLAY",
                                   level_file=os.path.join(LEVELS_FOLDER, fname),
                                   mode="solo")

    def _start_training(self):
        if self.selected_level_index is not None:
            fname = self.level_files[self.selected_level_index]
            level_path = os.path.join(LEVELS_FOLDER, fname)
            self.game.change_state("TRAINING", level_file=level_path)

    def handle_event(self, event):
        if self.show_ai_select:
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.show_ai_select = False
                return
            for btn in self.ai_buttons:
                btn.handle_event(event)
            return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.game.change_state("MAIN_MENU")
            return

        for btn in (self.level_buttons + [self.solo_btn, self.vs_ai_btn, self.train_btn, self.back_btn]):
            btn.handle_event(event)

    def update(self): pass

    def draw(self, screen):
        screen.fill((30, 30, 30))
        sw, sh = self.game.screen_width, self.game.screen_height
        font = self.font
        small = self.small_font

        draw_text(screen, "Select Level", 80, 100, font)
        for btn in self.level_buttons:
            btn.draw(screen)

        panel_x = sw - 520
        pygame.draw.rect(screen, (50, 50, 50), (panel_x - 10, 200, 520, 600))
        if self.selected_level_index is not None:
            text = f"Level {self.selected_level_index+1}"
            draw_text(screen, text, panel_x, 250, font)
            draw_text(screen, self.pb_text, panel_x, 280, small, (200,200,200))

            level_name = os.path.splitext(self.level_files[self.selected_level_index])[0]
            level_dir = os.path.join(MODELS_FOLDER, f"level_{level_name}")
            ai_times = []
            if os.path.exists(level_dir):
                for m in os.listdir(level_dir):
                    if m.endswith('.pkl'):
                        try:
                            agent = NeuroevoAgent.load(os.path.join(level_dir, m), device='cpu')
                            t = getattr(agent, 'pb_time', None)
                            if t is not None:
                                ai_times.append((m, t, getattr(agent, 'deaths', 0)))
                        except:
                            pass
            ai_times.sort(key=lambda x: x[1])
            y = 310
            for i, (name, t, d) in enumerate(ai_times[:3]):
                line = f"{name}: {t:.2f}s Deaths: {d}"
                draw_text(screen, line, panel_x, y, small, (150,150,150))
                y += 30

        self.solo_btn.draw(screen)
        self.vs_ai_btn.draw(screen)
        self.train_btn.draw(screen)
        self.back_btn.draw(screen)

        if self.show_ai_select:
            overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen.blit(overlay, (0, 0))

            draw_text(screen, "Select AI Model", 100, 100, font)
            for btn in self.ai_buttons:
                btn.draw(screen)

# ---------- Gameplay (добавлены экраны YOU WON и AI WON, клавиша R везде) ----------
class Gameplay:
    def __init__(self, game, level_file, mode, model=None):
        self.game = game
        self.level_file = level_file
        self.mode = mode
        self.model = model
        self.env = SkySprintEnv(level_file, training_mode=False)
        if mode == "vs_ai":
            self.env.add_second_player()
        self.paused = False
        self.time_elapsed = 0.0
        self.camera_x = 0
        self.camera_y = 0
        self.font = pygame.font.Font(None, 36)
        self.player_action = 0

        self.ai_won = False
        self.player_won = False
        self.winner = None
        self.winner_time = 0.0

        self.use_sprites = True
        self.smoke_particles = []
        self.ai_randomness = False

        sw, sh = game.screen_width, game.screen_height
        self.pause_buttons = [
            Button(sw//2 - 100, sh//2, 200, 50, "Continue", self._unpause),
            Button(sw//2 - 100, sh//2 + 60, 200, 50, "Restart", self._restart),
            Button(sw//2 - 100, sh//2 + 120, 200, 50, "Back", self._quit_to_menu),
            Button(sw//2 - 100, sh//2 + 180, 200, 50, "Sprites: ON", self._toggle_sprites),
            Button(sw//2 - 100, sh//2 + 240, 200, 50, "AI Random: OFF", self._toggle_ai_random)
        ]
        self.end_buttons = [
            Button(sw//2 - 100, sh//2 + 20, 200, 50, "Restart", self._restart),
            Button(sw//2 - 100, sh//2 + 80, 200, 50, "Quit to Menu", self._quit_to_menu)
        ]

    def _unpause(self): self.paused = False
    def _restart(self):
        self.env.reset()
        if self.mode == "vs_ai":
            self.env.add_second_player()
        self.time_elapsed = 0.0
        self.paused = False
        self.ai_won = False
        self.player_won = False
        self.winner = None
        self.smoke_particles.clear()

    def _quit_to_menu(self):
        self.game.change_state("LEVEL_SELECT")

    def _toggle_sprites(self):
        self.use_sprites = not self.use_sprites
        for btn in self.pause_buttons:
            if "Sprites:" in btn.text:
                btn.text = "Sprites: ON" if self.use_sprites else "Sprites: OFF"

    def _toggle_ai_random(self):
        self.ai_randomness = not self.ai_randomness
        for btn in self.pause_buttons:
            if "AI Random:" in btn.text:
                btn.text = "AI Random: ON" if self.ai_randomness else "AI Random: OFF"

    def handle_event(self, event):
        if self.ai_won or self.player_won:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    self._restart()
                elif event.key == pygame.K_ESCAPE:
                    self._quit_to_menu()
            for btn in self.end_buttons:
                btn.handle_event(event)
            return

        if self.paused:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._quit_to_menu()
                elif event.key == pygame.K_r:
                    self._restart()
            for btn in self.pause_buttons:
                btn.handle_event(event)
            return

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.paused = True
            elif event.key == pygame.K_SPACE:
                self.player_action = 1
            elif event.key == pygame.K_r:
                self._restart()
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_SPACE:
                self.player_action = 0

    def update(self):
        if self.paused or self.ai_won or self.player_won:
            return
        ai_action = 0
        if self.mode == "vs_ai" and self.model:
            state2 = self.env.get_state(2)
            if state2 and state2['alive']:
                eps = 0.05 if self.ai_randomness else 0.0
                ai_action = self.model.predict(state2, epsilon=eps)
        state1, state2, done1, done2 = self.env.step(self.player_action, ai_action)
        self.time_elapsed += 1 / 60

        if self.use_sprites:
            for p, action in [(self.env.player1, self.player_action),
                              (self.env.player2, ai_action) if self.env.player2 else (None, None)]:
                if p and p['alive'] and action == 1:
                    back_x = p['x'] - 15 * math.cos(p['angle'])
                    back_y = p['y'] + 15 * math.sin(p['angle'])
                    self.smoke_particles.append([back_x, back_y, 1.0])

        dt = 1.0 / FPS
        for p in self.smoke_particles[:]:
            p[2] -= dt
            if p[2] <= 0:
                self.smoke_particles.remove(p)

        if done1 or done2:
            if done1 and not done2:
                self.winner = "player"
            elif done2 and not done1:
                self.winner = "ai"
            else:
                self.winner = "player" if done1 else "ai"

            self.winner_time = self.time_elapsed

            if done1:
                level_name = os.path.basename(self.level_file)
                records.save_record(level_name, self.time_elapsed, state1.get('deaths', 0))

            if self.winner == "ai":
                self.ai_won = True
                self.paused = True
            else:
                self.player_won = True
                self.paused = True
            return

        self.camera_x = state1['x'] - self.game.screen_width // 2
        self.camera_y = state1['y'] - self.game.screen_height // 2

    def draw(self, screen):
        screen.fill((135, 206, 235))
        for obs in self.env.obstacles:
            rect = pygame.Rect(obs['x'] - self.camera_x, obs['y'] - self.camera_y, obs['w'], obs['h'])
            pygame.draw.rect(screen, (100, 100, 100), rect)
        for i, cp in enumerate(self.env.checkpoints):
            cx, cy = cp['x'] - self.camera_x, cp['y'] - self.camera_y
            color = (255, 215, 0) if i == self.env.player1['next_cp'] else (200, 200, 200)
            pygame.draw.circle(screen, color, (int(cx), int(cy)), cp.get('radius', 40), 3)
            num = self.font.render(str(i + 1), True, (255, 255, 255))
            screen.blit(num, (cx - 10, cy - 10))

        if self.use_sprites:
            for x, y, life in self.smoke_particles:
                sx = x - self.camera_x
                sy = y - self.camera_y
                alpha = max(0, min(255, int(255 * life)))
                smoke_surf = SMOKE_IMG.copy()
                smoke_surf.set_alpha(alpha)
                screen.blit(smoke_surf, (sx - smoke_surf.get_width()//2, sy - smoke_surf.get_height()//2))

        p1 = self.env.player1
        self._draw_plane(screen, p1['x'] - self.camera_x, p1['y'] - self.camera_y, p1['angle'], (0, 0, 255))
        if self.env.player2:
            p2 = self.env.player2
            self._draw_plane(screen, p2['x'] - self.camera_x, p2['y'] - self.camera_y, p2['angle'], (255, 0, 0))

        timer = self.font.render(f"Time: {self.time_elapsed:.2f}", True, (255, 255, 255))
        screen.blit(timer, (20, 20))
        deaths1 = self.env.player1.get('deaths', 0)
        death_text1 = self.font.render(f"Your Deaths: {deaths1}", True, (255, 255, 255))
        screen.blit(death_text1, (20, 50))
        if self.env.player2:
            deaths2 = self.env.player2.get('deaths', 0)
            death_text2 = self.font.render(f"AI Deaths: {deaths2}", True, (255, 255, 255))
            screen.blit(death_text2, (self.game.screen_width - 200, 50))

        if self.player_won:
            overlay = pygame.Surface((self.game.screen_width, self.game.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            win_text = self.font.render(f"YOU WON! Your time: {self.winner_time:.2f}s", True, (255, 255, 255))
            screen.blit(win_text, (self.game.screen_width//2 - 180, self.game.screen_height//2 - 100))
            for btn in self.end_buttons:
                btn.draw(screen)
        elif self.ai_won:
            overlay = pygame.Surface((self.game.screen_width, self.game.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            win_text = self.font.render(f"AI WON! His time: {self.winner_time:.2f}s", True, (255, 255, 255))
            screen.blit(win_text, (self.game.screen_width//2 - 180, self.game.screen_height//2 - 100))
            for btn in self.end_buttons:
                btn.draw(screen)
        elif self.paused:
            overlay = pygame.Surface((self.game.screen_width, self.game.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))
            for btn in self.pause_buttons:
                btn.draw(screen)

    def _draw_plane(self, screen, x, y, angle, color):
        if self.use_sprites:
            rotated = pygame.transform.rotate(PLANE_IMG, math.degrees(angle))
            rect = rotated.get_rect(center=(x, y))
            screen.blit(rotated, rect.topleft)
        else:
            points = [(12, 0), (-8, -6), (-8, 6)]
            rotated_pts = []
            sa = math.sin(-angle)
            ca = math.cos(-angle)
            for px, py in points:
                rx = px * ca - py * sa
                ry = px * sa + py * ca
                rotated_pts.append((x + rx, y + ry))
            pygame.draw.polygon(screen, color, rotated_pts)

# ---------- TrainingScreen (с поддержкой загрузки модели) ----------
class TrainingScreen:
    def __init__(self, game, level_file):
        self.game = game
        self.level_file = level_file
        sw, sh = game.screen_width, game.screen_height
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

        self.pop_size = 50
        self.max_steps = 2000
        self.speed_multiplier = 1
        self.weak_mutation_rate = 0.01
        self.weak_mutation_scale = 0.01
        self.medium_mutation_rate = 0.1
        self.medium_mutation_scale = 0.2
        self.strong_mutation_scale = 5.0
        self.weak_ratio = 0.6
        self.medium_ratio = 0.3

        self.trainer = None
        self.training = False
        self.paused = False
        self.generation_complete = False
        self.save_dialog = None
        self.load_dialog = None
        self.loaded_weights = None    # numpy-массив весов загруженной модели

        self.use_sprites = True

        panel_width = 350
        self.panel_width = panel_width

        btn_y = sh - 100
        self.start_btn = Button(20, btn_y, 120, 40, "Start", self._start_training)
        self.stop_btn = Button(20, btn_y, 120, 40, "Stop", self._stop_training)
        self.pause_btn = Button(150, btn_y, 100, 40, "Pause", self._toggle_pause)
        self.save_btn = Button(150, btn_y+45, 150, 40, "Save Best", self._open_save_dialog)
        self.load_btn = Button(20, btn_y-50, 150, 40, "Load Model", self._open_load_dialog)
        self.back_btn = Button(20, 20, 100, 40, "Back", lambda: self._go_back())

        self.speed_slider = Slider(20, 650, 200, 1, 10, 1, self.speed_multiplier, self._set_speed)

        self.param_boxes = {}
        labels = [
            ("pop_size", "Pop"),
            ("max_steps", "Steps"),
            ("weak_mutation_rate", "W rate"),
            ("weak_mutation_scale", "W scale"),
            ("medium_mutation_rate", "M rate"),
            ("medium_mutation_scale", "M scale"),
            ("strong_mutation_scale", "S scale"),
            ("weak_ratio", "W ratio"),
            ("medium_ratio", "M ratio"),
        ]
        y0 = 100
        for i, (attr, label) in enumerate(labels):
            col = i % 2
            row = i // 2
            x = 20 + col * 170
            y = y0 + row * 60
            box = InputBox(x, y+20, 70, 28, str(getattr(self, attr)), self.small_font, True,
                           lambda text, a=attr: self._set_param(a, text))
            self.param_boxes[attr] = (label, box)

        self.stats_rect = pygame.Rect(panel_width+20, 20, 250, 180)
        self.sprites_btn = Button(0, 0, 130, 30, "", self._toggle_sprites)
        self.dragging_stats = False
        self.drag_offset = (0, 0)

        self.camera_x = 0
        self.camera_y = 0

    def _set_param(self, attr, text):
        try:
            val = float(text) if '.' in text else int(text)
            setattr(self, attr, val)
        except:
            pass

    def _set_speed(self, val):
        self.speed_multiplier = int(val)

    def _go_back(self):
        if self.training:
            self._stop_training()
        self.game.change_state("LEVEL_SELECT")

    def _toggle_sprites(self):
        self.use_sprites = not self.use_sprites

    def _apply_all_params(self):
        for attr, (_, box) in self.param_boxes.items():
            self._set_param(attr, box.text)

    def _open_load_dialog(self):
        if self.training:
            return  # не даём менять модель во время тренировки
        level_name = os.path.splitext(os.path.basename(self.level_file))[0]
        self.load_dialog = LoadModelDialog(self.game, level_name, on_load=self._on_model_loaded, on_cancel=self._close_load_dialog)

    def _on_model_loaded(self, path):
        try:
            agent = NeuroevoAgent.load(path, device='cpu')
            self.loaded_weights = agent.get_weights_flat()
            print(f"Loaded model from {path}")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def _close_load_dialog(self):
        self.load_dialog = None

    def _start_training(self):
        if self.training: return
        self._apply_all_params()
        self.training = True
        self.paused = False
        self.generation_complete = False
        self.trainer = PopulationTrainer(
            self.level_file,
            pop_size=self.pop_size,
            max_steps=self.max_steps,
            weak_mutation_rate=self.weak_mutation_rate,
            weak_mutation_scale=self.weak_mutation_scale,
            medium_mutation_rate=self.medium_mutation_rate,
            medium_mutation_scale=self.medium_mutation_scale,
            strong_mutation_scale=self.strong_mutation_scale,
            weak_ratio=self.weak_ratio,
            medium_ratio=self.medium_ratio,
            init_weights=self.loaded_weights,   # передаём веса, если есть
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

    def _stop_training(self):
        if self.trainer:
            self._save_best("best")
        self.training = False
        self.trainer = None

    def _toggle_pause(self):
        self.paused = not self.paused
        self.pause_btn.text = "Resume" if self.paused else "Pause"

    def _open_save_dialog(self):
        if self.trainer:
            level_name = os.path.splitext(os.path.basename(self.level_file))[0]
            best_time = self.trainer.best_time if self.trainer.best_time else 0.0
            self.save_dialog = SaveDialog(self.game, level_name, best_time, on_save=self._on_save_dialog, on_cancel=self._close_save_dialog)

    def _on_save_dialog(self, folder, name):
        self._save_best(name, subfolder=folder)

    def _close_save_dialog(self):
        self.save_dialog = None

    def _save_best(self, model_name, subfolder=None):
        if self.trainer:
            agent, time, deaths = self.trainer.get_best_info()
            level_name = os.path.splitext(os.path.basename(self.level_file))[0]
            if subfolder:
                save_dir = os.path.join(MODELS_FOLDER, subfolder)
            else:
                save_dir = os.path.join(MODELS_FOLDER, f"level_{level_name}")
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{model_name}.pkl")
            agent.save(path, pb_time=time, deaths=deaths)
            print(f"Model saved to {path}")

    def handle_event(self, event):
        if self.save_dialog:
            self.save_dialog.handle_event(event)
            return
        if self.load_dialog:
            self.load_dialog.handle_event(event)
            return

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._go_back()
            return

        self.speed_slider.handle_event(event)

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.sprites_btn.rect.collidepoint(event.pos):
                self.sprites_btn.callback()
            elif self.stats_rect.collidepoint(event.pos):
                self.dragging_stats = True
                self.drag_offset = (self.stats_rect.x - event.pos[0], self.stats_rect.y - event.pos[1])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging_stats = False
        elif event.type == pygame.MOUSEMOTION and self.dragging_stats:
            self.stats_rect.x = event.pos[0] + self.drag_offset[0]
            self.stats_rect.y = event.pos[1] + self.drag_offset[1]
            view_x = self.panel_width + 20
            view_y = 20
            view_w = self.game.screen_width - self.panel_width - 40
            view_h = self.game.screen_height - 40
            self.stats_rect.clamp_ip(pygame.Rect(view_x, view_y, view_w, view_h))

        for box in self.param_boxes.values():
            box[1].handle_event(event)

        if self.training:
            self.stop_btn.handle_event(event)
            self.pause_btn.handle_event(event)
            self.save_btn.handle_event(event)
        else:
            self.start_btn.handle_event(event)
            self.load_btn.handle_event(event)
        self.back_btn.handle_event(event)

    def update(self):
        if self.save_dialog or self.load_dialog or not self.training or not self.trainer or self.paused:
            return
        steps = self.speed_multiplier
        for _ in range(steps):
            if not self.trainer.generation_complete():
                self.trainer.step_all()
            else:
                self.trainer.evolve()
                self.generation_complete = True
                break
        if self.trainer and not self.trainer.generation_complete():
            best_state = self.trainer.get_best_state()
            if best_state:
                self.camera_x = best_state['x'] - self.game.screen_width // 2
                self.camera_y = best_state['y'] - self.game.screen_height // 2

    def draw(self, screen):
        screen.fill((30, 30, 30))
        sw, sh = self.game.screen_width, self.game.screen_height
        pw = self.panel_width
        pygame.draw.rect(screen, (40, 40, 40), (0, 0, pw, sh))

        draw_text(screen, "Training", 20, 60, self.font)

        for attr, (label, box) in self.param_boxes.items():
            draw_text(screen, label, box.rect.x, box.rect.y-15, self.small_font, (200,200,200))
            box.draw(screen)

        draw_text(screen, "Speed:", 20, 620, self.small_font, (200,200,200))
        self.speed_slider.draw(screen)

        if self.training:
            self.stop_btn.draw(screen)
            self.pause_btn.draw(screen)
            self.save_btn.draw(screen)
        else:
            self.start_btn.draw(screen)
            self.load_btn.draw(screen)
        self.back_btn.draw(screen)

        view_x = pw + 20
        view_y = 20
        view_w = sw - pw - 40
        view_h = sh - 40
        pygame.draw.rect(screen, (60, 60, 60), (view_x-2, view_y-2, view_w+4, view_h+4), 2)

        if self.trainer and self.trainer.states:
            env = self.trainer.envs[0]
            for obs in env.obstacles:
                rx = obs['x'] - self.camera_x
                ry = obs['y'] - self.camera_y
                if rx + obs['w'] > 0 and rx < view_w and ry + obs['h'] > 0 and ry < view_h:
                    rect = pygame.Rect(view_x + rx, view_y + ry, obs['w'], obs['h'])
                    pygame.draw.rect(screen, (100, 100, 100), rect)
            for i, cp in enumerate(env.checkpoints):
                cx = view_x + cp['x'] - self.camera_x
                cy = view_y + cp['y'] - self.camera_y
                color = (255, 215, 0) if i == env.player1['next_cp'] else (200, 200, 200)
                pygame.draw.circle(screen, color, (int(cx), int(cy)), cp.get('radius', 40), 2)

            for i in range(self.trainer.pop_size):
                state = self.trainer.states[i]
                if state is None: continue
                sx = state['x'] - self.camera_x
                sy = state['y'] - self.camera_y
                if not (-50 < sx < view_w + 50 and -50 < sy < view_h + 50): continue
                x = view_x + sx
                y = view_y + sy
                done = self.trainer.dones[i]
                alive = state['alive']
                if done or not alive:
                    color = (255, 0, 0)
                else:
                    sorted_idx = np.argsort(self.trainer.fitnesses)[::-1]
                    if i == sorted_idx[0]:
                        color = (0, 255, 0)
                    elif i in sorted_idx[:10]:
                        color = (0, 200, 0)
                    else:
                        color = (0, 150, 0)
                self._draw_plane(screen, x, y, state['angle'], color)

            best_idx = self.trainer.get_best_agent_index()
            best_state = self.trainer.states[best_idx]
            if best_state and 'rays' in best_state:
                ax = view_x + best_state['x'] - self.camera_x
                ay = view_y + best_state['y'] - self.camera_y
                base_angle = best_state['angle']
                ray_angles = [-0.8, -0.4, 0.0, 0.4, 0.8]
                for j, d in enumerate(best_state['rays']):
                    angle = base_angle + ray_angles[j]
                    length = d * 400.0
                    ex = ax + math.cos(angle) * length
                    ey = ay - math.sin(angle) * length
                    color = (0, 255, 0) if d >= 0.999 else (255, 0, 0)
                    pygame.draw.line(screen, color, (ax, ay), (ex, ey), 1)

        stats_surf = pygame.Surface((self.stats_rect.width, self.stats_rect.height), pygame.SRCALPHA)
        stats_surf.fill((0, 0, 0, 180))
        screen.blit(stats_surf, (self.stats_rect.x, self.stats_rect.y))
        y_off = 5
        if self.trainer:
            lines = [
                f"Gen: {self.trainer.generation}",
                f"Best fitness: {self.trainer.best_fitness:.1f}",
                f"Avg fitness: {self.trainer.avg_fitness:.1f}",
                f"Alive: {sum(not d for d in self.trainer.dones)}/{self.pop_size}",
                f"Steps: {self.trainer.step_counter}/{self.trainer.max_steps}",
                f"Best time: {self.trainer.best_time:.2f}s" if self.trainer.best_time else "Best time: N/A",
            ]
            for line in lines:
                draw_text(screen, line, self.stats_rect.x+10, self.stats_rect.y+y_off, self.small_font)
                y_off += 20

        self.sprites_btn.text = "Sprites: ON" if self.use_sprites else "Sprites: OFF"
        self.sprites_btn.rect.x = self.stats_rect.x + 10
        self.sprites_btn.rect.y = self.stats_rect.y + y_off + 5
        self.sprites_btn.draw(screen)

        if self.save_dialog:
            self.save_dialog.draw(screen)
        if self.load_dialog:
            self.load_dialog.draw(screen)

    def _draw_plane(self, screen, x, y, angle, color):
        if self.use_sprites:
            rotated = pygame.transform.rotate(PLANE_IMG, math.degrees(angle))
            rect = rotated.get_rect(center=(x, y))
            screen.blit(rotated, rect.topleft)
        else:
            points = [(12, 0), (-8, -6), (-8, 6)]
            rotated_pts = []
            sa = math.sin(-angle)
            ca = math.cos(-angle)
            for px, py in points:
                rx = px * ca - py * sa
                ry = px * sa + py * ca
                rotated_pts.append((x + rx, y + ry))
            pygame.draw.polygon(screen, color, rotated_pts)