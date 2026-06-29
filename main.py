import pygame
import sys
import os
from config import BORDERLESS_FULLSCREEN, FPS
from screens import MainMenu, LevelSelect, Gameplay, TrainingScreen

class Game:
    def __init__(self):
        pygame.init()
        if BORDERLESS_FULLSCREEN:
            info = pygame.display.Info()
            self.screen = pygame.display.set_mode(
                (info.current_w, info.current_h),
                pygame.NOFRAME
            )
        else:
            self.screen = pygame.display.set_mode((1920, 1080))
        self.screen_width, self.screen_height = self.screen.get_size()
        pygame.display.set_caption("Sky Sprint")
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = "MAIN_MENU"
        self.current_screen = MainMenu(self)
        os.makedirs("levels", exist_ok=True)
        os.makedirs("models", exist_ok=True)

    def change_state(self, new_state, **kwargs):
        if new_state == "MAIN_MENU":
            self.current_screen = MainMenu(self)
        elif new_state == "LEVEL_SELECT":
            self.current_screen = LevelSelect(self)
        elif new_state == "GAMEPLAY":
            self.current_screen = Gameplay(self, kwargs['level_file'],
                                           kwargs['mode'], kwargs.get('model'))
        elif new_state == "TRAINING":
            self.current_screen = TrainingScreen(self, kwargs['level_file'])
        self.state = new_state

    def run(self):
        while self.running:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    self.running = False
                else:
                    self.current_screen.handle_event(event)
            self.current_screen.update()
            self.current_screen.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    Game().run()