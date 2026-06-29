import pygame
import os
pygame.init()
screen = pygame.display.set_mode((900,500))

if os.path.exists("../assets/plane.png"):
    img = pygame.image.load("../assets/plane.png").convert_alpha()
screen.blit(img, (50,50))
pygame.display.flip()
pygame.time.wait(3000)