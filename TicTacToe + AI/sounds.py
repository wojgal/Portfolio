import pygame

pygame.mixer.init()

DRAW_SOUND = pygame.mixer.Sound('sounds/draw.mp3')
DRAW_SOUND.set_volume(0.5)

CLICK_SOUND = pygame.mixer.Sound('sounds/click.mp3')
CLICK_SOUND.set_volume(0.08)

GAME_END_SOUND = pygame.mixer.Sound('sounds/game_end.mp3')
GAME_END_SOUND.set_volume(0.3)