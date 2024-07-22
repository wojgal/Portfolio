import pygame

AI_BUTTON_IMAGE = pygame.image.load('images/ai.png')
AI_PRESSED_BUTTON_IMAGE = pygame.image.load('images/ai_pressed.png')

ONE_VS_ONE_IMAGE = pygame.image.load('images/1v1.png')
ONE_VS_ONE_PRESSED_IMAGE = pygame.image.load('images/1v1_pressed.png')

RESTART_IMAGE = pygame.image.load('images/restart.png')

class Button():
    def __init__(self, x, y, image, image_pressed) -> None:
        self.x = x
        self.y = y
        self.mask = pygame.mask.from_surface(image)
        self.width, self.height = self.mask.get_size()
        self.rect = self.mask.get_rect()
        self.rect.move_ip(x, y)
        self.image = image
        self.image_pressed = image_pressed
        self.active = False

    def get_cords(self):
        return (self.x, self.y)
    
    def get_rect(self):
        return self.rect
    
    def is_active(self):
        return self.active
    
    def switch_active(self):
        self.active = not self.active

    def check_click(self):
        mouse_position = pygame.mouse.get_pos()

        return self.rect.collidepoint(mouse_position)
    
    def draw(self, screen):
        if not self.active:
            screen.blit(self.image, (self.x, self.y))

        else:
            screen.blit(self.image_pressed, (self.x, self.y))
