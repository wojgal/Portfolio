import pygame

class Field:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.width = 125
        self.height = 125
        self.rect = pygame.Rect(x, y, self.width, self.height)
        self.type = None
        self.image_x = pygame.image.load('images/x.png')
        self.image_o = pygame.image.load('images/o.png')

    def is_empty(self) -> bool:
        if self.type is None:
            return True
        
        return False

    def set_type(self, type) -> None:
        if self.type is None:
            self.type = type

    def get_type(self) -> str:
        return str(self.type)

    def clear_type(self) -> None:
        self.type = None

    def get_cords(self) -> tuple:
        return (self.x, self.y)

    def draw(self, screen) -> None:
        if self.type == 'x':
            screen.blit(self.image_x, self.get_cords())

        elif self.type == 'o':
            screen.blit(self.image_o, self.get_cords())

    def check_click(self) -> bool:
        mouse_position = pygame.mouse.get_pos()

        return self.rect.collidepoint(mouse_position)
