import pygame
import random



class Block(pygame.sprite.Sprite):
    def __init__(self, x, y, width, height, image, square_image) -> None:
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rect = pygame.Rect(x, y, width, height)
        self.mask = pygame.mask.from_surface(image)
        self.image = image
        self.square_image = square_image

    def get_coord(self) -> tuple:
        return (self.x, self.y)
    
    def get_square_image(self) -> pygame.image:
        return self.square_image
    
    def move_down(self) -> None:
        self.y += 24
        self.rect.move_ip(0, 24)
        
    def move_up(self) -> None:
        self.y -= 24
        self.rect.move_ip(0, -24)

    def move_left(self) -> None:
        self.x -= 24
        self.rect.move_ip(-24, 0)

    def move_right(self) -> None:
        self.x += 24
        self.rect.move_ip(24, 0)

    def rotate(self) -> None:
        self.image = pygame.transform.rotate(self.image, -90)
        self.mask = pygame.mask.from_surface(self.image)

    def fall_down(self) -> None:
        pass

    def draw(self, screen) -> None:
        screen.blit(self.image, self.get_coord())

    def update(self, max_y):
        if self.y < max_y:
            self.move_down()



SQUARE_RED_BLOCK_IMAGE = pygame.image.load('images/blocks/square_red.png')
SQUARE_BLUE_BLOCK_IMAGE = pygame.image.load('images/blocks/square_blue.png')
SQUARE_YELLOW_BLOCK_IMAGE = pygame.image.load('images/blocks/square_yellow.png')
SQUARE_GREEN_BLOCK_IMAGE = pygame.image.load('images/blocks/square_green.png')
SQUARE_CYAN_BLOCK_IMAGE = pygame.image.load('images/blocks/square_cyan.png')
SQUARE_GREY_BLOCK_IMAGE = pygame.image.load('images/blocks/square_grey.png')
SQUARE_MAGENTA_BLOCK_IMAGE = pygame.image.load('images/blocks/square_magenta.png')

I_BLOCK_IMAGE = pygame.image.load('images/blocks/i.png')
T_BLOCK_IMAGE = pygame.image.load('images/blocks/t.png')
O_BLOCK_IMAGE = pygame.image.load('images/blocks/o.png')
L_BLOCK_IMAGE = pygame.image.load('images/blocks/l.png')
J_BLOCK_IMAGE = pygame.image.load('images/blocks/j.png')
S_BLOCK_IMAGE = pygame.image.load('images/blocks/s.png')
Z_BLOCK_IMAGE = pygame.image.load('images/blocks/z.png')


def generate_block(x_start, y_start) -> Block:
    block_list = ['i', 't', 'o', 'l', 'j', 's', 'z']

    random.shuffle(block_list)

    block_type = block_list[0]

    if block_type == 'i':
        block = Block(x_start, y_start, 24, 96, I_BLOCK_IMAGE, SQUARE_RED_BLOCK_IMAGE)

    elif block_type == 't':
        block = Block(x_start, y_start, 72, 48, T_BLOCK_IMAGE, SQUARE_GREY_BLOCK_IMAGE)

    elif block_type == 'o':
        block = Block(x_start, y_start, 48, 48, O_BLOCK_IMAGE, SQUARE_CYAN_BLOCK_IMAGE)

    elif block_type == 'l':
        block = Block(x_start, y_start, 48, 72, L_BLOCK_IMAGE, SQUARE_YELLOW_BLOCK_IMAGE)

    elif block_type == 'j':
        block = Block(x_start, y_start, 48, 72, J_BLOCK_IMAGE, SQUARE_MAGENTA_BLOCK_IMAGE)

    elif block_type == 's':
        block = Block(x_start, y_start, 72, 48, S_BLOCK_IMAGE, SQUARE_BLUE_BLOCK_IMAGE)

    elif block_type == 'z':
        block = Block(x_start, y_start, 72, 48, Z_BLOCK_IMAGE, SQUARE_GREEN_BLOCK_IMAGE)

    return block




