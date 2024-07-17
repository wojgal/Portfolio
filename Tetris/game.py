import pygame
import sys
from block import *
from events import *



class Tetris:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption('Tetris')

        self.clock = pygame.time.Clock()
        self.window_width = 288
        self.window_height = 528
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.fps = 60
        self.block = generate_block(96, 24)
        self.background_image = pygame.image.load('images/background.png')

        self.map_walls = pygame.sprite.Sprite()
        self.map_walls.image = self.background_image
        self.map_walls.rect = self.map_walls.image.get_rect()

        self.map_collision_group = pygame.sprite.Group()
        self.map_collision_group.add(self.map_walls)

        self.blocks_collision_group = pygame.sprite.Group()

        self.score = 0

        self.font = pygame.font.Font(pygame.font.get_default_font(), 32)

        pygame.mixer.music.load('sounds/theme.mp3')
        pygame.mixer.music.set_volume(0.25)
        pygame.mixer.music.play(-1)
        self.clear_sound = pygame.mixer.Sound('sounds/clear.mp3')
        self.clear_sound.set_volume(0.25)
        self.game_over_sound = pygame.mixer.Sound('sounds/game_over.mp3')
        self.game_over_sound.set_volume(0.25)



    def draw(self) -> None:
        self.screen.fill(0)
        self.map_collision_group.draw(self.screen)
        self.blocks_collision_group.draw(self.screen)
        self.block.draw(self.screen)
        score_text = self.font.render(str(self.score), False, (255, 0, 0), None)
        self.screen.blit(score_text, ((self.window_width - score_text.get_width()) // 2, 0))
        pygame.display.update()


    def add_points(self, lines_clear):
        if lines_clear == 1:
            self.score += 40

        elif lines_clear == 2:
            self.score += 100

        elif lines_clear == 3:
            self.score += 300

        elif lines_clear == 4:
            self.score += 1200



    def check_completed_rows(self) -> list:
        lines_clear = 0
        square_size = 24
        y = 21

        while y > 1:
            y-= 1
            block_amount = 0

            for x in range(1, 11):
                square = Block(x * square_size, y * square_size, square_size, square_size, SQUARE_RED_BLOCK_IMAGE, SQUARE_RED_BLOCK_IMAGE)

                if pygame.sprite.spritecollide(square, self.blocks_collision_group, False, pygame.sprite.collide_mask):
                    block_amount += 1

            if block_amount == 10:
                for x in range(1, 11):
                    square = Block(x * square_size, y * square_size, square_size, square_size, SQUARE_RED_BLOCK_IMAGE, SQUARE_RED_BLOCK_IMAGE)

                    pygame.sprite.spritecollide(square, self.blocks_collision_group, True, pygame.sprite.collide_mask)
                
                self.blocks_collision_group.update(y * square_size)
                lines_clear += 1
                y = 21

        if lines_clear > 0:
            self.add_points(lines_clear)
            self.clear_sound.play()

                


    def reset_board(self) -> None:
        self.blocks_collision_group.empty()
        self.block = generate_block(96, 24)
        self.score = 0



    def check_block_destroyed(self) -> None:
        if pygame.sprite.spritecollide(self.block, self.map_collision_group, False, pygame.sprite.collide_mask) or pygame.sprite.spritecollide(self.block, self.blocks_collision_group, False, pygame.sprite.collide_mask):
            self.block.move_up()

            x_cord, y_cord = self.block.get_coord()
            square_size = 24

            for y in range(4):
                for x in range(4):
                    square = Block(x_cord + x*square_size, y_cord + y * square_size, square_size, square_size, self.block.get_square_image(), self.block.get_square_image)
                    block_group = pygame.sprite.Group()
                    block_group.add(self.block)

                    if pygame.sprite.spritecollide(square, block_group, False, pygame.sprite.collide_mask):
                        self.blocks_collision_group.add(square)

            self.block = generate_block(96, 24)
            self.check_completed_rows()

            if pygame.sprite.spritecollide(self.block, self.blocks_collision_group, False, pygame.sprite.collide_mask):
                pygame.mixer.music.stop()
                pygame.time.delay(100)
                self.game_over_sound.play()
                pygame.time.delay(2000)
                self.reset_board()
                pygame.mixer.music.play()



    def game(self) -> None:
        pygame.time.set_timer(MOVE_DOWN_BLOCK, 500)

        while True:
            self.clock.tick(self.fps)

            for event in pygame.event.get():
                
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == MOVE_DOWN_BLOCK:
                    self.block.move_down()

                    self.check_block_destroyed()

                keys_pressed = pygame.key.get_pressed()

                if keys_pressed[pygame.K_s] or keys_pressed[pygame.K_DOWN]:
                    self.block.move_down()

                    self.check_block_destroyed()

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a or event.key == pygame.K_LEFT:
                        self.block.move_left()

                        if pygame.sprite.spritecollide(self.block, self.blocks_collision_group, False, pygame.sprite.collide_mask) or pygame.sprite.spritecollide(self.block, self.map_collision_group, False, pygame.sprite.collide_mask):
                            self.block.move_right()

                    if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                        self.block.move_right()

                        if pygame.sprite.spritecollide(self.block, self.blocks_collision_group, False, pygame.sprite.collide_mask) or pygame.sprite.spritecollide(self.block, self.map_collision_group, False, pygame.sprite.collide_mask):
                            self.block.move_left()

                    if event.key == pygame.K_SPACE:
                        self.block.rotate()

                        while pygame.sprite.spritecollide(self.block, self.blocks_collision_group, False, pygame.sprite.collide_mask) or pygame.sprite.spritecollide(self.block, self.map_collision_group, False, pygame.sprite.collide_mask):
                            self.block.move_left()



            self.draw()

