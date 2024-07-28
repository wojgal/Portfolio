import pygame
import sys
from field import *
from button import * 
from events import *
from sounds import *



class TicTacToe:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("TicTacToe")

        self.clock = pygame.time.Clock()
        self.window_width = 1000
        self.window_height = 500
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        self.fps = 60

        self.current_move = 'x'
        self.next_move_possible = True
        self.board = [Field(y, x) for x in range(75, 400, 125) for y in range(75, 400, 125)]
        self.image_board = pygame.image.load('images/board.png')

        self.button_1v1 = Button(650, 50, ONE_VS_ONE_IMAGE, ONE_VS_ONE_PRESSED_IMAGE)
        self.button_1v1.set_active(True)
        self.button_ai = Button(650, 150, AI_BUTTON_IMAGE, AI_PRESSED_BUTTON_IMAGE)
        self.button_restart = Button(690, 400, RESTART_IMAGE, RESTART_IMAGE)
        self.button_x_player = Button(830, 165, X_PLYAER_IMAGE, X_PLYAER_PRESSED_IMAGE)
        self.button_x_player.set_active(True)
        self.button_x_player.set_visible(False)
        self.button_o_player = Button(905, 165, O_PLYAER_IMAGE, O_PLYAER_PRESSED_IMAGE)
        self.button_o_player.set_visible(False)
        self.buttons = [self.button_1v1, self.button_ai, self.button_restart, self.button_x_player, self.button_o_player]

        self.label_x_win = Button(650, 275, X_WIN_IMAGE, X_WIN_IMAGE)
        self.label_x_win.set_visible(False)
        self.label_o_win = Button(650, 275, O_WIN_IMAGE, O_WIN_IMAGE)
        self.label_o_win.set_visible(False)
        self.label_draw = Button(650, 275, DRAW_IMAGE, DRAW_IMAGE)
        self.label_draw.set_visible(False)
        self.labels = [self.label_x_win, self.label_o_win, self.label_draw]

        self.gamemode = '1v1'
        self.vs_ai_player_sign = 'x'
        self.ai_sign = 'o'

    
    def reset_board(self):
        for field in self.board:
            field.clear_type()

        for lbl in self.labels:
            lbl.set_visible(False)

        self.current_move = 'x'
        self.next_move_possible = True


    def board_to_list(self) -> list:
        board_list = []

        for field in self.board:
            board_list.append(field.get_type())

        return board_list
    

    def switch_current_move(self) -> None:
        if self.current_move == 'x':
            self.current_move = 'o'

        elif self.current_move == 'o':
            self.current_move = 'x'


    def button_1v1_action(self) -> None:
        self.gamemode = '1v1'
        self.button_1v1.set_active(True)
        self.button_ai.set_active(False)
        self.button_x_player.set_visible(False)
        self.button_o_player.set_visible(False)
        self.reset_board()
        CLICK_SOUND.play()


    def button_ai_action(self) -> None:
        self.gamemode = 'ai'
        self.button_1v1.set_active(False)
        self.button_ai.set_active(True)
        self.button_x_player.set_visible(True)
        self.button_o_player.set_visible(True)
        self.reset_board()
        CLICK_SOUND.play()


    def button_restart_action(self) -> None:
        self.reset_board()
        CLICK_SOUND.play()


    def button_x_player_action(self) -> None:
        self.vs_ai_player_sign = 'x'
        self.ai_sign = 'o'
        self.button_x_player.set_active(True)
        self.button_o_player.set_active(False)
        self.reset_board()
        CLICK_SOUND.play()


    def button_o_player_action(self) -> None:
        self.vs_ai_player_sign = 'o'
        self.ai_sign ='x'
        self.button_o_player.set_active(True)
        self.button_x_player.set_active(False)
        self.reset_board()
        CLICK_SOUND.play()


    def draw(self):
        self.screen.fill(0)
        self.screen.blit(self.image_board, (0, 0))

        for field in self.board:
            field.draw(self.screen)

        for btn in self.buttons:
            btn.draw(self.screen)

        for lbl in self.labels:
            lbl.draw(self.screen)

        pygame.display.update()


    def check_win(self, board) -> str:
        '''
        0 1 2
        3 4 5
        6 7 8
        '''

        first_row = board[0] + board[1] + board[2]
        second_row = board[3] + board[4] + board[5]
        third_row = board[6] + board[7] + board[8]

        first_column = board[0] + board[3] + board[6]
        second_column = board[1] + board[4] + board[7]
        third_column = board[2] + board[5] + board[8]

        first_diagonal = board[0] + board[4] + board[8]
        second_diagonal = board[2] + board[4] + board[6]

        all_winning_lines = [first_row, second_row, third_row, 
                             first_column, second_column, third_column, 
                             first_diagonal, second_diagonal]
        
        for line in all_winning_lines:
            if line == 'xxx':
                return 'x'
            
            if line =='ooo':
                return 'o'
            
        for field in board:
            if field == 'None':
                return False
        else:
            return 'draw'
        
    
    def check_win_event(self):
        winning = self.check_win(self.board_to_list())

        if winning:
            if winning == 'x':
                self.label_x_win.set_visible(True)

            elif winning == 'o':
                self.label_o_win.set_visible(True)

            elif winning == 'draw':
                self.label_draw.set_visible(True)

            self.next_move_possible = False
            GAME_END_SOUND.play()
        

    def minimax(self, board, depth, is_maximizing, max_depth) -> int:
        winning = self.check_win(board)

        if winning == self.ai_sign:
            return 1
        
        if winning == self.vs_ai_player_sign:
            return -1
        
        if winning == 'draw' or depth == max_depth:
            return 0

        if is_maximizing:
            max_evaluation = -float('inf')

            for idx in range(9):
                if board[idx] == 'None':
                    board[idx] = self.ai_sign

                    evaluation = self.minimax(board, depth + 1, False, max_depth)

                    board[idx] = 'None'

                    max_evaluation = max(max_evaluation, evaluation)
            
            return max_evaluation
        
        if not is_maximizing:
            min_evaluation = float('inf')

            for idx in range(9):
                if board[idx] == 'None':
                    board[idx] = self.vs_ai_player_sign

                    evaluation = self.minimax(board, depth + 1, True, max_depth)

                    board[idx] = 'None'

                    min_evaluation = min(min_evaluation, evaluation)

            return min_evaluation
        

    def get_ai_move(self) -> int:
        best_evaluation = -float('inf')
        best_move = None
        board = self.board_to_list()


        for idx in range(9):
            if board[idx] == 'None':
                board[idx] = self.ai_sign
                evaluation = self.minimax(board, 0, False, 9)
                board[idx] = 'None'

                if evaluation > best_evaluation:
                    best_evaluation = evaluation
                    best_move = idx

        return best_move
    

    def ai_move_event(self):
        if not self.next_move_possible:
            return
        
        if self.current_move == self.ai_sign:
            ai_move = self.get_ai_move()
            ai_field = self.board[ai_move]
            ai_field.set_type(self.ai_sign)

            self.switch_current_move()
            pygame.event.post(CHECK_WIN_EVENT)
            DRAW_SOUND.play()


    def game(self) -> None:
        while True:
            self.clock.tick(self.fps)

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()


                if event.type == pygame.MOUSEBUTTONDOWN:

                    if self.button_1v1.check_click() and not self.button_1v1.is_active():
                        self.button_1v1_action()

                    if self.button_ai.check_click() and not self.button_ai.is_active():
                        self.button_ai_action()

                    if self.button_restart.check_click():
                        self.button_restart_action()

                    if self.button_x_player.check_click() and not self.button_x_player.is_active():
                        self.button_x_player_action()

                    if self.button_o_player.check_click() and not self.button_o_player.is_active():
                        self.button_o_player_action()

                    if self.gamemode == '1v1' and self.next_move_possible:
                        for field in self.board:
                            if field.check_click() and field.is_empty():
                                field.set_type(self.current_move)
                                
                                self.switch_current_move()

                                pygame.event.post(CHECK_WIN_EVENT)
                                DRAW_SOUND.play()
                                break
                        
                if self.gamemode == 'ai' and self.next_move_possible:
                    if self.vs_ai_player_sign == 'x':
                        if self.current_move == 'x' and event.type == pygame.MOUSEBUTTONDOWN:
                            for field in self.board:
                                if field.check_click() and field.is_empty():
                                    field.set_type(self.vs_ai_player_sign)
                                    self.switch_current_move()

                                    pygame.event.post(CHECK_WIN_EVENT)
                                    DRAW_SOUND.play()
                                    break

                        elif self.current_move == 'o':
                            pygame.time.set_timer(AI_MOVE_EVENT, 1000, 1)

                    elif self.vs_ai_player_sign == 'o':
                        if self.current_move == 'x':
                            pygame.time.set_timer(AI_MOVE_EVENT, 1000, 1)
                            
                        elif self.current_move == 'o' and event.type == pygame.MOUSEBUTTONDOWN:
                            for field in self.board:
                                if field.check_click() and field.is_empty():
                                    field.set_type(self.vs_ai_player_sign)
                                    self.switch_current_move()

                                    pygame.event.post(CHECK_WIN_EVENT)
                                    DRAW_SOUND.play()
                                    break

                if event == AI_MOVE_EVENT:
                    self.ai_move_event()
                    

                if event == CHECK_WIN_EVENT:
                    self.check_win_event()


            self.draw()



