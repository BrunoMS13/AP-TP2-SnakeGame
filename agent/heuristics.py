from game.snake_game import SnakeGame

# go straight for food
# important: considering only 1 apple
# directions: 0=up, 1=right, 2=down, 3=left
# x grows from left to right, starting at 0
# y grows from top to bottom, starting at 0
def min_distance_heuristic(game: SnakeGame) -> int:
    score, apple, head, tail, direction = game.get_state()
    head_x, head_y = head
    apple_x, apple_y = apple[0]
    optimal_direction = 0
    # check what is the "optimal" direction to go to the apple
    if apple_x > head_x:
        optimal_direction = 1
    elif apple_x < head_x:
        optimal_direction = 3
    elif apple_y > head_y:
        optimal_direction = 2
    else: # apple_y < head_y
        optimal_direction = 0
    # check which way to turn
    if direction == 0: # up
        if optimal_direction == 1: # right
            return 2
        elif optimal_direction == 3: # left
            return 0
    elif direction == 1: # right
        if optimal_direction == 0: # up
            return 0
        elif optimal_direction == 2: # down
            return 2
    elif direction == 2: # down
        if optimal_direction == 1: # right
            return 0
        elif optimal_direction == 3: # left
            return 2
    else: # direction == 3, left
        if optimal_direction == 0: # up
            return 2
        elif optimal_direction == 2: # down
            return 0
    return 1


def do_not_hit_yourself_heuristic(game: SnakeGame) -> int:
    score, apple, head, tail, direction = game.get_state()
    head_x, head_y = head
    # check which sides of the head are free
    up = (head_x, head_y-1) not in tail
    right = (head_x+1, head_y) not in tail
    down = (head_x, head_y+1) not in tail
    left = (head_x-1, head_y) not in tail

    # then turn to the side of the apple if possible
    go_up = False
    go_right = False
    go_down = False
    go_left = False
    apple_x, apple_y = apple[0]
    if apple_x > head_x and right:
        go_right = True
    elif apple_x < head_x and left:
        go_left = True
    elif apple_y > head_y and down:
        go_down = True
    elif apple_y < head_y and up:
        go_up = True
    
    if direction == 0: # up
        if go_right:
            return 1
        elif go_left:
            return -1
    elif direction == 1: # right
        if go_up:
            return -1
        elif go_down:
            return 1
    elif direction == 2: # down
        if go_right:
            return -1
        elif go_left:
            return 1
    else: # direction == 3, left
        if go_up:
            return 1
        elif go_down:
            return -1
    return 0
