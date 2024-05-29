from game.game_wrapper import SnakeGameWrapper

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

TURN_LEFT = 0
GO_STRAIGHT = 1
TURN_RIGHT = 2


class Heuristic:
    def get_action(self, game: SnakeGameWrapper) -> int:
        raise NotImplementedError


class MinDistanceHeuristic(Heuristic):
    # x grows from top to bottom, starting at 0
    # y grows from left to right, starting at 0
    def get_action(self, game: SnakeGameWrapper) -> int:
        score, apple, head, tail, direction = game.game.get_state()
        head_x, head_y = head
        apple_x, apple_y = apple[0]

        # Determine the optimal direction to go to the apple
        dx = apple_x - head_x
        dy = apple_y - head_y

        if direction == UP:
            if dy > 0:
                return TURN_RIGHT  # apple is to the right
            elif dy < 0:
                return TURN_LEFT  # apple is to the left
            elif dx < 0:
                return GO_STRAIGHT  # apple is straight ahead
        elif direction == RIGHT:
            if dx > 0:
                return TURN_RIGHT  # apple is downwards
            elif dx < 0:
                return TURN_LEFT  # apple is upwards
            elif dy > 0:
                return GO_STRAIGHT  # apple is straight ahead
        elif direction == DOWN:
            if dy > 0:
                return TURN_LEFT  # apple is to the right
            elif dy < 0:
                return TURN_RIGHT  # apple is to the left
            elif dx > 0:
                return GO_STRAIGHT  # apple is straight ahead
        elif direction == LEFT:
            if dx > 0:
                return TURN_LEFT  # apple is downwards
            elif dx < 0:
                return TURN_RIGHT  # apple is upwards
            elif dy < 0:
                return GO_STRAIGHT  # apple is straight ahead

        # If none of the above conditions are met, return TURN_RIGHT as a fallback
        return TURN_RIGHT
