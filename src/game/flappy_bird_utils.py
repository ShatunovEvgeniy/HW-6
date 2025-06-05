import random
from pathlib import Path
import pygame

SPRITES_PATH = Path(__file__).parent.parent.parent / "assets" / "sprites"

# list of all possible players (tuple of 3 positions of flap)
PLAYERS_LIST = (
    # red bird
    (
        str(SPRITES_PATH / "redbird-upflap.png"),
        str(SPRITES_PATH / "redbird-midflap.png"),
        str(SPRITES_PATH / "redbird-downflap.png"),
    ),
    # blue bird
    (
        str(SPRITES_PATH / "bluebird-upflap.png"),
        str(SPRITES_PATH / "bluebird-midflap.png"),
        str(SPRITES_PATH / "bluebird-downflap.png"),
    ),
    # yellow bird
    (
        str(SPRITES_PATH / "yellowbird-upflap.png"),
        str(SPRITES_PATH / "yellowbird-midflap.png"),
        str(SPRITES_PATH / "yellowbird-downflap.png"),
    ),
)

# list of backgrounds
BACKGROUNDS_LIST = (
    str(SPRITES_PATH / "background-day.png"),
    str(SPRITES_PATH / "background-night.png"),
)

# list of pipes
PIPES_LIST = (
    str(SPRITES_PATH / "pipe-green.png"),
    str(SPRITES_PATH / "pipe-red.png"),
)


def load():
    # path of player with different states
    PLAYER_PATH = (
        str(SPRITES_PATH / "redbird-upflap.png"),
        str(SPRITES_PATH / "redbird-midflap.png"),
        str(SPRITES_PATH / "redbird-downflap.png"),
    )

    # path of pipe
    PIPE_PATH = str(SPRITES_PATH / "pipe-green.png")

    IMAGES, HITMASKS = {}, {}

    # numbers sprites for score display
    IMAGES["numbers"] = (
        pygame.image.load(SPRITES_PATH / "0.png").convert_alpha(),
        pygame.image.load(SPRITES_PATH / "1.png").convert_alpha(),
        pygame.image.load(SPRITES_PATH / "2.png").convert_alpha(),
        pygame.image.load(SPRITES_PATH / "3.png").convert_alpha(),
        pygame.image.load(SPRITES_PATH / "4.png").convert_alpha(),
        pygame.image.load(SPRITES_PATH / "5.png").convert_alpha(),
        pygame.image.load(SPRITES_PATH / "6.png").convert_alpha(),
        pygame.image.load(SPRITES_PATH / "7.png").convert_alpha(),
        pygame.image.load(SPRITES_PATH / "8.png").convert_alpha(),
        pygame.image.load(SPRITES_PATH / "9.png").convert_alpha(),
    )

    # base (ground) sprite
    IMAGES["base"] = pygame.image.load(SPRITES_PATH / "base.png").convert_alpha()

    # select random background sprites
    randBg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
    IMAGES["background"] = pygame.image.load(BACKGROUNDS_LIST[randBg]).convert()

    # select random player sprites
    randPlayer = random.randint(0, len(PLAYERS_LIST) - 1)
    IMAGES["player"] = (
        pygame.image.load(PLAYERS_LIST[randPlayer][0]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][1]).convert_alpha(),
        pygame.image.load(PLAYERS_LIST[randPlayer][2]).convert_alpha(),
    )

    # select random pipe sprites
    pipeindex = random.randint(0, len(PIPES_LIST) - 1)
    IMAGES["pipe"] = (
        pygame.transform.flip(pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(), False, True),
        pygame.image.load(PIPES_LIST[pipeindex]).convert_alpha(),
    )

    # hismask for pipes
    HITMASKS["pipe"] = (
        getHitmask(IMAGES["pipe"][0]),
        getHitmask(IMAGES["pipe"][1]),
    )

    # hitmask for player
    HITMASKS["player"] = (
        getHitmask(IMAGES["player"][0]),
        getHitmask(IMAGES["player"][1]),
        getHitmask(IMAGES["player"][2]),
    )

    return IMAGES, HITMASKS


def getHitmask(image):
    """returns a hitmask using an image's alpha."""
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask
