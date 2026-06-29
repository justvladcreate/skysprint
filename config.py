import os

BORDERLESS_FULLSCREEN = True
FPS = 60
TRAINING_FPS = 30  # Частота кадров при визуализации обучения (не используется в новом коде, можно убрать)

# Управление вращением носа
ROTATION_UP = 0.04
ROTATION_RETURN = 0.05
TARGET_ANGLE = -0.5236

# Физика
THRUST_FORCE = 1.2
GLIDE_FORCE = 0.25
GRAVITY = 0.3
DRAG = 0.02
MAX_SPEED = 8

# Лучи
RAY_COUNT = 5
RAY_ANGLES = [-0.8, -0.4, 0.0, 0.4, 0.8]
RAY_MAX_DIST = 400.0

# Параметры обучения по умолчанию
DEFAULT_POP_SIZE = 50
DEFAULT_ELITE_COUNT = 5
DEFAULT_MAX_STEPS = 2000
DEFAULT_MUTATION_RATE = 0.1   # уже не используется напрямую, но оставлено для совместимости
DEFAULT_MUTATION_SCALE = 0.2
DEFAULT_INPUT_SIZE = 11
DEFAULT_HIDDEN_SIZES = (64, 64)

# Папки
LEVELS_FOLDER = "levels"
MODELS_FOLDER = "models"