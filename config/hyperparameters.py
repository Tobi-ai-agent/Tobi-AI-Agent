# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 100000
TARGET_UPDATE = 10  # Update target network every N episodes
UPDATE_EVERY = 4    # Update network every N steps

# Model parameters
INPUT_CHANNELS = 3  # RGB images
IMAGE_SIZE = (224, 224)  # Size to resize images to

# Hardware acceleration
USE_MPS = True  # Use Metal Performance Shaders on Mac