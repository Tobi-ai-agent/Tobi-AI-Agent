
# CNN Architecture settings
CNN_LAYERS = [
    # (out_channels, kernel_size, stride)
    (32, 8, 4),
    (64, 4, 2),
    (64, 3, 1),
    (128, 3, 1)
]

FC_LAYERS = [512, 256]  # Hidden layer sizes for fully connected layers

# Advanced model settings
USE_BATCH_NORM = True
DROPOUT_RATE = 0.2  # Set to 0 to disable dropout
WEIGHT_INIT = 'xavier'  # Options: 'xavier', 'kaiming', 'normal'

# Optional: Feature extraction settings
USE_PRETRAINED = False  # Whether to use transfer learning with pretrained model
PRETRAINED_MODEL = 'resnet18'  # Options: 'resnet18', 'resnet34', 'mobilenet_v2'
FREEZE_LAYERS = True  # Whether to freeze pretrained layers during training