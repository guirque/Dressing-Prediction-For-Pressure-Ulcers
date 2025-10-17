from torch import nn

def GetConvBlock(in_channels, list_conv, maxpool=True):
    """
        Returns a sequential block consisting of 
        multiple convolutional layers and an optional
        maxpool layer of k = 2 and s = 2.

        *in_channels*: number of channels as input for the first convolutional layer of the block.
        *list_conv*: list of lists. Each sublist contains [num_kernels, k, s, p]
    """

    list_layers = []
    i = 0
    for conv in list_conv:
        list_layers.append(
            nn.Conv2d(in_channels, conv[0], conv[1], conv[2], conv[3])
        )
        list_layers.append(
            nn.LeakyReLU()
        )
        in_channels = conv[0]

    if maxpool:
        list_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*list_layers)

    # https://stackoverflow.com/questions/3941517/converting-list-to-args-when-calling-function

class YOLOModel(nn.Module):

    def __init__(self, img_size=448, num_classes=20, grid_size=7, num_bounding_boxes = 2, num_channels=3):
        super().__init__()

        # output_size = (width - kernel_size + 2*padding)/stride + 1 
        # k = 2, s = 2, p = 0 -> output_size = w/2
        # k = 7, s = 2, p = 1 -> output_size = (w - 3)/2 (ceil value)
        # k = 3, s = 1, p = 1 -> output_size = w
        # k = 1, s = 1, p = 0 -> output_size = w

        self.grid_size = grid_size # S
        self.num_classes = num_classes # C
        self.num_bounding_boxes = num_bounding_boxes # B

        # https://arxiv.org/pdf/1506.02640
        self.layers = nn.Sequential(

            # First block reduces input to 1/4
            nn.Conv2d(num_channels, 64, 7, 2, 3), # k = 7, s = 2, num_k = 64 (outputs), p = 3; output_size = w/2 (values are floored)
            nn.MaxPool2d(2, 2, 0), # k = 2, s = 2, p = 0

            # All the other blocks have convolutional layers that maintain size. Maxpools cut by half.
            # For 3x3 kernels, we'll maintain size with k=3, p=1, s=1
            # For 1x1 kernels, we'll maintain size with k=1, s=1, p=0
            
            # Block 2
            GetConvBlock(64, [
                [192, 3, 1, 1],
            ], maxpool=True),

            # Block 3
            GetConvBlock(192, [
                [128, 1, 1, 0],
                [256, 3, 1, 1],
                [256, 1, 1, 0],
                [512, 3, 1, 1]
            ], maxpool=True),

            # Block 4
            GetConvBlock(512, [
                [256, 1, 1, 0],
                [512, 3, 1, 1],

                [256, 1, 1, 0],
                [512, 3, 1, 1],

                [256, 1, 1, 0],
                [512, 3, 1, 1],

                [256, 1, 1, 0],
                [512, 3, 1, 1],

                [512, 1, 1, 0],
                [1024, 3, 1, 1]
            ], maxpool=True),

            # Block 5
            GetConvBlock(1024, [
                [512, 1, 1, 0],
                [1024, 3, 1, 1],

                [512, 1, 1, 0],
                [1024, 3, 1, 1],

                [1024, 3, 1, 1],

                [1024, 3, 2, 1]  # w = 14. Output should be floored for us to get w = 7, with p = 1
            ], maxpool=False),

            # Block 6

            GetConvBlock(1024, [
                [1024, 3, 1, 1],
                [1024, 3, 1, 1]
            ], maxpool=False),

            # output is supposed to be of size 7 * 7 * 1024 (w, h, c)

            # # Flatten 
            nn.Flatten(),
            
            # Dense Layers
            nn.Linear(7*7*1024, 4096),
            nn.Linear(4096, self.grid_size * self.grid_size* (self.num_bounding_boxes * 5 + self.num_classes) ) # output should be of shape S * S * (B * 5 + C)
        )

        # https://discuss.pytorch.org/t/basic-question-about-number-of-kernels-used-for-torch-nn-conv2d-3-32-3/186963/6
        # https://iaee.substack.com/p/yolo-intuitively-and-exhaustively

    def forward(self, x):

        # Running model
        x = self.layers(x)

        # Reshaping Output
        return x.reshape(-1, self.grid_size, self.grid_size, (self.num_bounding_boxes * 5 + self.num_classes))