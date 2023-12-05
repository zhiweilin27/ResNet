import torch
from torch.functional import Tensor
import torch.nn as nn

""" This script defines the network.
"""

class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.
        
        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.
        
        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """

        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        self.start_layer = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size =3, stride=1, padding=1,bias=False)
        
        ### YOUR CODE HERE
        
        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
                eps=1e-5, 
                momentum=0.1,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters))
        self.output_layer = output_layer(filters*4, self.resnet_version, self.num_classes)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self,  num_features, eps=1e-5, momentum=0.1) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.batch_norm = nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.1)
        self.relu = nn.ReLU()
     
        ### YOUR CODE HERE
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        norm_data = self.batch_norm(inputs)
        outputs = self.relu(norm_data)
        return outputs
        ### YOUR CODE HERE
        
class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
                 convolution.
        projection_shortcut: The function to use for projection shortcuts
                             (typically a 1x1 convolution when downsampling the input).
        strides: A positive integer. The stride to use for the block. If
                 greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
                           first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        self.filters = filters
        self.stride = strides
        self.projection_shortcut = projection_shortcut
        
        self.conv1 = nn.Conv2d(filters//strides, filters, kernel_size=3, stride=strides, padding=1, bias=False)
        self.bn_relu1 = batch_norm_relu_layer(filters,eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters,eps=1e-5, momentum=0.1)
        self.relu2 = nn.ReLU()
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        residual = inputs
        out = self.conv1(inputs)
        out = self.bn_relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.projection_shortcut is not None:
            residual = self.projection_shortcut(residual)
        out += residual
        out = self.relu2(out)
        return out

        ### YOUR CODE HERE

class bottleneck_block(nn.Module):
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()
        ### YOUR CODE HERE
        # Hint: Different from standard lib implementation, you need pay attention to 
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.
        self.filters = filters
        self.projection_shortcut = projection_shortcut
        self.strides = strides
        self.first_num_filters = first_num_filters
        in_filters = self.filters
        if self.projection_shortcut is not None:
            if filters  == first_num_filters * 4:
                in_filters = self.first_num_filters
            else:
                in_filters = self.filters // 2
                
        hidden_filters = self.filters // 4
    
        self.bn_relu1 = batch_norm_relu_layer(in_filters,eps=1e-5, momentum=0.1)
        self.conv1 = nn.Conv2d(in_filters, hidden_filters, kernel_size=1, stride=1, padding=0,bias=False)
        self.bn_relu2 = batch_norm_relu_layer(hidden_filters,eps=1e-5, momentum=0.1)
        self.conv2 = nn.Conv2d(hidden_filters, hidden_filters, kernel_size=3, stride=self.strides, padding=1,bias=False)
        self.bn_relu3 = batch_norm_relu_layer(hidden_filters,eps=1e-5, momentum=0.1)
        self.conv3 = nn.Conv2d(hidden_filters, self.filters, kernel_size=1, stride=1, padding=0,bias=False)
        
        ### YOUR CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.
        #print('before bn_relue1:',inputs.shape)
        residual = inputs
        out = self.bn_relu1(inputs)
        #print('after bn_relu1:', inputs.shape)
        out = self.conv1(out)
        #print('after conv1:', out.shape)
        out = self.bn_relu2(out)
        #print('after bn_relu2:', out.shape)
        out = self.conv2(out)
        #print('after conv2:', out.shape)
        out = self.bn_relu3(out)
        #print('after bn_relu3:', out.shape)
        out = self.conv3(out)
        #print('after conv3:', out.shape)
        if self.projection_shortcut is not None:
            residual = self.projection_shortcut(residual)
        out += residual
        #print('after addition:', out.shape)
        return out
        ### YOUR CODE HERE
        
                
class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """

    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()
        
        filters_out = filters * 4 if block_fn is bottleneck_block else filters
        
        self.strides = strides  
        self.resnet_size = resnet_size
        self.blocks = nn.ModuleList()
        
        for i in range(resnet_size):
            projection_shortcut = None
            self.strides = strides if i ==0 else 1
            if block_fn is standard_block and i== 0:
                 if first_num_filters != filters_out:
                    projection_shortcut = nn.Sequential(
                                              nn.Conv2d(filters_out//2, filters_out, stride=strides, kernel_size=1,bias=False),
                                              nn.BatchNorm2d(filters_out,eps=1e-5, momentum=0.1))
            
            elif block_fn is bottleneck_block:
                if first_num_filters == filters_out // 4 and i == 0:
                    projection_shortcut = nn.Sequential(
                            nn.Conv2d(filters_out//4, filters_out, kernel_size = 1, stride=strides, padding=0,bias=False),
                            nn.BatchNorm2d(filters_out,eps=1e-5, momentum=0.1)
                    )
                elif first_num_filters < filters_out and i == 0:
                    projection_shortcut = nn.Sequential(
                            nn.Conv2d(filters_out//2, filters_out, kernel_size = 1, stride=strides, padding=0,bias=False),
                            nn.BatchNorm2d(filters_out,eps=1e-5, momentum=0.1)
                    )
            self.blocks.append(block_fn(filters_out, projection_shortcut, self.strides, first_num_filters))
                
    def forward(self, inputs: Tensor):
        outputs = inputs 
        for block in self.blocks:
            outputs = block(outputs)
        return outputs

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        ### END CODE HERE
        if resnet_version == 2:
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.1)
        self.resnet_version = resnet_version
        self.filters = filters if resnet_version == 2 else filters//4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.filters, num_classes)
        
    def forward(self, inputs: Tensor):
        if self.resnet_version == 2:
            inputs = self.bn_relu(inputs)

        inputs = self.avgpool(inputs)
        inputs = torch.flatten(inputs, 1)
        outputs = self.fc(inputs)

        return outputs
