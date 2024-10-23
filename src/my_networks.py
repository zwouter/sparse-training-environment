from flax import linen as nn
from functools import partial
from enum import Enum
import jax.numpy as jnp
import numpy as np
import math
import logging


logger = logging.getLogger(__name__)


class NetworkArchitecture(Enum):
    MLP = 'mlp'
    RESNET = 'resnet'


class LayerType(Enum):
    DENSE = 'dense'
    RELU = 'relu'
    FLATTEN = 'flatten'
    CONV = 'conv'
    NORM = 'norm'
    POOL = 'pool'
    RES_START = 'res_start'
    RES_END = 'res_end'
    MEAN = 'mean'
    SOFTMAX = 'softmax'


class Network():
    """
    Wrapper class that defines the structure of a network and its layers.
    Can create a network from a given architecture and hyperparameters.
    Can be used to calculate the efficiency of the network and to fet the flax linen network.
    Layers are defined by a list of tuples of the form (LayerType, dict).
    From such a list, we can dynamically create MLPs and ResNets and calculate their inference FLOPs.
    """
    network: nn.Module
    layers: list[tuple[LayerType, dict]]
    num_classes: int
    architecture: str
    
    def __init__(self, architecture, num_classes = 1, num_mlp_layers = 1, size_first_mlp_layer = 1, size_middle_mlp_layer = 1, size_last_mlp_layer = 1, conv_stage_1 = 1, conv_stage_2 = 1, conv_stage_3 = 1, conv_stage_4 = 1, use_dropout = True, dropout = 0) -> None:
        if architecture == NetworkArchitecture.MLP.value:
            layers = self.__get_mlp_layers(num_mlp_layers, size_first_mlp_layer, size_middle_mlp_layer, size_last_mlp_layer, num_classes)
        elif architecture == NetworkArchitecture.RESNET.value:
            layers = self.__get_resnet_layers(conv_stage_1, conv_stage_2, conv_stage_3, num_classes)
        else:
            raise ValueError(f"Architecture {architecture} not supported.")
        
        self.layers = layers
        self.num_classes = num_classes
        self.architecture = architecture
        self.network = self.__Flax_Network(layers, use_dropout, dropout)
    
    
    def get_flax_network(self) -> nn.Module:
        return self.network

    
    def get_inference_flops(self, input_dimensions) -> list[tuple[str, float]]:
        """
        Calculates the number of flops (multiplications) needed for inference of each layer of the network.
        Creates a list of tuples of the form (layer_name, flops) for each layer.
        """
        flops = []
        if self.architecture == NetworkArchitecture.MLP.value:
            prev_layer_size = np.prod(input_dimensions)
            for layer_type, layer_info in self.layers:
                if layer_type == LayerType.DENSE:
                    flops.append((layer_info['name'] + "/kernel", float(prev_layer_size * layer_info['features'])))
                    # flops.append((layer_info['name'], "bias", layer_info['features']))
                    prev_layer_size = layer_info['features']
        if self.architecture == NetworkArchitecture.RESNET.value:
            # ResNets are used for image data, so we assume the input is 3D
            width, height, channels = input_dimensions
            for layer_type, layer_info in self.layers:
                if layer_type == LayerType.CONV:
                    width = round(width / layer_info.get('strides', (1, 1))[0])
                    height = round(height / layer_info.get('strides', (1, 1))[1])
                    
                    # Conv FLOPs = output_width * output_height * kernel_width * kernel_height * input_channels * output_channels
                    flops.append((layer_info['name'] + "/kernel", float(height * width * layer_info.get('kernel_size', (0, 0))[1] * layer_info.get('kernel_size', (0, 0))[0] * channels * layer_info['features'])))
                    channels = layer_info['features']
                elif layer_type == LayerType.POOL:
                    # Emit FLOPs for the pooling layer, but it does change the size
                    width = round(width / layer_info.get('strides', (1, 1))[0])
                    height = round(height / layer_info.get('strides', (1, 1))[1])
                elif layer_type == LayerType.DENSE:
                    # ResNets only have 1 dense layer at the end, before which the input is flattened to the number of channels.
                    flops.append((layer_info['name'] + "/kernel", float(channels * layer_info['features'])))
        return flops


    def get_network_summary_table(self, input, rng) -> str:
        """
        Have Flax build a summary table of the network.
        Prints the table and returns it as a string.
        Does not work on the HPC cluster.
        """
        logger.info("Creating network summary table using Flax tabulate.")
        tabulate_fn = nn.tabulate(self.network, rng, compute_flops=True, compute_vjp_flops=False, console_kwargs={'width': 1000000, 'soft_wrap': True})
        table = tabulate_fn(input)
        print(table) # Logger.info ruins the styling
        return table


    def get_inference_flops_from_tabulate(self, input, rng) -> tuple[int, list[tuple[str, int]]]:
        """
        Does not always work on all machines!
        Have Flax calculate the number of flops for the inference of the network.
        Only possibly through the tabulate method, which automatically builds a string table.
        Returns the total inference flops, and a list of the inference flops per layer.
        SparseOptimization can also calculate FLOPs itself using the same underlying calculations as tabulate.
        """
        table = self.get_network_summary_table(input, rng)
        flops = []
        for row in table.split("\n"):
            cells = row.split("│")
            if len(cells) > 4 :
                flops_cell = row.split("│")[5].replace(" ", "")
                path_cell = row.split("│")[1].replace(" ", "")
                if flops_cell.isnumeric():
                    flops.append((path_cell, int(flops_cell)))
        return flops[0], flops[1:]


    def __get_mlp_layers(self, num_mlp_layers, size_first_mlp_layer, size_middle_mlp_layer, size_last_mlp_layer, num_classes) -> list[tuple[LayerType, dict]]:
        """
        Create the structure of a neural network.
        Dense etwork has the number of layers specified in the layers parameter.
        The sizes of the layers between the first, middle and last layers are linearly interpolated.
        Example networks: input (5, 1, 5, 10)    -> (1-3-5-8-10)
                          input (8, 100, 40, 80) -> (100-80-60-40-50-60-70-80)
                          input (3, 100, 40, 80) -> (100-40-80)
                          input (2, 100, 40, 80) -> (100-80)
                          input (1, 100, 40, 80) -> (100)
        """
        # Calculate the sizes of all dense layers
        dense_layers = []
        dense_layers.append(size_first_mlp_layer)

        if num_mlp_layers > 1:
            if num_mlp_layers > 2:
                size_cur_layer = size_first_mlp_layer
                middle_layer = math.ceil(num_mlp_layers / 2)

                step_size_between_layers = (size_middle_mlp_layer - size_first_mlp_layer) / (middle_layer - 1)
                num_layers = num_mlp_layers - middle_layer - 1 if num_mlp_layers % 2 == 0 else num_mlp_layers - middle_layer
                for _ in range(num_layers):
                    size_cur_layer += step_size_between_layers
                    dense_layers.append(round(size_cur_layer))

                step_size_between_layers = (size_last_mlp_layer - size_middle_mlp_layer) / (num_mlp_layers - middle_layer)
                num_layers = middle_layer - 1 if num_mlp_layers % 2 == 0 else middle_layer - 2
                for _ in range(num_layers):
                    size_cur_layer += step_size_between_layers
                    dense_layers.append(round(size_cur_layer))

            dense_layers.append(size_last_mlp_layer)
        
        # Change to parsable format
        layers = []
        layers.append((LayerType.FLATTEN, {}))
        for i, size in enumerate(dense_layers):
            layers.append((LayerType.DENSE, {
                'features': size,
                'name': f'dense_{i}'
            }))
            layers.append((LayerType.RELU, {}));
        layers.append((LayerType.DENSE, {
            'features': num_classes,
            'name': f'dense_{i + 1}'
        }))
        return layers
    
    
    def __get_resnet_layers(self, conv_stage_1, conv_stage_2, conv_stage_3, num_classes) -> list[tuple[LayerType, dict]]:
        layers = []
        
        layers.append((LayerType.CONV, {
            'features': 16,
            'kernel_size': (3, 3),
            'strides': (1, 1),
            'padding': [(3, 3), (3, 3)],
            'name': 'conv_init'
            }))
        layers.append((LayerType.NORM, {'name': 'bn_init'}))
        layers.append((LayerType.RELU, {}))
        
        for i, block_size in enumerate([conv_stage_1, conv_stage_2, conv_stage_3]):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                filters = 16 * 2**i
                
                # ResNet Block
                layers.append((LayerType.RES_START, {}))
                layers.append((LayerType.CONV, {
                    'features': filters,
                    'kernel_size': (3, 3),
                    'strides': strides,
                    'name': f'conv_{i + 1}_{j + 1}_1'
                }))
                layers.append((LayerType.NORM, {'name': f'bn_{i + 1}_{j + 1}_1'}))
                layers.append((LayerType.RELU, {}))
                layers.append((LayerType.CONV, {
                    'features': filters,
                    'kernel_size': (3, 3),
                    'name': f'conv_{i + 1}_{j + 1}_2'
                }))
                layers.append((LayerType.NORM, {
                    'scale_init': nn.initializers.zeros_init(),
                    'name': f'bn_{i + 1}_{j + 1}_2'
                }))
                layers.append((LayerType.RES_END, {
                    'kernel_size': (1, 1),
                    'features': filters,
                    'strides': strides,
                    'name': f'conv_{i + 1}_{j + 1}_3'
                }))
        layers.append((LayerType.MEAN, {'axis': (1, 2)}))
        layers.append((LayerType.DENSE, {
            'features': num_classes,
            'dtype': jnp.float32,
            'name': 'dense'
        }))
        return layers
    
    
    class __Flax_Network(nn.Module):
        layers: list[tuple[LayerType, dict]]
        use_dropout: bool
        dropout: float
        
        @nn.compact
        def __call__(self, x, train: bool = False):
            conv = partial(nn.Conv, use_bias=False, dtype=jnp.float32)
            norm = partial(
                nn.BatchNorm,
                use_running_average=not train,
                momentum=0.9,
                epsilon=1e-5,
                dtype=jnp.float32,
            )               
            
            residual = None
            for layer, kwargs in self.layers:
                if layer == LayerType.FLATTEN:
                    x = x.reshape((x.shape[0], -1))
                elif layer == LayerType.DENSE:
                    x = nn.Dense(**kwargs)(x)
                    if self.use_dropout:
                        x = nn.Dropout(rate=self.dropout, deterministic=not train)(x)
                elif layer == LayerType.RELU:
                    x = nn.relu(x)
                elif layer == LayerType.CONV:
                    x = conv(**kwargs)(x)
                elif layer == LayerType.NORM:
                    x = norm(**kwargs)(x)
                elif layer == LayerType.POOL:
                    x = nn.max_pool(x, **kwargs)
                elif layer == LayerType.RES_START:
                    residual = x
                elif layer == LayerType.RES_END:
                    if residual.shape != x.shape:
                        residual = conv(**kwargs)(residual)
                        residual = norm()(residual)
                    x = nn.relu(residual + x)
                elif layer == LayerType.MEAN:
                    x = jnp.mean(x, **kwargs)
                elif layer == LayerType.SOFTMAX:
                    x = nn.softmax(x, **kwargs)
                else:
                    raise ValueError(f"Layer type {layer} not supported.")
            return x
        