import jaxpruner
import jaxpruner.base_updater
import ml_collections
import jax
import optax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from functools import partial
from typing import Any
from enum import Enum

from my_networks import Network
from my_datasets import Dataset


class SparseAlgorithm(Enum):
    """
    Enum class to represent the different sparse algorithms.
    """
    SET = "set"
    RIGL = "rigl"
    NO_PRUNE = "no_prune"


class TrainState(train_state.TrainState):
    """
    Extension of train_state.TrainState
    Also able to store the batch_stats for convolutional neural networks
    """
    batch_stats: Any = None


class EfficiencyCalculator():
    """
    Helper class to calculate the expected efficiency of a model.
    Without this class, the algorithm, sparsity, update_freq, train_size and epochs would have to be added to self of SparseOptimization.
    """
    def __init__(self, network, dataset, algorithm, sparsity, update_freq, train_size, rng) -> None:
        self.network: Network = network
        self.dataset: Dataset = dataset
        self.algorithm: str = algorithm
        self.sparsity: float = sparsity
        self.update_freq: int = update_freq
        self.train_size: int = train_size
        self.rng: Any = rng
    
    
    def __get_inference_flops_old(self, params) -> tuple[float, float]:
        """
        Alternative method to get the inference flops
        Kept here for reference and later comparison
        More precise method, using the exact FLOP calculations of Flax.
        However, more error prone and does not always work on cluster.
        ------------------------------
        Calculate the dense and sparse inference flops
        Uses the network's tabulate calculation of the flops.
        """
        self.rng, rng = jax.random.split(self.rng)
        sample_input = jnp.ones([1, *self.dataset.get_input_dimensions()])
        total_dense_inference_flops, individual_dense_inference_flops = self.network.get_inference_flops_from_tabulate(sample_input, rng)
        total_sparse_inference_flops = 0
        layer_sparsities = jaxpruner.summarize_sparsity(params)
        for path, flops in individual_dense_inference_flops:
            if any(module in path for module in ['Dense', 'Conv']):
                total_sparse_inference_flops += flops * (1.0 - layer_sparsities[path + "/kernel"])
            if 'BatchNorm' in path:
                total_sparse_inference_flops += flops
        return total_dense_inference_flops, total_sparse_inference_flops

    
    def __get_inference_flops(self, params) -> tuple[float, float]:
        """
        Gets the inference flops per layer from the network
        Calculates the total dense flops as a sum of these layers
        Sparse flops calculations adds a sparsity multiplier to this sum
        Returns dense_inference_flops_per_sample, sparse_inference_flops_per_sample
        """
        layer_inference_flops = self.network.get_inference_flops(self.dataset.get_input_dimensions())
        layer_sparsities = jaxpruner.summarize_sparsity(params)
        dense_inference_flops_per_sample = 0
        sparse_inference_flops_per_sample = 0
        for layer_name, flops in layer_inference_flops:
            dense_inference_flops_per_sample += flops
            if layer_name in layer_sparsities:
                sparse_inference_flops_per_sample += flops * (1 - float(layer_sparsities[layer_name]))
        return dense_inference_flops_per_sample, sparse_inference_flops_per_sample
        
    
    def get_model_efficiency(self, params) -> tuple[float, float]:
        """
        Calculate the expected total inference flops and training flops of the model.
        Returns: sparse_inference_flops_per_sample, total_training_flops
        """
        dense_inference_flops_per_sample, sparse_inference_flops_per_sample = self.__get_inference_flops(params)
        if self.algorithm == SparseAlgorithm.NO_PRUNE.value:
            sparse_training_flops_per_sample = 3 * dense_inference_flops_per_sample
        elif self.algorithm == SparseAlgorithm.SET.value:
            sparse_training_flops_per_sample = 3 * sparse_inference_flops_per_sample
        elif self.algorithm == SparseAlgorithm.RIGL.value:
            # The training flops for RigL are slightly more complex due to calculations of the gradients every update_freq
            # Line split into 4 at an attempt to increase readability
            sparse_training_flops_per_sample = (
                3 * sparse_inference_flops_per_sample * self.update_freq +
                2 * sparse_inference_flops_per_sample + dense_inference_flops_per_sample
                ) / (self.update_freq + 1)
        
        # Multiple sparse_training_flops_per_sample by the number of samples it is trained on
        total_training_flops = sparse_training_flops_per_sample * self.train_size
        
        return sparse_inference_flops_per_sample, total_training_flops
    
    

class SparseOptimization():
    """
    Class that is able to train a neural network model using sparse optimization techniques given a hyperparameter configuration.
    Keeps track of the current trainstate, and has functions to train, test and validate the model.
    """

    def __init__(
        self,
        dataset: Dataset,
        architecture: str = 'mlp',
        algorithm: str = 'set',
        sparsity: float = 0.8,
        sparsity_distribution: str = 'uniform',
        num_mlp_layers: int = 1,
        size_first_mlp_layer: int = 1,
        size_middle_mlp_layer: int = 1,
        size_last_mlp_layer: int = 1,
        conv_stage_1: int = 1,
        conv_stage_2: int = 0,
        conv_stage_3: int= 0,
        conv_stage_4: int= 0,
        update_freq: int = 10,
        update_end: float = 0.75,
        learning_rate: float = 0.01,
        learning_rate_scheduler: str = 'constant',
        momentum: float = 0.9,
        weight_decay: float = 0.001,
        use_dropout: bool | str = False,
        dropout: float = 0,
        use_label_smoothing: bool | str = False,
        label_smoothing: float = 0,
        batch_size: int = 128,
        epochs: int = 1,
        prng_key: int = 0,
        **kwargs,
        ) -> None:
        """
        Initialize the SparseOptimization class with the given configuration.
        Initialization parameters:
        dataset: Dataset object containing the train, validation and test data
        architecture: String representing the architecture of the model. Options: 'mlp', 'resnet32', 'resnet224'
        algorithm: String representing the sparse training algorithm. Options: 'set', 'rigl', 'no_prune'
        sparsity: Float representing the desired sparsity level of the model. 0.0 represents no sparsity, 1.0 represents full sparsity.
        sparsity_distribution: String representing the distribution of the sparsity. Options: 'uniform', 'erk'
        update_freq: Integer representing the frequency of the sparsity updates
        update_end: Float indicating at what percentage of training to stop with sparsity updates
        learning_rate: Float representing the learning rate for stochastic gradient descent (SGD)
        learning_rate_scheduler: String representing the learning rate scheduler. Options: 'constant', 'cosine'
        momentum: Float representing the momentum for SGD
        weight_decay: Float representing the weight decay for SGD
        dropout: Float representing the dropout rate for the model
        label_smoothing: Float representing the label smoothing factor
        batch_size: Integer representing the batch size for training
        epochs: Integer representing the number of epochs to train the model
        prng_key: Integer representing the seed for the random number generator
        kwargs: Additional keywords arguments. Currently unused, but are allowed to be passed.
        if architecture == 'mlp':
            num_mlp_layers: Integer representing the number of layers in the MLP model. Sizes of the layers are linearly scaled between the first, middle and last layer.
            size_first_mlp_layer: Integer representing the size of the first layer in the MLP model
            size_middle_mlp_layer: Integer representing the size of the middle layers in the MLP model. Only used if num_mlp_layers > 2
            size_last_mlp_layer: Integer representing the size of the last layer in the MLP model. Only used if num_mlp_layers > 1
        if architecture == 'resnet32' or 'resnet224':
            conv_stage_1: Integer representing the number of filters in the first convolutional stage
            conv_stage_2: Integer representing the number of filters in the second convolutional stage
            conv_stage_3: Integer representing the number of filters in the third convolutional stage
            conv_stage_4: Integer representing the number of filters in the fourth convolutional stage (only used in 'resnet224')
        """        
        if type(use_dropout) != bool:
            use_dropout = use_dropout == 'True'
        if type(use_label_smoothing) != bool:
            use_label_smoothing = use_label_smoothing == 'True'

        self.rng = jax.random.PRNGKey(prng_key)
        self.dataset = dataset
        self.steps_per_epoch = dataset.get_train_size() // batch_size
        self.label_smoothing = label_smoothing if use_label_smoothing else 0
        self.state, self.sparsity_updater, self.efficiency_calculator = self.__prepare_configuration(
            architecture=architecture,
            algorithm=algorithm,
            sparsity=sparsity,
            sparsity_distribution=sparsity_distribution,
            update_freq=update_freq,
            update_end=update_end,
            num_mlp_layers=num_mlp_layers,
            size_first_mlp_layer=size_first_mlp_layer,
            size_middle_mlp_layer=size_middle_mlp_layer,
            size_last_mlp_layer=size_last_mlp_layer,
            conv_stage_1=conv_stage_1,
            conv_stage_2=conv_stage_2,
            conv_stage_3=conv_stage_3,
            conv_stage_4=conv_stage_4,
            learning_rate=learning_rate,
            learning_rate_scheduler=learning_rate_scheduler,
            momentum=momentum,
            weight_decay=weight_decay,
            use_dropout=use_dropout,
            dropout=dropout,
            batch_size=batch_size,
            epochs=epochs,
        )        
        
    def __prepare_configuration(
        self,
        architecture: str,
        algorithm: str,
        sparsity: float,
        sparsity_distribution: str,
        update_freq: int,
        update_end: float,
        num_mlp_layers: int,
        size_first_mlp_layer: int,
        size_middle_mlp_layer: int,
        size_last_mlp_layer: int,
        conv_stage_1: int,
        conv_stage_2: int,
        conv_stage_3: int,
        conv_stage_4: int,
        learning_rate: float,
        learning_rate_scheduler,
        momentum: float,
        weight_decay: float,
        use_dropout: bool,
        dropout: float,
        batch_size: int,
        epochs: int,
        ) -> tuple[TrainState, jaxpruner.BaseUpdater, EfficiencyCalculator]:
        """
        Generate a network and optimizer based on the configuration.
        Returns: train_state, sparsity_updater, efficiency_calculator
        """
        num_classes = self.dataset.get_output_dimensions()
        network = Network(architecture, num_classes, num_mlp_layers, size_first_mlp_layer, size_middle_mlp_layer, size_last_mlp_layer, conv_stage_1, conv_stage_2, conv_stage_3, conv_stage_4, use_dropout, dropout)
        # network.get_network_summary_table(jnp.ones([1, *self.dataset.get_input_dimensions()]), self.rng)

        self.rng, rng = jax.random.split(self.rng)
        efficiency_calculator = EfficiencyCalculator(network, self.dataset, algorithm, sparsity, update_freq, epochs * self.dataset.get_train_size(), rng)

        total_steps = epochs * self.steps_per_epoch
        sparsity_updater, optimizer = self.__create_optimizer(algorithm, sparsity, sparsity_distribution, update_freq, total_steps, update_end, learning_rate, learning_rate_scheduler, momentum, weight_decay)

        state = self.__create_train_state(network.get_flax_network(), optimizer, batch_size)
        
        return state, sparsity_updater, efficiency_calculator

    def __create_optimizer(
        self,
        algorithm: str,
        sparsity: float,
        sparsity_distribution: str,
        update_freq: int,
        total_steps: int,
        update_end: float,
        learning_rate: float,
        learning_rate_scheduler: str,
        momentum: float,
        weight_decay: float
        ) -> tuple[jaxpruner.BaseUpdater, optax.GradientTransformationExtraArgs]:
        """
        Create a sparsity updater and optax optimizer using the given configuration.
        """
        update_end_step = int(total_steps * update_end)
        if update_end_step <= update_freq:
            update_end_step = update_freq + 1
        
        sparsity_config = ml_collections.ConfigDict()
        sparsity_config.algorithm = algorithm
        sparsity_config.sparsity = sparsity
        sparsity_config.dist_type = sparsity_distribution
        sparsity_config.update_start_step = 1
        sparsity_config.update_end_step = update_end_step
        sparsity_config.update_freq = update_freq
        sparsity_updater = jaxpruner.create_updater_from_config(sparsity_config)

        if learning_rate_scheduler == 'cosine':
            scheduler = optax.cosine_decay_schedule(learning_rate, total_steps, alpha=1e-4)
        elif learning_rate_scheduler == 'constant':
            scheduler = optax.constant_schedule(learning_rate)
            
        tx = optax.chain(
            optax.add_decayed_weights(weight_decay), 
            optax.sgd(scheduler, momentum),
        )
        tx = sparsity_updater.wrap_optax(tx)

        return sparsity_updater, tx



    def __create_train_state(
        self,
        network,
        optimizer,
        batch_size: int,
        ) -> TrainState:
        """
        Initializes the network and creates a TrainState object with starting parameters.
        """
        self.rng, rng = jax.random.split(self.rng)
        input_shape = (batch_size, *self.dataset.get_input_dimensions())
        init = network.init(rng, jnp.ones(input_shape))

        params = init['params']
        batch_stats = init.get('batch_stats', None)

        state = TrainState.create(
            apply_fn=network.apply,
            params=params,
            batch_stats=batch_stats,
            tx=optimizer,
        )
        
        return state

    
    @partial(jax.jit, static_argnums=(0, 4, 5))
    def __apply_model(
        self,
        state: TrainState,
        X,
        y,
        num_classes: int,
        train: bool = False,
        rng: Any = None,
        label_smoothing: bool = 0,
        ) -> tuple[Any, Any, Any, Any]:
        """
        Computes gradients, loss, accuracy and batch_stats for a single batch.
        If train == True, rng should be added to the function for dropout.
        Jitted functions should not use values in self directly, thus they are passed as arguments.
        """
        def loss_fn(params: Any, batch_stats: Any) -> tuple[float, tuple[Any, Any]]:
            batch_stats_updates = None
            variables = {'params': params}
            if batch_stats:
                variables['batch_stats'] = batch_stats
            if train:
                if batch_stats:
                    logits, updates = state.apply_fn(variables, X, train=True, mutable=['batch_stats'], rngs={'dropout': rng})
                    batch_stats_updates = updates['batch_stats']
                else:
                    logits = state.apply_fn(variables, X, train=True, rngs={'dropout': rng})
            else: 
                logits = state.apply_fn(variables, X)
            one_hot = jax.nn.one_hot(y, num_classes)
            one_hot = optax.losses.smooth_labels(one_hot, label_smoothing)
            loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))            
            return loss, (logits, batch_stats_updates)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (logits, batch_stats)), gradients = grad_fn(state.params, state.batch_stats)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == y)
        
        return gradients, loss, accuracy, batch_stats
    

    @partial(jax.jit, static_argnums=(0,))
    def __update_model(
        self,
        state: TrainState,
        gradients,
        new_batch_stats
        ) -> TrainState:
        """
        Given the calculated gradients, update the model parameters.
        Jitted functions should not use values in self directly, thus they are passed as arguments.
        """
        updates, new_opt_state = state.tx.update(gradients, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return state.replace(
            step=state.step + 1,
            params=new_params,
            batch_stats=new_batch_stats,
            opt_state=new_opt_state
            )
            

    def train_model(self) -> tuple[float, float]:
        """
        Train for a single epoch.
        """
        self.rng, dropout_rng = jax.random.split(self.rng)

        epoch_loss = []
        epoch_accuracy = []
        post_op = jax.jit(self.sparsity_updater.post_gradient_update)
        
        for batch_input, batch_labels in self.dataset.get_train_iterator():
            gradients, loss, accuracy, batch_stats = self.__apply_model(
                self.state,
                batch_input,
                batch_labels,
                self.dataset.get_output_dimensions(),
                train=True,
                rng=dropout_rng,
                label_smoothing=self.label_smoothing
                )
            state = self.__update_model(self.state, gradients, batch_stats)

            post_params = post_op(state.params, state.opt_state)
            self.state = state.replace(params=post_params)
            
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)
            
        train_loss = np.mean(epoch_loss)
        train_accuracy = np.mean(epoch_accuracy)
        return train_loss, train_accuracy


    def validate_model(self) -> tuple[float, float]:
        """
        Return the total loss and accuracy of the trained model over the validation set.
        """
        epoch_loss = []
        epoch_accuracy = []
        
        for batch_input, batch_labels in self.dataset.get_val_iterator():
            _, loss, accuracy, _ = self.__apply_model(
                self.state,
                batch_input,
                batch_labels,
                self.dataset.get_output_dimensions(),
                )
            epoch_loss.append(loss)
            epoch_accuracy.append(accuracy)
        
        loss = np.mean(epoch_loss)
        accuracy = np.mean(epoch_accuracy)
        
        return loss, accuracy
        

    def get_model_efficiency(self) -> tuple[float, float]:
        """
        Stub function that calls the EfficiencyCalculator.
        To ensure that we correctly use sparsity, the current parameters are pruned before calculating the efficiency.
        """
        sparsity_op = self.sparsity_updater.post_gradient_update
        sparsified_params = sparsity_op(self.state.params, self.state.opt_state)
        return self.efficiency_calculator.get_model_efficiency(sparsified_params)
