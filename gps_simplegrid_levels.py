import copy
import os
import random
import time
import math
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Type, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
import tyro
from gymnasium.vector.utils import create_empty_array
from stable_baselines3.common.vec_env import VecTransposeImage
from torch.utils.tensorboard import SummaryWriter
from wandb.sdk.wandb_run import Run

from generative.models.actor_cnn import Actor
from generative.models.critic_cnn import Critic
from generative.models.dense_autoencoder.db.generate_valid_sequences import generate_valid_sequences, \
    sequence_to_one_hot
from generative.models.dense_autoencoder.decoder_utils import get_decoder_api, ActionGen
from generative.models.dense_autoencoder.dense_var_auto_encoder_vin_1 import DenseVAE

from gym_simplegrid.envs.simple_grid_levels import (
    RewardStrategy,
    ObservationEncodingStrategy,
    PartialObservabilityStrategy,
)
from modules.replay_memory import ReplayMemory, Transition
from simplegrid_dataset import PositionXY, simplegrid_dataset_setup

import wandb
from utils.torch_utils import SoftAverager


class SaveModelStrategy(Enum):
    REWARD = "reward"
    SUCCESS_RATE = "success_rate"


class ActionSeqRepresentation(Enum):
    ACTION_SEQ_AS_INT = "action_seq_as_int"
    ACTION_SEQ_AS_ONE_HOT = "action_seq_as_one_hot"
    ACTION_SEQ_AS_PROB = "action_seq_as_prob"


@dataclass
class Config:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 123
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    mps: bool = False
    """if toggled, mps will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "grid-experiments"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model_to_wandb: bool = False
    """whether to upload model to the wandb project"""
    save_model_strategy: SaveModelStrategy = SaveModelStrategy.SUCCESS_RATE
    """Specifies the criteria for saving the best model"""
    verbose: int = 1
    """verbosity of the experiment logging"""
    verbose_steps_interval: int = 5000
    """The interval between printing training statistics"""
    validate_actor_weights_update: bool = False
    """whether to validate that actor's weights are updated during training"""
    slurm_job_id: Optional[int] = None
    """the experiment's slurm job id"""

    # Dataset specific arguments
    val_eval_freq: int = 5000
    """the frequency of evaluation during training on validation environments"""
    train_eval_freq: int = 5000
    """the frequency of evaluation during training on training environments"""
    eval_test_dataset_during_training_freq: int = 100000
    """the frequency of evaluation during training on test environments"""
    train_dataset_size: Optional[int] = 100
    """the size of the training dataset"""
    train_eval_dataset_size: Optional[int] = 100
    """the size of the training dataset for evaluation"""
    val_dataset_size: Optional[int] = 100
    """the size of the val dataset"""
    test_dataset_size: Optional[int] = 1000
    """the size of the test dataset"""

    # Env specific arguments
    env_id: str = "SimpleGrid-v0"
    """the id of the environment"""
    max_episode_steps: int = 75
    """the environment episode maximum number steps"""
    reward_strategy: RewardStrategy = RewardStrategy.NEGATIVE_BASED_ON_MAX_LEVEL_WITH_PENALTIES
    """specifies the reward strategy for the environment"""
    penalize_cyclic_position_revisits: bool = False
    """
    Flag to penalize cyclic or redundant position revisits during navigation.
    When True, tracks and potentially penalizes movement sequences that:
    - Immediately backtrack (e.g., up then down, left then right)
    - Return to previously visited positions
    Helps enforce more directed and goal-oriented movement in grid-based environments.
    """
    observation_encoding_strategy: ObservationEncodingStrategy = ObservationEncodingStrategy.DEFAULT
    """specifies the observation encoding strategy for the environment"""
    partial_observability_strategy: PartialObservabilityStrategy = PartialObservabilityStrategy.FULL
    """specifies the partial observability strategy for the environment"""
    view_radius: int = 3
    """radius of the agent's view when using partial observability (LOCAL_VIEW strategy)"""
    is_slippery: bool = False
    """if True, actions have stochastic outcomes with perpendicular movement probabilities"""
    slippery_prob: float = 1/3
    """probability of executing intended action when is_slippery=True (remaining probability split equally between perpendicular directions)"""
    sticky_action_prob: float = 0.0
    """probability of repeating the previous action instead of executing the intended action"""
    random_action_prob: float = 0.0
    """probability of selecting a uniformly random action instead of executing the intended action"""
    obstacle_map: str = "8x8_empty"
    """the obstacle map of the environment"""
    max_level: int = 14
    """the environment level of the dataset's difficulty"""
    start_level: int = 1
    """the environment start level of the dataset's difficulty"""
    end_of_sequence_token: int = 4
    """token signifying the end of a sequence (e.g., token 4)"""
    action_space_size: int = 4
    """number of possible actions in the environment"""

    # Algorithm specific arguments
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    actor_learning_rate: float = 1e-4
    """the learning rate of the actor network optimizer"""
    critic_learning_rate: float = 1e-4
    """the learning rate of the critic network optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 50000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """the target network update rate"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 1000
    """timestep to start learning"""
    actor_policy_frequency: int = 4
    """the frequency of actor training policy (delayed)"""
    push_sub_sequences_to_buffer_move_start_point: bool = True
    """control whether sub-sequences that the start point of the proto-plan is changing are saved to buffer"""
    push_sub_sequences_to_buffer_move_end_point: bool = True
    """control whether sub-sequences that the end point of the proto-plan is changing are saved to buffer"""
    sub_sequences_min_jump_move_start_point: int = 1
    """minimum number of jumps between elements in a proto-plan sequence to create a sub-sequence for move start point case. For example, if the sequence is [1, 2, 3, 4, 5] and this value is set to 2, then sub-sequences
       like [2, 3, 4, 5] and [4, 5] will be generated"""
    sub_sequences_min_jump_move_end_point: int = 1
    """minimum number of jumps between elements in a proto-plan sequence to create a sub-sequence for move end point case. For example, if the sequence is [1, 2, 3, 4, 5] and this value is set to 2, then sub-sequences
       like [1, 2, 3] and [1] will be generated"""
    push_every_one_step_transition_to_buffer: bool = True
    """control whether to store every transition into the buffer"""
    actor_target_network_frequency: int = 100
    """actor's frequency of updates for the target network"""
    critic_target_network_frequency: int = 10
    """critic's frequency of updates for the target network"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.1
    """the ending epsilon for exploration"""
    total_steps_e: int = 15000
    """the total number of steps for epsilon decay"""
    action_seq_representation: ActionSeqRepresentation = ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT
    "Specifies how the critic evaluates the Actions sequence"
    sample_from_candidates: float = 0.1
    """Specifies if to sample from the candidates based on the q-values distribution"""
    explore_without_critic: bool = True
    """Specifies if to explore without critic"""
    exploration_sequence_length_constraint: int = -1
    """Specifies if the actor should explore using a sequence length constraint, filtering valid sequences based on their length.
        A value of -1 disables the constraint; any positive integer sets a maximum sequence length."""
    use_adaptive_temperature: bool = True
    """Determines whether to use an adaptive temperature for weighting the losses from different heads.
       If True, the temperature decays over time to focus more on a specific head later in training."""
    temperature: float = 20
    """The temperature parameter used for weighting the losses from different heads.
       Higher temperatures lead to more uniform weighting (more exploration across heads).
       As temperature decreases (e.g., via adaptive decay), the model focuses more on the best-performing head."""
    diversity_weight: float = -1
    """The weight applied to the diversity regularization term in the actor loss function.
       Higher values encourage the model to produce more diverse embeddings from different heads.
       A value of -1 disables this regularization"""
    sequence_length_weight: float = -1
    """The weight applied to the sequence length regularization term in the actor loss function.
       A value of -1 disables this regularization."""
    min_lengths_per_head: list[int] = field(default_factory=lambda: [1, 2, 4, 6, 8])
    """Specifies the minimum sequence lengths expected from each head.
       Each element in the list corresponds to a head and defines the minimum length
       for sequences generated by that head."""
    max_lengths_per_head: list[int] = field(default_factory=lambda: [2, 4, 6, 8, 10])
    """Specifies the maximum sequence lengths expected from each head.
       Each element in the list corresponds to a head and defines the maximum length
       for sequences generated by that head."""
    invalid_actions_weight: float = -1
    """The weight applied to the invalid actions regularization term in the actor loss function.
       This regularization penalizes invalid actions in the sequences. A value of -1 disables this regularization."""
    evaluate_actor_diversity_every_k_epochs: float = 50000
    """if value > 0 the algorithm will calculate actor sequences on given start-end tuples and monitor the diversity along the training"""

    # actor params:
    actor_n_output_channels: int = 16
    """the size of the proto-action-plan which is the decoder's input"""
    n_proto_plan_candidates: int = 1
    """the number of different proto plan candidates embeddings entities to generated actor's proto-action-plan"""

    actor_linear_layers: list[int] = field(default_factory=lambda: [512, 128, 32])
    """the hidden layer sizes for the actor's linear layers"""
    actor_linear_layers_activation_function: str = "leaky_relu"
    """activation function to use in the actor's linear layers"""
    actor_use_batchnorm_linear_layers: bool = False
    """whether to use batch normalization in the actor's linear layers"""
    actor_weight_decay: float = 1e-4
    """the weight decay for the actor's network"""
    actor_grad_clip: float = 0.0
    """if > 0, clip the global gradient norm of the actor (and, in end-to-end mode, the
       decoder) to this value before stepping. Off by default. Typical actor gradient norms
       are 0.2-0.4, so a clip around 1.0 only bites on spikes. Motivated by two runs that
       destabilised the *actor* specifically: a larger decoder (n_layer 3 / n_embd 64)
       NaN'd at step 159k with the NaN appearing in cnn.0.weight, and actor_learning_rate
       3e-4 scored 0.445. Both amplify what reaches the actor. Note this is preventive, not
       curative: clipping cannot repair a gradient that is already NaN, it only limits the
       growth that leads there"""
    pe_embedding_dim: int = 128
    """the embedding dimension of the position encoding for concatenation with the CNN output in the actor network. Set to -1 to disable position encoding."""
    exclude_decoder_from_computation_graph: bool = False
    """whether to exclude the decoder from the computation graph"""
    train_decoder_end_to_end: bool = False
    """if True, train decoder end-to-end with actor/critic (GPS-E2E mode). If False, use frozen pretrained decoder (GPS mode)"""
    decoder_learning_rate: float = 1e-5
    """the learning rate of the decoder network optimizer (only used when train_decoder_end_to_end=True)"""
    decoder_weight_decay: float = 1e-3
    """the weight decay for the decoder's network (only used when train_decoder_end_to_end=True)"""
    initialize_decoder_from_pretrained: bool = True
    """if True, initialize decoder from pretrained checkpoint even in E2E mode. If False, use random initialization (only relevant when train_decoder_end_to_end=True)"""

    # critic params:
    min_qf_value: Optional[float] = None
    """the minimum Q-value for the critic"""
    max_qf_value: float = 1.0
    """the maximum Q-value for the critic"""
    critic_linear_layers: list[int] = field(default_factory=lambda: [512, 128, 32])
    """the hidden layer sizes for the critic's linear layers"""
    critic_linear_layers_activation_function: str = "leaky_relu"
    """activation function to use in the critic's linear layers"""
    critic_use_batchnorm_linear_layers: bool = False
    """whether to use batch normalization in the critic's linear layers"""
    critic_weight_decay: float = 1e-4
    """the weight decay for the critic's network"""
    include_actor_embedding_in_critic_input: bool = False
    """Whether to include the actor's output embedding as part of the critic's input, alongside the decoder's action sequence"""

    # decoder params:
    decoder_model_path: str = "generative/models/dense_autoencoder/denseAE_generic_2act_seq_VAE_v7_bs=32_epochs=19999-20000_lr=0.0001_end_size_16_var_1_leakyrelu_normalized_by_seq_len_without_aug_instance_norm.pt"

    """model path of pretrained VAE decoder"""
    n_actions_in_seq: int = 10
    """the number of actions in the decoder output"""
    decoder_n_layer: int = 2
    """number of Transformer blocks in the minGPT decoder trunk. The decoder is run
       autoregressively (one trunk pass per generated token), so this is close to a linear
       knob on decoder wall-clock cost"""
    decoder_n_head: int = 4
    """number of attention heads in the minGPT decoder (must divide decoder_n_embd)"""
    decoder_n_embd: int = 32
    """embedding / hidden width of the minGPT decoder"""
    decoder_dropout: float = 0.0
    """dropout rate applied to the minGPT decoder's embeddings, attention and residuals.
       Defaults to 0 to match the old MLP decoder, which had no dropout at all: any
       non-zero value adds noise to generated action sequences on every path where the
       decoder is in train mode"""
    decoder_use_cuda_graph: bool = False
    """replay the decoder's autoregressive decode loop from a captured CUDA graph on the
       gradient-free paths (action selection, target-Q, evaluation). The loop issues ~550
       tiny kernels per call and is almost entirely dispatch overhead, so this is worth
       ~8x at batch 1 and ~2x at batch 256. The actor update stays eager so gradients
       still flow. Requires CUDA; falls back to eager with a warning if capture fails"""
    decoder_soft_token_feedback: bool = False
    """feed the emitted one-hot row back into the next decode step instead of a detached
       argmax index. Generation is unchanged (for a one-hot row, row @ tok_emb.weight ==
       tok_emb(argmax)); only the backward pass differs, letting the critic's gradient
       reach z through all 10 positions rather than only the prefix token. Aimed at the
       actor failing to learn state-dependent embeddings -- see §8.6 of
       TRANSFORMER_INTEGRATION.md. Only applies to the one-hot emission modes"""
    use_gumble_in_decoder: bool = True
    """
    If True, the decoder's `gen_action_seq` function uses the Gumbel-Softmax method, which preserves gradients for backpropagation.
    If False, it employs a deterministic argmax operation with a Straight-Through Estimator (STE)
    """
    deterministic_inference: bool = True
    """
    If True, the decoder will operate in a deterministic fashion, 
    ensuring that the same input always results in the same output.
    If False, the decoder will utilize Gumbel sampling for 
    stochastic action sequence generation, introducing randomness.
    """

    def __post_init__(self):
        if self.total_steps_e < self.learning_starts:
            raise ValueError("total_steps_e must be larger than or equal to learning_starts")
        if self.total_steps_e > self.total_timesteps:
            raise ValueError("total_steps_e must be smaller than or equal to total_timesteps")
        if self.sub_sequences_min_jump_move_start_point > self.n_actions_in_seq:
            raise ValueError("sub_sequences_min_jump_move_start_point must be smaller to n_actions_in_seq")
        if self.sub_sequences_min_jump_move_end_point > self.n_actions_in_seq:
            raise ValueError("sub_sequences_min_jump_move_end_point must be smaller to n_actions_in_seq")
        if self.train_decoder_end_to_end and self.exclude_decoder_from_computation_graph:
            raise ValueError("Cannot train decoder end-to-end when exclude_decoder_from_computation_graph is True")
        if not self.train_decoder_end_to_end and not self.initialize_decoder_from_pretrained:
            raise ValueError("When train_decoder_end_to_end is False (GPS mode), decoder must be initialized from pretrained checkpoint")


def make_env(
    env_id: str,
    seed: int,
    idx: int,
    capture_video: bool,
    run_name: str,
    max_episode_steps: int,
    reward_strategy: RewardStrategy,
    observation_encoding_strategy: ObservationEncodingStrategy,
    partial_observability_strategy: PartialObservabilityStrategy,
    view_radius: int,
    is_slippery: bool,
    slippery_prob: float,
    sticky_action_prob: float,
    random_action_prob: float,
    obstacle_map: str,
    max_level: int,
) -> Callable[..., gym.Env]:
    def thunk():
        env = gym.make(
            env_id,
            obstacle_map=obstacle_map,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps,
            max_steps=max_episode_steps,
            reward_strategy=reward_strategy,
            observation_encoding_strategy=observation_encoding_strategy,
            partial_observability_strategy=partial_observability_strategy,
            view_radius=view_radius,
            is_slippery=is_slippery,
            slippery_prob=slippery_prob,
            sticky_action_prob=sticky_action_prob,
            random_action_prob=random_action_prob,
            max_level=max_level,
        )

        if capture_video and idx == 0:
            env = gym.wrappers.RecordVideo(
                env, f"videos/{run_name}", episode_trigger=lambda x: x % 20 == 0
            )

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.TransformObservation(
            env, lambda obs: np.transpose(obs, (2, 0, 1))
        )

        env.action_space.seed(seed)
        return env

    return thunk


def env_setup(
    cfg: Config, run_name: str, capture_video: bool = False
) -> gym.vector.SyncVectorEnv:
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                cfg.env_id,
                cfg.seed + i,
                i,
                capture_video,
                run_name,
                cfg.max_episode_steps,
                cfg.reward_strategy,
                cfg.observation_encoding_strategy,
                cfg.partial_observability_strategy,
                cfg.view_radius,
                cfg.is_slippery,
                cfg.slippery_prob,
                cfg.sticky_action_prob,
                cfg.random_action_prob,
                cfg.obstacle_map,
                cfg.max_level,
            )
            for i in range(cfg.num_envs)
        ]
    )
    envs.single_observation_space = VecTransposeImage.transpose_space(
        envs.envs[0].observation_space
    )
    envs.observations = create_empty_array(
        envs.single_observation_space, n=envs.num_envs, fn=np.zeros
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    return envs


def dataset_setup(
    cfg: Config, run_name: str, max_level: int = -1, start_level: int = 1
) -> tuple[
    list[tuple[PositionXY, PositionXY, int]],
    list[tuple[PositionXY, PositionXY, int]],
    list[tuple[PositionXY, PositionXY, int]],
    list[tuple[PositionXY, PositionXY, int]],
]:
    envs = env_setup(cfg, run_name)
    env = envs.envs[0]

    train_dataset, val_dataset, test_dataset = simplegrid_dataset_setup(
        env.unwrapped, max_level, start_level
    )

    if cfg.train_dataset_size is not None:
        train_dataset = train_dataset[: cfg.train_dataset_size]

    if cfg.val_dataset_size is not None:
        val_dataset = val_dataset[: cfg.val_dataset_size]

    if cfg.test_dataset_size is not None:
        test_dataset = test_dataset[: cfg.test_dataset_size]

    num_elements_to_choose = cfg.train_eval_dataset_size if cfg.train_eval_dataset_size is not None else max(1, int(len(train_dataset) * 0.1))
    sub_training_envs = random.sample(train_dataset, num_elements_to_choose)

    train_dataset_array = np.array([e[-1] for e in train_dataset])
    train_dataset_levels_average = np.mean(train_dataset_array)
    val_dataset_array = np.array([e[-1] for e in val_dataset])
    val_dataset_levels_average = np.mean(val_dataset_array)
    test_dataset_array = np.array([e[-1] for e in test_dataset])
    test_dataset_levels_average = np.mean(test_dataset_array)
    sub_training_envs_array = np.array([e[-1] for e in sub_training_envs])
    sub_training_envs_levels_average = np.mean(sub_training_envs_array)
    print(
        f"train dataset size={len(train_dataset)}, levels average={train_dataset_levels_average}"
    )
    print(
        f"val dataset size={len(val_dataset)}, levels average={val_dataset_levels_average}"
    )
    print(
        f"test dataset size={len(test_dataset)}, levels average={test_dataset_levels_average}"
    )
    print(
        f"train_eval dataset size={len(sub_training_envs)}, levels average={sub_training_envs_levels_average}"
    )

    return train_dataset, val_dataset, test_dataset, sub_training_envs


def compare_dicts(dict1, dict2):
    cond = True
    # Check if the keys are the same
    if dict1.keys() != dict2.keys():
        return False

    # Check if the values (tensors) are the same for each key
    for key in dict1:
        tensor1 = dict1[key]
        tensor2 = dict2[key]
        if not torch.equal(tensor1, tensor2):
            cond = False

    return cond


@torch.no_grad()
def eval_model(
    cfg: Config,
    dataset: list[tuple[PositionXY, PositionXY, int]],
    run_name: str,
    actor: torch.nn.Module,
    decoder: ActionGen,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    critic: Optional[torch.nn.Module] = None,
    verbose_stats: bool = False,
    deterministic_eval: Optional[bool] = False
) -> Union[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[list[float]], list[float]], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], list[list[float]], list[float]]]:
    assert critic is not None

    envs = env_setup(cfg, run_name, capture_video=capture_video)
    envs.envs[0].unwrapped.start_goal_dataset = dataset

    actor.eval()
    if critic is not None:
        critic.eval()

    # The decoder must be evaluated deterministically too. The old MLP decoder had no
    # dropout layers, so its train/eval mode was a no-op and nothing here ever set it;
    # the Transformer decoder does have dropout, and leaving it in train mode injects
    # noise into every generated sequence at evaluation time.
    decoder_was_training = decoder.model.training
    decoder.model.eval()

    eval_episodes = len(dataset)

    obs, obs_info = envs.reset()
    episodic_returns: list[float] = []
    episodic_successes = []
    episodic_num_decoder_generations = []
    num_decoder_generations = 0
    valid_actions_in_sequence_per_episode_list, unused_actions_in_sequence_per_episode_list = [], []
    avg_valid_actions_in_sequences_list, avg_unused_actions_in_sequences_list = [], []
    avg_agent_step_ratio_in_sequences_list = []
    current_episode_inference_times = []  # Track inference times for current episode
    successful_episode_inference_times = []  # Track inference times per successful episode
    successful_episode_wall_clock_times_per_optimal_step = []  # Track wall-clock time per optimal step
    
    # Track episode timing
    episode_start_time = time.perf_counter()

    avg_global_visited_position_count_in_sequences_list = []
    avg_prev_visited_position_count_in_sequences_list = []
    avg_current_visited_position_count_in_sequences_list = []

    if verbose_stats:
        all_env_eval_stats: list[dict] = []
        current_env_eval_stats = {
            'start_position': envs.envs[0].unwrapped.start_xy,
            'goal_position': envs.envs[0].unwrapped.goal_xy,
            'level': envs.envs[0].unwrapped.level,
            'actions': [],
            'forward_actions': []
        }

        global_visited_position_count_total = 0
        prev_visited_position_count_total = 0
        current_visited_position_count_total = 0

        visited_positions_stats = {"global_visited_positions": set(), "prev_visited_positions": set()}

    while len(episodic_returns) < eval_episodes:
        optimal_episode_length = envs.envs[0].unwrapped.level

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

        # Measure inference time
        start_time = time.perf_counter()
        actor_proto_plan_emb = actor(obs_tensor)

        action_list_batch, action_list_proba_batch = decoder.gen_action_seq(gen_input=actor_proto_plan_emb, get_actions_as_one_hot=(cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT), deterministic_mode=deterministic_eval or False)

        action_input_for_critic = get_action_format(action_format=cfg.action_seq_representation,
                                                    action_probas=action_list_proba_batch,
                                                    act_one_hot=action_list_batch,
                                                    cfg=cfg)

        batch_q_stack = torch.tensor([[-1]], device=device)
        if action_input_for_critic.shape[0] > 1:
            batch_q_stack = critic(obs_tensor.repeat(cfg.n_proto_plan_candidates, 1, 1, 1).to(
                device), actor_proto_plan_emb, action_input_for_critic)

        if batch_q_stack.shape == (1, 1):
            # Directly use index 0 if both dimensions are 1 or if none
            max_q_val, max_q_idx = batch_q_stack[0], torch.tensor([0], device=device)
        else:
            max_q_val, max_q_idx = batch_q_stack.squeeze().topk(1)
        action_list = action_list_batch if cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT else action_list_proba_batch
        action_sequence = action_list[max_q_idx][0].argmax(-1)

        actions, _ = DenseVAE.trim_action_sequence_from_eos_tokens(action_sequence)
        end_time = time.perf_counter()
        current_episode_inference_times.append(end_time - start_time)

        next_obs, _, termination, truncation, infos, cur_n_steps, forward_actions, _, obs_info, _, _, _ = decoder.get_reward_per_sequence(
            current_env=envs, act_list=actions, old_state=obs_info['agent_xy'][0], visited_positions_stats=visited_positions_stats if verbose_stats else None)
        num_decoder_generations += 1

        if verbose_stats:
            current_env_eval_stats['actions'].append(actions.tolist())
            current_env_eval_stats['forward_actions'].append(forward_actions)

            # Revisit tracking logic
            visited_positions = visited_positions_stats["visited_positions"]
            visited_positions_stats["global_visited_positions"] = visited_positions_stats["global_visited_positions"] | visited_positions
            visited_positions_stats["prev_visited_positions"] = visited_positions
            global_visited_position_count_total += visited_positions_stats["global_sequence_revisits_count"]
            prev_visited_position_count_total += visited_positions_stats["prev_sequence_revisits_count"]
            current_visited_position_count_total += visited_positions_stats["current_sequence_revisits_count"]

        valid_actions_in_sequence_per_episode_list.append(len(actions))
        unused_actions_in_sequence_per_episode_list.append(len(actions) - cur_n_steps)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue

                episode_length = info['episode']['l'][0]
                avg_agent_step_ratio_per_episode = np.round(optimal_episode_length / episode_length, 2)
                is_successful = termination[0]
                
                # Calculate episode wall-clock time
                episode_end_time = time.perf_counter()
                episode_wall_clock_time = episode_end_time - episode_start_time
                
                msg = f"eval_episode={len(episodic_returns)}, episodic_length={info['episode']['l']}, episodic_return={info['episode']['r']}"

                if not truncation[0]:  # exclude environments where the agent couldn't reach the goal
                    avg_valid_actions_in_sequences_per_episode = np.round(
                        np.mean(valid_actions_in_sequence_per_episode_list), 2)
                    avg_unused_actions_in_sequences_per_episode = np.round(
                        np.mean(unused_actions_in_sequence_per_episode_list), 2)
                    msg += f", num_decoder_generations={num_decoder_generations}"
                    msg += f", avg_valid_actions_per_sequence_in_episode={avg_valid_actions_in_sequences_per_episode}"
                    msg += f", avg_unused_actions_per_sequence_in_episode={avg_unused_actions_in_sequences_per_episode}"
                    msg += f", avg_agent_step_ratio_in_episode={avg_agent_step_ratio_per_episode}"
                    if verbose_stats:
                        msg += f", global_visited_position_count_total={global_visited_position_count_total}"
                        msg += f", prev_visited_position_count_total={prev_visited_position_count_total}"
                        msg += f", current_visited_position_count_total={current_visited_position_count_total}"
                msg += f", successful={is_successful}"
                print(msg)

                episodic_returns += [info["episode"]["r"]]
                episodic_successes += [1 if is_successful else 0]

                # Only track inference times and wall-clock times for successful episodes
                if is_successful and len(current_episode_inference_times) > 0:
                    successful_episode_inference_times.append(current_episode_inference_times.copy())
                    # Calculate wall-clock time per optimal step
                    wall_clock_time_per_optimal_step = episode_wall_clock_time / optimal_episode_length
                    successful_episode_wall_clock_times_per_optimal_step.append(wall_clock_time_per_optimal_step)
                if not truncation[0]:
                    episodic_num_decoder_generations += [num_decoder_generations]
                    avg_valid_actions_in_sequences_list.append(avg_valid_actions_in_sequences_per_episode)
                    avg_unused_actions_in_sequences_list.append(avg_unused_actions_in_sequences_per_episode)
                    avg_agent_step_ratio_in_sequences_list.append(avg_agent_step_ratio_per_episode)

                    if verbose_stats:
                        avg_global_visited_position_count_in_sequences_list.append(global_visited_position_count_total)
                        avg_prev_visited_position_count_in_sequences_list.append(prev_visited_position_count_total)
                        avg_current_visited_position_count_in_sequences_list.append(current_visited_position_count_total)

                        num_actions_in_episode = sum([len(a) for a in current_env_eval_stats['actions']])
                        current_env_eval_stats.update(
                            {
                                'episode_return': info["episode"]["r"][0],
                                'episode_success': 1 if termination[0] else 0,
                                'num_decoder_generations': num_decoder_generations,
                                'num_actions_in_episode': num_actions_in_episode,
                                'forward_action_list_length': len(current_env_eval_stats['forward_actions']),
                                'avg_agent_step_ratio_per_episode': avg_agent_step_ratio_per_episode,
                                'global_visited_position_count_total': global_visited_position_count_total,
                                'prev_visited_position_count_total': prev_visited_position_count_total,
                                'current_visited_position_count_total': current_visited_position_count_total,
                            }
                        )
                        all_env_eval_stats.append(current_env_eval_stats)

                valid_actions_in_sequence_per_episode_list, unused_actions_in_sequence_per_episode_list = [], []
                num_decoder_generations = 0
                
                # Reset for next episode
                current_episode_inference_times = []
                episode_start_time = time.perf_counter()  # Start timing next episode

                if verbose_stats:
                    current_env_eval_stats = {
                        'start_position': envs.envs[0].unwrapped.start_xy,
                        'goal_position': envs.envs[0].unwrapped.goal_xy,
                        'level': envs.envs[0].unwrapped.level,
                        'actions': [],
                        'forward_actions': []
                    }

                    global_visited_position_count_total = 0
                    prev_visited_position_count_total = 0
                    current_visited_position_count_total = 0

                    visited_positions_stats = {"global_visited_positions": set(), "prev_visited_positions": set()}

        obs = next_obs

    envs.close()

    if decoder_was_training:
        decoder.model.train()

    if verbose_stats:
        return np.asarray(episodic_returns), np.asarray(episodic_successes), np.asarray(episodic_num_decoder_generations), \
               np.asarray(avg_valid_actions_in_sequences_list), np.asarray(avg_unused_actions_in_sequences_list), np.asarray(avg_agent_step_ratio_in_sequences_list), \
               all_env_eval_stats, successful_episode_inference_times, successful_episode_wall_clock_times_per_optimal_step

    return np.asarray(episodic_returns), np.asarray(episodic_successes), np.asarray(episodic_num_decoder_generations), \
           np.asarray(avg_valid_actions_in_sequences_list), np.asarray(avg_unused_actions_in_sequences_list), np.asarray(avg_agent_step_ratio_in_sequences_list), \
           successful_episode_inference_times, successful_episode_wall_clock_times_per_optimal_step


@torch.no_grad()
def eval_with_pretrained_model(
    cfg: Config,
    dataset: list[tuple[PositionXY, PositionXY, int]],
    run_name: str,
    actor: Type[Actor],
    actor_model_path: str,
    critic: Optional[Type[Critic]],
    critic_model_path: Optional[str],
    n_input_channels_actor: int,
    n_input_channels_critic: int,
    n_output_channels: int,
    observation_shape: tuple[int, int],
    decoder: ActionGen,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    deterministic_evaluation: Optional[bool] = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict], dict]:
    message = f"Evaluation with pretrained actor model {actor_model_path}"
    if critic_model_path is not None:
        message += f", with pretrained critic {critic_model_path}"
    border = "*" * 20
    print(border)
    print("* " + message + " *")

    proto_plan_noise = None
    actor = actor(n_input_channels=n_input_channels_actor,
                  n_output_channels=n_output_channels,
                  observation_shape=observation_shape,
                  noise=proto_plan_noise,
                  linear_layers=cfg.actor_linear_layers,
                  linear_layers_activation=cfg.actor_linear_layers_activation_function,
                  use_batchnorm_linear_layers=cfg.actor_use_batchnorm_linear_layers,
                  num_heads=cfg.n_proto_plan_candidates,
                  pe_embedding_dim=cfg.pe_embedding_dim).to(device)
    actor.load_state_dict(torch.load(actor_model_path, map_location=device))

    if critic_model_path is not None:
        action_seq_dim = (cfg.n_actions_in_seq if cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_INT else cfg.n_actions_in_seq * (cfg.action_space_size + 1)) + (cfg.actor_n_output_channels if cfg.include_actor_embedding_in_critic_input else 0)
        critic = critic(n_input_channels=n_input_channels_critic,
                        observation_shape=observation_shape,
                        action_seq_dim=action_seq_dim,
                        linear_layers=cfg.critic_linear_layers,
                        linear_layers_activation=cfg.critic_linear_layers_activation_function,
                        use_batchnorm_linear_layers=cfg.critic_use_batchnorm_linear_layers,
                        include_actor_embedding_in_critic_input=cfg.include_actor_embedding_in_critic_input).to(device)
        critic.load_state_dict(torch.load(critic_model_path, map_location=device))
    else:
        critic = None

    # The actor and critic above are loaded from the best-on-validation checkpoint, but
    # ``decoder`` arrives as the *live* object. With a frozen pretrained decoder those are
    # the same weights, so this never mattered -- under --train_decoder_end_to_end they are
    # not: the decoder kept training after that checkpoint was written, so the evaluation
    # would pair a best-on-validation actor with whatever state the decoder happened to end
    # in. Load the decoder that was saved alongside them.
    saved_decoder_state = None
    if cfg.train_decoder_end_to_end and critic_model_path is not None:
        decoder_model_path = critic_model_path.replace('/critic', '/decoder')
        if os.path.isfile(decoder_model_path):
            saved_decoder_state = copy.deepcopy(decoder.model.state_dict())
            decoder.model.load_state_dict(torch.load(decoder_model_path, map_location=device))
            print(f"loaded matching decoder checkpoint from {decoder_model_path}")
        else:
            print(f"WARNING: {decoder_model_path} not found; evaluating with the live "
                  f"decoder, which does not match the actor/critic checkpoint.")

    decoder_was_training = decoder.model.training
    decoder.model.eval()

    try:
        episodic_returns, episodic_successes, episodic_num_decoder_generations, episodic_avg_valid_actions_in_sequences, \
            episodic_avg_unused_actions_in_sequences, episodic_avg_agent_step_ratio_in_sequences, all_env_eval_stats, successful_episode_inference_times, successful_episode_wall_clock_times_per_optimal_step = eval_model(cfg, dataset, run_name, actor, decoder, device, capture_video, critic=critic, verbose_stats=True, deterministic_eval=deterministic_evaluation)
    finally:
        # Put the caller's decoder back exactly as it was: this function is also called
        # mid-training (eval_test_dataset_during_training_freq), where swapping in a
        # checkpoint and leaving it there would corrupt the run that follows.
        if saved_decoder_state is not None:
            decoder.model.load_state_dict(saved_decoder_state)
        if decoder_was_training:
            decoder.model.train()

    # Calculate inference timing statistics for successful episodes only
    total_episodes = len(episodic_returns)
    successful_episodes = len(successful_episode_inference_times)
    
    if successful_episodes > 0:
        # Calculate wall-clock time per optimal step (fair comparison metric)
        avg_wall_clock_time_per_optimal_step = np.mean(successful_episode_wall_clock_times_per_optimal_step)
        std_wall_clock_time_per_optimal_step = np.std(successful_episode_wall_clock_times_per_optimal_step)
        
        # Also calculate overall stats for all decisions in successful episodes
        all_successful_decisions = [time for episode_times in successful_episode_inference_times for time in episode_times]
        avg_inference_time_per_decision_successful = np.mean(all_successful_decisions)
        std_inference_time_per_decision_successful = np.std(all_successful_decisions)
        total_decisions_successful = len(all_successful_decisions)
    else:
        avg_wall_clock_time_per_optimal_step = 0.0
        std_wall_clock_time_per_optimal_step = 0.0
        avg_inference_time_per_decision_successful = 0.0
        std_inference_time_per_decision_successful = 0.0
        total_decisions_successful = 0

    with_critic_message = f"{' with critic' if critic_model_path else ''}"
    deterministic_message = f"{'-deterministic inference' if deterministic_evaluation else ''}"
    print(f"test dataset{with_critic_message}{deterministic_message} mean reward: {np.round(np.mean(episodic_returns), 3)}")
    print(f"test dataset{with_critic_message}{deterministic_message} mean success: {np.round(np.mean(episodic_successes), 3)}")
    print(f"test dataset{with_critic_message}{deterministic_message} mean num decoder generations: {np.round(np.mean(episodic_num_decoder_generations), 3)}")
    print(f"test dataset{with_critic_message}{deterministic_message} mean num valid actions in sequences per episode: {np.round(np.mean(episodic_avg_valid_actions_in_sequences), 3)}")
    print(
        f"test dataset{with_critic_message}{deterministic_message} mean num unused actions in sequences per episode: {np.round(np.mean(episodic_avg_unused_actions_in_sequences), 3)}")
    print(f"test dataset{with_critic_message}{deterministic_message} mean agent step ratio in sequences per episode: {np.round(np.mean(episodic_avg_agent_step_ratio_in_sequences), 3)}")
    print(f"successful episodes: {successful_episodes}/{total_episodes}")
    print(f"average wall-clock time per optimal step: {avg_wall_clock_time_per_optimal_step:.6f} seconds")
    print(f"std wall-clock time per optimal step: {std_wall_clock_time_per_optimal_step:.6f} seconds")
    print(f"average inference time per decision (successful episodes only): {avg_inference_time_per_decision_successful:.6f} seconds")
    print(f"std inference time per decision (successful episodes only): {std_inference_time_per_decision_successful:.6f} seconds")
    print(f"total decisions in successful episodes: {total_decisions_successful}")

    # Create timing statistics dictionary like in DQN version
    timing_stats = {
        'avg_wall_clock_time_per_optimal_step': avg_wall_clock_time_per_optimal_step,
        'std_wall_clock_time_per_optimal_step': std_wall_clock_time_per_optimal_step,
        'avg_inference_time_per_decision_successful': avg_inference_time_per_decision_successful,
        'std_inference_time_per_decision_successful': std_inference_time_per_decision_successful,
        'total_episodes': total_episodes,
        'successful_episodes': successful_episodes,
        'total_decisions_successful': total_decisions_successful
    }

    if len(all_env_eval_stats) > 0:
        print(f"test dataset{with_critic_message}{deterministic_message} mean num actions per episode: {np.round(sum([stats['num_actions_in_episode'] for stats in all_env_eval_stats]) / len(all_env_eval_stats), 3)}")
        print(f"test dataset{with_critic_message}{deterministic_message} mean global visited position count in sequences per episode: {np.round(sum([stats['global_visited_position_count_total'] for stats in all_env_eval_stats]) / len(all_env_eval_stats), 3)}")
        print(f"test dataset{with_critic_message}{deterministic_message} mean prev visited position count in sequences per episode: {np.round(sum([stats['prev_visited_position_count_total'] for stats in all_env_eval_stats]) / len(all_env_eval_stats), 3)}")
        print(f"test dataset{with_critic_message}{deterministic_message} mean current visited position count in sequences per episode: {np.round(sum([stats['current_visited_position_count_total'] for stats in all_env_eval_stats]) / len(all_env_eval_stats), 3)}")
    else:
        print(f"test dataset{with_critic_message}{deterministic_message} mean num actions per episode: N/A (no data)")
        print(f"test dataset{with_critic_message}{deterministic_message} mean global visited position count in sequences per episode: N/A (no data)")
        print(f"test dataset{with_critic_message}{deterministic_message} mean prev visited position count in sequences per episode: N/A (no data)")
        print(f"test dataset{with_critic_message}{deterministic_message} mean current visited position count in sequences per episode: N/A (no data)")

    print(border)

    return episodic_returns, episodic_successes, episodic_num_decoder_generations, episodic_avg_valid_actions_in_sequences, \
           episodic_avg_unused_actions_in_sequences, episodic_avg_agent_step_ratio_in_sequences, all_env_eval_stats, timing_stats


@torch.no_grad()
def eval_during_training(
    cfg: Config,
    dataset: list[tuple[PositionXY, PositionXY, int]],
    run_name: str,
    actor: torch.nn.Module,
    decoder: ActionGen,
    device: torch.device = torch.device("cpu"),
    capture_video: bool = False,
    critic: Optional[torch.nn.Module] = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    actor_in_training = actor.training
    critic_in_training = critic.training if critic is not None else False

    episodic_returns, episodic_successes, episodic_num_decoder_generations, episodic_avg_valid_actions_in_sequence, \
    episodic_avg_unused_actions_in_sequence, episodic_avg_agent_step_ratio_in_sequence, _, _ = eval_model(cfg, dataset, run_name, actor, decoder, device, capture_video, critic=critic)

    if actor_in_training:
        actor.train()
    if critic is not None and critic_in_training:
        critic.train()

    return episodic_returns, episodic_successes, episodic_num_decoder_generations, episodic_avg_valid_actions_in_sequence, \
           episodic_avg_unused_actions_in_sequence, episodic_avg_agent_step_ratio_in_sequence


def perform_evaluation(
        cfg: Config,
        dataset: list[tuple[PositionXY, PositionXY, int]],
        run_name: str,
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        decoder: ActionGen,
        device: torch.device,
        global_step: int,
        writer,
        eval_on_train_dataset: bool,
        with_critic: bool,
        eval_freq: int,
        actor_path: str,
        critic_path: Optional[str] = None,
        best_mean_reward: float = float('-inf'),
        best_mean_success: float = float('-inf'),
        best_mean_num_decoder_generations: float = float('-inf'),
        best_mean_valid_actions_in_sequences: float = float('-inf'),
        best_mean_unused_actions_in_sequences: float = float('-inf'),
        best_mean_value_global_step: float = float('-inf'),
        best_mean_gent_step_ratio_in_sequences: float = float('-inf'),
        save_prefix: str = "train-eval",
) -> tuple[float, float, float, float, float, float, float, bool]:
    found_new_best_model = False

    if eval_freq > 0 and global_step % eval_freq == 0 or global_step == cfg.total_timesteps - 1:
        with torch.no_grad():
            with_critic_message = f"{' with critic' if with_critic else ''}"
            train_or_eval_message = f"{'train' if eval_on_train_dataset else 'val'}"
            save_prefix = f"{save_prefix if eval_on_train_dataset else 'val-eval'}{'-with-critic' if with_critic else ''}"

            message = f"Evaluation on {train_or_eval_message} dataset{with_critic_message}"
            border = "*" * 20
            print(border)
            print("* " + message + " *")

            episode_rewards, episodic_successes, episode_num_decoder_generations, episode_avg_valid_actions_in_sequences, \
            episode_avg_unused_actions_in_sequences, episodic_avg_agent_step_ratio_in_sequences = eval_during_training(
                cfg,
                dataset=dataset,
                run_name=run_name,
                actor=actor,
                decoder=decoder,
                device=device,
                critic=critic if with_critic else None
            )

            mean_reward = np.round(np.mean(episode_rewards), 3)
            mean_success = np.round(np.mean(episodic_successes), 3)
            mean_num_decoder_generations = np.round(np.mean(episode_num_decoder_generations), 3)
            mean_valid_actions_in_sequences_per_episode = np.round(np.mean(episode_avg_valid_actions_in_sequences), 3)
            mean_unused_actions_in_sequences_per_episode = np.round(np.mean(episode_avg_unused_actions_in_sequences), 3)
            mean_agent_step_ratio_in_sequences_per_episode = np.round(np.mean(episodic_avg_agent_step_ratio_in_sequences), 3)
            print_msg_prefix = f"{train_or_eval_message} dataset{with_critic_message}"
            print(f"{print_msg_prefix} mean reward: {mean_reward}")
            print(f"{print_msg_prefix} mean success: {mean_success}")
            print(f"{print_msg_prefix} mean num decoder generations: {mean_num_decoder_generations}")
            print(f"{print_msg_prefix} mean num valid actions in sequences per episode: {mean_valid_actions_in_sequences_per_episode}")
            print(f"{print_msg_prefix} mean num unused actions in sequences per episode: {mean_unused_actions_in_sequences_per_episode}")
            print(f"{print_msg_prefix} mean agent step ratio in sequences per episode: {mean_agent_step_ratio_in_sequences_per_episode}")

            if cfg.save_model_strategy == SaveModelStrategy.SUCCESS_RATE:
                if mean_success > best_mean_success or (
                        mean_success == best_mean_success and mean_reward > best_mean_reward):
                    best_mean_reward = mean_reward
                    best_mean_success = mean_success
                    best_mean_num_decoder_generations = mean_num_decoder_generations
                    best_mean_valid_actions_in_sequences = mean_valid_actions_in_sequences_per_episode
                    best_mean_unused_actions_in_sequences = mean_unused_actions_in_sequences_per_episode
                    best_mean_gent_step_ratio_in_sequences = mean_agent_step_ratio_in_sequences_per_episode
                    best_mean_value_global_step = global_step
                    found_new_best_model = True
            else:
                if mean_reward > best_mean_reward or (
                        mean_reward == best_mean_reward and mean_success > best_mean_success):
                    best_mean_reward = mean_reward
                    best_mean_success = mean_success
                    best_mean_num_decoder_generations = mean_num_decoder_generations
                    best_mean_valid_actions_in_sequences = mean_valid_actions_in_sequences_per_episode
                    best_mean_unused_actions_in_sequences = mean_unused_actions_in_sequences_per_episode
                    best_mean_gent_step_ratio_in_sequences = mean_agent_step_ratio_in_sequences_per_episode
                    best_mean_value_global_step = global_step
                    found_new_best_model = True

            if found_new_best_model:
                msg = f"New best {train_or_eval_message} model{with_critic_message} mean values, episodic-return: {best_mean_reward}"
                msg += f", success-rate: {best_mean_success}, num-decoder-generations: {best_mean_num_decoder_generations}"
                msg += f", avg_num_valid_actions_per_sequence_in_episode={best_mean_valid_actions_in_sequences}"
                msg += f", avg_num_unused_actions_per_sequence_in_episode={best_mean_unused_actions_in_sequences}"
                msg += f", avg_agent_step_ratio_per_sequence_in_episode={best_mean_gent_step_ratio_in_sequences}"
                print(msg)
                writer.add_scalar(
                    f"{save_prefix}/best_episodic_reward",
                    best_mean_reward,
                    global_step,
                )
                writer.add_scalar(
                    f"{save_prefix}/best_episodic_success",
                    best_mean_success,
                    global_step,
                )
                writer.add_scalar(
                    f"{save_prefix}/best_episodic_num_decoder_generations",
                    best_mean_num_decoder_generations,
                    global_step,
                )
                writer.add_scalar(
                    f"{save_prefix}/best_episodic_avg_num_valid_actions_per_sequence",
                    best_mean_valid_actions_in_sequences,
                    global_step,
                )
                writer.add_scalar(
                    f"{save_prefix}/best_episodic_avg_num_unused_actions_per_sequence",
                    best_mean_unused_actions_in_sequences,
                    global_step,
                )
                writer.add_scalar(
                    f"{save_prefix}/best_mean_value_global_step",
                    best_mean_value_global_step,
                    global_step,
                )
                writer.add_scalar(
                    f"{save_prefix}/best_mean_avg_agent_step_ratio_per_sequence",
                    best_mean_gent_step_ratio_in_sequences,
                    global_step
                )

            if not eval_on_train_dataset and cfg.save_model and found_new_best_model:
                torch.save(actor.state_dict(), actor_path)
                torch.save(critic.state_dict(), critic_path)
                save_msg = f"{train_or_eval_message}{with_critic_message}: actor model saved to {actor_path} and critic model saved to {critic_path}"
                
                # Save decoder if training end-to-end
                if cfg.train_decoder_end_to_end and critic_path is not None:
                    decoder_path = critic_path.replace('/critic', '/decoder')
                    torch.save(decoder.model.state_dict(), decoder_path)
                    save_msg += f" and decoder model saved to {decoder_path}"
                
                print(save_msg)

            print(border)
            writer.add_scalar(
                f"{save_prefix}/episodic_return", mean_reward, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_success_rate", mean_success, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_num_decoder_generations", mean_num_decoder_generations, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_avg_num_valid_actions_per_sequence", mean_valid_actions_in_sequences_per_episode, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_avg_num_unused_actions_per_sequence", mean_unused_actions_in_sequences_per_episode, global_step
            )
            writer.add_scalar(
                f"{save_prefix}/episodic_avg_agent_step_raio_per_sequence", mean_agent_step_ratio_in_sequences_per_episode, global_step
            )

    return best_mean_reward, best_mean_success, best_mean_num_decoder_generations, best_mean_value_global_step, \
           best_mean_valid_actions_in_sequences, best_mean_unused_actions_in_sequences, best_mean_gent_step_ratio_in_sequences, found_new_best_model


def eval_test_dataset(test_dataset: list[tuple[PositionXY, PositionXY, int]],
                      cfg: Config,
                      writer: SummaryWriter,
                      n_input_channels_actor: int,
                      n_input_channels_critic: int,
                      observation_shape: tuple[int, int],
                      decoder: ActionGen,
                      device: torch.device,
                      actor_model_path: str,
                      critic_model_path: Optional[str] = None,
                      global_step: Optional[int] = None):

    message = "Evaluation on test dataset"
    if critic_model_path is not None:
        message += " with critic"
    border = "*" * 20
    print(border)
    print("* " + message + " *")

    tag_name = "test-eval"
    if critic_model_path is not None:
        tag_name += "-with-critic"

    if global_step is not None:
        tag_name += f"-{global_step}"

    def actual_eval_with_pretrained_model(c_cfg: Config,
                                          c_dataset: list[tuple[PositionXY, PositionXY, int]],
                                          c_writer: SummaryWriter,
                                          c_n_input_channels_actor: int,
                                          c_n_input_channels_critic: int,
                                          c_observation_shape: tuple[int, int],
                                          c_decoder: ActionGen,
                                          c_device: torch.device,
                                          c_actor_model_path: str,
                                          c_run_name: str,
                                          c_tag_name: str,
                                          deterministic_evaluation: Optional[bool] = False,
                                          c_critic_model_path: Optional[str] = None,
                                          c_global_step: Optional[int] = None):
        if deterministic_evaluation:
            c_run_name += "-deterministic"
            c_tag_name += "-deterministic"
        episodic_returns, episodic_successes, episodic_num_decoder_generations, episodic_avg_valid_actions_in_sequences, \
        episodic_avg_unused_actions_in_sequences, episodic_avg_agent_step_ratio_in_sequences, all_env_eval_stats, timing_stats = eval_with_pretrained_model(
            cfg=c_cfg,
            dataset=c_dataset,
            run_name=c_run_name,
            actor=Actor,
            actor_model_path=c_actor_model_path,
            critic=Critic,
            critic_model_path=c_critic_model_path,
            n_input_channels_actor=c_n_input_channels_actor,
            n_input_channels_critic=c_n_input_channels_critic,
            n_output_channels=c_cfg.actor_n_output_channels,
            observation_shape=c_observation_shape,
            decoder=c_decoder,
            device=c_device,
            capture_video=True if c_global_step is None else False,
            deterministic_evaluation=deterministic_evaluation
        )

        if global_step is None:
            for idx, episodic_return in enumerate(episodic_returns):
                c_writer.add_scalar(f"{c_tag_name}/episodic_return", episodic_return, idx)
            for idx, agent_step_ratio in enumerate(episodic_avg_agent_step_ratio_in_sequences):
                c_writer.add_scalar(f"{c_tag_name}/episodic_agent_step_ratio", agent_step_ratio, idx)

        test_metrics = {
            f"{c_tag_name}/mean_episodic_return": np.round(np.mean(episodic_returns), 3),
            f"{c_tag_name}/mean_episodic_success": np.round(np.mean(episodic_successes), 3),
            f"{c_tag_name}/mean_num_decoder_generations": np.round(np.mean(episodic_num_decoder_generations), 3),
            f"{c_tag_name}/mean_num_valid_actions_per_sequence_in_episode": np.round(np.mean(episodic_avg_valid_actions_in_sequences), 3),
            f"{c_tag_name}/mean_num_unused_actions_per_sequence_in_episode": np.round(np.mean(episodic_avg_unused_actions_in_sequences), 3),
            f"{c_tag_name}/mean_agent_step_ratio_per_sequence_in_episode": np.round(np.mean(episodic_avg_agent_step_ratio_in_sequences), 3),
            # Timing statistics (successful episodes only)
            f"{c_tag_name}/avg_wall_clock_time_per_optimal_step": timing_stats['avg_wall_clock_time_per_optimal_step'],
            f"{c_tag_name}/std_wall_clock_time_per_optimal_step": timing_stats['std_wall_clock_time_per_optimal_step'],
            f"{c_tag_name}/avg_inference_time_per_decision_successful": timing_stats['avg_inference_time_per_decision_successful'],
            f"{c_tag_name}/std_inference_time_per_decision_successful": timing_stats['std_inference_time_per_decision_successful'],
            f"{c_tag_name}/total_episodes": timing_stats['total_episodes'],
            f"{c_tag_name}/successful_episodes": timing_stats['successful_episodes'],
            f"{c_tag_name}/total_decisions_successful": timing_stats['total_decisions_successful'],
        }

        for metric_name, metric_value in test_metrics.items():
            c_writer.add_scalar(metric_name, metric_value)

        # Mirror the same numbers into wandb's run summary.
        #
        # The add_scalar calls above pass no step, so tensorboard records them at step 0 --
        # but by the time the final test evaluation runs, the run's step counter is at
        # total_timesteps. wandb requires non-decreasing steps when syncing tensorboard, so
        # these land behind the run and never appear as columns: the headline test numbers
        # were only ever readable by grepping the slurm log.
        #
        # Summary values carry no step, so they surface directly in the runs table. Keys
        # are identical to the tensorboard tags, e.g.
        #   test-eval-with-critic/mean_episodic_success
        #   test-eval-with-critic-deterministic/mean_episodic_success
        if c_cfg.track:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.run.summary.update(
                        {k: float(v) for k, v in test_metrics.items()}
                    )
            except Exception as exc:  # pragma: no cover - logging must not break evaluation
                print(f"[GPS] (wandb summary update skipped: {exc})", flush=True)

        if len(all_env_eval_stats) > 0:
            writer.add_scalar(
                f"{c_tag_name}/mean_num_actions_per_episode",
                np.round(sum([stats['num_actions_in_episode'] for stats in all_env_eval_stats]) / len(all_env_eval_stats), 3))
            writer.add_scalar(
                f"{c_tag_name}/mean_global_visited_position_count_in_sequences_per_episode",
                np.round(sum([stats['global_visited_position_count_total'] for stats in all_env_eval_stats]) / len(all_env_eval_stats), 3))
            writer.add_scalar(
                f"{c_tag_name}/mean_prev_visited_position_count_in_sequences_per_episode",
                np.round(sum([stats['prev_visited_position_count_total'] for stats in all_env_eval_stats]) / len(all_env_eval_stats), 3))
            writer.add_scalar(
                f"{c_tag_name}/mean_current_visited_position_count_in_sequences_per_episode",
                np.round(sum([stats['current_visited_position_count_total'] for stats in all_env_eval_stats]) / len(all_env_eval_stats), 3))

    if cfg.deterministic_inference:  # deterministic_evaluation == True
        set_seed(seed=cfg.seed, deterministic_torch=cfg.torch_deterministic)

        actual_eval_with_pretrained_model(
            c_cfg=cfg,
            c_dataset=test_dataset,
            c_writer=writer,
            c_n_input_channels_actor=n_input_channels_actor,
            c_n_input_channels_critic=n_input_channels_critic,
            c_observation_shape=observation_shape,
            c_decoder=decoder,
            c_device=device,
            c_actor_model_path=actor_model_path,
            c_run_name=f"{run_name}-{tag_name}",
            c_tag_name=tag_name,
            deterministic_evaluation=True,
            c_critic_model_path=critic_model_path,
            c_global_step=global_step
        )

    set_seed(seed=cfg.seed, deterministic_torch=cfg.torch_deterministic)
    actual_eval_with_pretrained_model(  # deterministic_evaluation == False (default)
        c_cfg=cfg,
        c_dataset=test_dataset,
        c_writer=writer,
        c_n_input_channels_actor=n_input_channels_actor,
        c_n_input_channels_critic=n_input_channels_critic,
        c_observation_shape=observation_shape,
        c_decoder=decoder,
        c_device=device,
        c_actor_model_path=actor_model_path,
        c_run_name=f"{run_name}-{tag_name}",
        c_tag_name=tag_name,
        deterministic_evaluation=False,
        c_critic_model_path=critic_model_path,
        c_global_step=global_step
    )

    if cfg.track and cfg.upload_model_to_wandb and global_step is None:
        wandb.save(actor_model_path)
        wandb.save(critic_model_path)
        # Upload decoder if training end-to-end
        if cfg.train_decoder_end_to_end and critic_model_path is not None:
            decoder_model_path = critic_model_path.replace('/critic', '/decoder')
            wandb.save(decoder_model_path)

    print(border)


class LinearNoiseDecay:
    def __init__(self, initial_noise, final_noise, total_steps):
        self.initial_noise = initial_noise
        self.final_noise = final_noise
        self.total_steps = total_steps
        self.noise_range = initial_noise - final_noise
        self.current_noise = initial_noise

    def get_noise(self, step) -> float:
        if step >= self.total_steps:
            self.current_noise = self.final_noise
        else:
            self.current_noise = self.final_noise + self.noise_range * (1 - step / self.total_steps)
        return self.current_noise


def get_action_format(action_format: ActionSeqRepresentation,
                      act_one_hot: torch.Tensor,
                      action_probas: torch.Tensor,
                      cfg: Config) -> torch.Tensor:
    if action_format == ActionSeqRepresentation.ACTION_SEQ_AS_INT:
        if cfg.exclude_decoder_from_computation_graph:
            return action_probas.argmax(-1)
        else:
            # Sample indices from the probability distribution
            reshaped_probs = action_probas.view(-1, cfg.action_space_size + 1)

            # Convert probabilities to logits safely
            epsilon = 1e-20  # Small value to prevent log(0)
            reshaped_logits = torch.log(reshaped_probs + epsilon)

            # Apply Gumbel-Softmax with a temperature parameter
            tau = 1.0  # Adjust the temperature as needed
            gumbel_sample = F.gumbel_softmax(reshaped_logits, tau=tau, hard=True)

            # Create an indices vector
            indices_vector = torch.arange(cfg.action_space_size + 1, device=reshaped_logits.device).float()

            # Compute indices by matrix multiplication
            indices = torch.matmul(gumbel_sample, indices_vector)

            # Reshape back to original sequence shape
            indices = indices.view(action_probas.shape[0], cfg.n_actions_in_seq)
            return indices
    elif action_format == ActionSeqRepresentation.ACTION_SEQ_AS_PROB:
        return action_probas.flatten(1)
    elif action_format == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT:
        return act_one_hot.flatten(1)
    else:
        raise ValueError(f"Unknown action format: {action_format}")


def update_target_network(source_network, target_network, tau):
    for source_param, target_param in zip(source_network.parameters(), target_network.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def explore_without_critic(valid_sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    sampled_indices = torch.randint(0, valid_sequences.shape[0], (1,))
    action_list_batch = valid_sequences[sampled_indices]
    return action_list_batch[0], action_list_batch


@torch.no_grad()
def explore_with_critic(valid_sequences: torch.Tensor, obs_tensor: torch.Tensor, actor_proto_plan_emb: torch.Tensor,
                        critic_network: Critic, cfg: Config,
                        device: torch.device, rng_sample_from_candidates: random.Random):
    sampled_indices = torch.randint(0, valid_sequences.shape[0], (cfg.n_proto_plan_candidates,))
    action_list_batch = valid_sequences[sampled_indices]

    batch_q_stack = eval_candidates_with_critic(action_list_batch, action_list_batch, obs_tensor, actor_proto_plan_emb, critic_network, cfg, device)

    return select_action(action_list_batch, action_list_batch, batch_q_stack, cfg, rng_sample_from_candidates)


@torch.no_grad()
def eval_candidates_with_critic(action_list_batch: torch.Tensor, action_list_proba_batch: torch.Tensor,
                                obs_tensor: torch.Tensor, actor_proto_plan_emb: torch.Tensor,
                                critic_network: Critic, cfg: Config) -> torch.Tensor:
    action_input_for_critic = get_action_format(cfg.action_seq_representation, action_list_batch,
                                                action_list_proba_batch, cfg)

    batch_q_stack = torch.tensor([[-1]], device=action_input_for_critic.device)
    if action_input_for_critic.shape[0] > 1:
        if cfg.critic_use_batchnorm_linear_layers:
            repeated_obs = obs_tensor.repeat(2, 1, 1, 1)
            output = critic_network(repeated_obs, actor_proto_plan_emb.repeat(2, 1), action_input_for_critic.repeat(2, 1))
            batch_q_stack = output[0:1, :]
        else:
            repeated_obs = obs_tensor.repeat(cfg.n_proto_plan_candidates, 1, 1, 1)
            batch_q_stack = critic_network(repeated_obs, actor_proto_plan_emb, action_input_for_critic)

    return batch_q_stack


def select_action(action_list_batch: torch.Tensor, action_list_proba_batch: torch.Tensor, batch_q_stack, cfg: Config, rng_sample_from_candidates: random.Random) -> \
tuple[torch.Tensor, torch.Tensor]:
    # A non-finite Q value makes both branches below fail -- torch.multinomial raises
    # "probability tensor contains either `inf`, `nan` or element < 0" outright, which
    # killed three 3-hour runs before they could write their final evaluation. Fall back to
    # the first candidate instead: the run is already past saving at that point, but it
    # reaches the end of train() and evaluates its best saved checkpoint.
    if not torch.isfinite(batch_q_stack).all():
        print(f"WARNING: non-finite Q values in select_action; falling back to candidate 0.",
              flush=True)
        q_idx = torch.tensor([0], device=batch_q_stack.device)
    # Check if batch_q_stack has shape (1, 1)
    elif batch_q_stack.shape == (1, 1):
        # Directly use index 0 if both dimensions are 1
        q_idx = torch.tensor([0], device=batch_q_stack.device)
    else:
        if rng_sample_from_candidates.random() < cfg.sample_from_candidates:
            q_idx = torch.multinomial(F.softmax(batch_q_stack.squeeze(), dim=0), num_samples=1)
        else:
            _, q_idx = batch_q_stack.squeeze().topk(1)

    if cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT:
        return action_list_batch[q_idx][0], action_list_batch
    else:
        return action_list_proba_batch[q_idx][0], action_list_batch


def preprocess_valid_sequences(
        valid_sequences: list[tuple[int, ...]], device: torch.device,
        sequence_length: int, eos_token: int, exploration_sequence_length_constraint: int = -1
) -> torch.Tensor:
    """
    Preprocess valid sequences by converting them to one-hot encoding and optionally filtering them by length.

    Returns:
        torch.Tensor: Preprocessed valid sequences.
    """

    if exploration_sequence_length_constraint == -1:
        valid_sequences = torch.stack([sequence_to_one_hot(seq) for seq in valid_sequences]).to(device)
    else:
        def pad_sequence_with_eos(sequence: torch.Tensor, length: int, eos_token: int) -> torch.Tensor:
            padding_length = max(0, length - sequence.size(0))
            padded_sequence = torch.cat([sequence, torch.full((padding_length,), eos_token, device=device, dtype=torch.int32)])
            return padded_sequence[:length]

        valid_sequences = torch.stack([torch.tensor(seq) for seq in valid_sequences]).to(device)
        valid_sequences = [DenseVAE.trim_action_sequence_from_eos_tokens(seq)[0] for seq in valid_sequences]
        valid_sequences = [seq for seq in valid_sequences if len(seq) >= exploration_sequence_length_constraint]

        # Pad all sequences to the target length with the EOS token
        valid_sequences = [
            pad_sequence_with_eos(seq, sequence_length, eos_token) for seq in valid_sequences
        ]

        valid_sequences = torch.stack([sequence_to_one_hot(seq) for seq in valid_sequences]).to(device)

    return valid_sequences


def compute_sequence_lengths_differentiable(sequences, eos_token=4, epsilon=1e-6):
    # sequences is of shape (batch_size, n_heads, seq_len, vocab_size)
    batch_size, n_heads, seq_len, vocab_size = sequences.shape

    # Create the one-hot vector for the EOS token
    eos_one_hot = F.one_hot(torch.tensor(eos_token), num_classes=vocab_size).to(sequences.dtype).to(sequences.device)
    eos_one_hot = eos_one_hot.view(1, 1, 1, vocab_size).expand(batch_size, n_heads, seq_len, vocab_size)

    # Compute similarity between sequences and EOS token
    similarity = F.cosine_similarity(sequences, eos_one_hot, dim=-1)

    # Ensure numerical stability
    one_minus_similarity = 1 - similarity + epsilon

    # Initialize cumulative product list
    cum_prod_list = []
    previous_cum_prod = one_minus_similarity[..., 0]
    cum_prod_list.append(previous_cum_prod.unsqueeze(-1))  # Add an extra dimension

    # Compute cumulative product without in-place operations
    for i in range(1, seq_len):
        current_cum_prod = previous_cum_prod * one_minus_similarity[..., i]
        cum_prod_list.append(current_cum_prod.unsqueeze(-1))
        previous_cum_prod = current_cum_prod

    # Stack the list into a tensor
    cum_prod = torch.cat(cum_prod_list, dim=-1)  # Shape: (..., seq_len)

    # Compute sequence lengths. Deliberately NOT floored: torch.floor has zero gradient
    # everywhere, so rounding here made this function -- whose whole purpose is to be
    # differentiable -- return a constant with respect to the parameters. Both consumers
    # (invalid_actions_weight and sequence_length_weight) add their penalty to actor_loss,
    # so with the floor in place those terms were logged but contributed no gradient and
    # could not influence training at all. The soft length is what the penalties want.
    lengths = torch.sum(cum_prod, dim=-1)

    return lengths


def compute_diversity_loss_with_normalized_embeddings(actor_embeddings):
    """
    Compute the diversity loss between embeddings from different heads.

    Parameters:
    - actor_embeddings: Tensor of shape (batch_size, n_heads, embedding_dim)

    Returns:
    - diversity_loss: Scalar tensor representing the average cosine similarity
      between embeddings from different heads across the batch.
    """
    batch_size, n_heads, embedding_dim = actor_embeddings.shape

    # Normalize embeddings along the embedding dimension
    normalized_embeddings = F.normalize(actor_embeddings, p=2, dim=2)  # Shape: (batch_size, n_heads, embedding_dim)

    # Compute cosine similarities between all pairs of embeddings
    # Resulting shape: (batch_size, n_heads, n_heads)
    similarities = torch.matmul(normalized_embeddings, normalized_embeddings.transpose(1, 2))

    # Create a mask to select the upper triangular part of the similarity matrix, excluding the diagonal
    triu_indices = torch.triu_indices(n_heads, n_heads, offset=1)
    # Extract the relevant similarities using the indices
    pairwise_similarities = similarities[:, triu_indices[0], triu_indices[1]]  # Shape: (batch_size, num_pairs)

    # Take absolute value of similarities and compute the mean of the pairwise similarities across all pairs and the batch
    # First compute mean absolute similarity for each sample
    per_sample_loss = torch.abs(pairwise_similarities).mean(dim=1)  # Shape: (batch_size,)

    # Then take the mean across the batch
    diversity_loss = per_sample_loss.mean()
    # diversity_loss = torch.abs(pairwise_similarities).mean()

    return diversity_loss


def compute_diversity_loss_vectorized_cosine(actor_embeddings):
    batch_size, n_heads, embedding_dim = actor_embeddings.shape

    # Generate all pairs of indices
    triu_indices = torch.triu_indices(n_heads, n_heads, offset=1)

    # Gather embeddings for all pairs
    embeddings_i = actor_embeddings[:, triu_indices[0], :]  # (batch_size, num_pairs, embedding_dim)
    embeddings_j = actor_embeddings[:, triu_indices[1], :]  # (batch_size, num_pairs, embedding_dim)

    # Compute cosine similarities for all pairs
    similarities = F.cosine_similarity(embeddings_i, embeddings_j, dim=2)  # (batch_size, num_pairs)

    # Compute mean diversity loss
    diversity_loss = torch.abs(similarities).mean()

    return diversity_loss


def update_temperature(adaptive_temperature_decay: LinearNoiseDecay, global_step: int) -> float:
    return adaptive_temperature_decay.get_noise(global_step)


class TrainingLogger:
    def __init__(self, filepath, log_every_k=10000):
        self.filepath = Path(filepath)
        self.log_every_k = log_every_k

    def log_metrics(self, iteration: int, trimmed_action_seq_per_state: list[list]):
        """Write metrics to file with timestamp"""
        with open(self.filepath, 'a') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            str_action_seq_l = [str([a.tolist() for a in trimmed_sequences_per_state]) for trimmed_sequences_per_state in trimmed_action_seq_per_state]
            metrics_str = ('\n').join(str_action_seq_l)
            log_line = f'{timestamp} - Iteration {iteration}:\n{metrics_str}\n'
            f.write(log_line)


def prepare_fixed_states(observation_sample: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Prepare fixed states by updating agent and goal positions in the observation sample.

    Args:
        observation_sample (np.ndarray): Initial observation sample
        device (torch.device): Device to place the tensor

    Returns:
        torch.Tensor: Modified observation with predefined agent and goal positions
    """
    # Predefined start and goal positions
    # Adjusted for LOCAL_VIEW with view_radius=3 (7x7 observation space, coordinates 0-6)
    START_GOAL_POSITIONS = [
        ((0, 0), (4, 1)),  # Agent 0
        ((1, 6), (6, 2)),  # Agent 1
        ((6, 6), (2, 5)),  # Agent 2 - adjusted from (7,7) to (6,6)
        ((6, 3), (0, 3))  # Agent 3
    ]

    # Constants for cell types and colors
    EMPTY_CELL = 1
    AGENT_CELL = 10
    GOAL_CELL = 8
    AGENT_COLOR = 5
    GOAL_COLOR = 2

    # Convert observation to tensor and repeat for each scenario
    fixed_obs = _prepare_base_observation(observation_sample, device, len(START_GOAL_POSITIONS))

    # Find and replace old agent and goal positions
    old_agent_positions = torch.nonzero(fixed_obs[:, 0, :, :] == AGENT_CELL, as_tuple=True)
    old_goal_positions = torch.nonzero(fixed_obs[:, 0, :, :] == GOAL_CELL, as_tuple=True)

    # Update cell types
    fixed_obs[:, 0, :, :][old_agent_positions] = EMPTY_CELL
    fixed_obs[:, 0, :, :][old_goal_positions] = EMPTY_CELL

    # Find and replace old agent and goal colors
    old_agent_color_positions = torch.nonzero(fixed_obs[:, 1, :, :] == AGENT_COLOR, as_tuple=True)
    old_goal_color_positions = torch.nonzero(fixed_obs[:, 1, :, :] == GOAL_COLOR, as_tuple=True)

    # Update cell colors
    fixed_obs[:, 1, :, :][old_agent_color_positions] = 0
    fixed_obs[:, 1, :, :][old_goal_color_positions] = 0

    # Add new agent and goal positions
    for idx, ((start_x, start_y), (goal_x, goal_y)) in enumerate(START_GOAL_POSITIONS):
        # Update cell types
        fixed_obs[idx, 0, start_x, start_y] = AGENT_CELL
        fixed_obs[idx, 0, goal_x, goal_y] = GOAL_CELL

        # Update colors
        fixed_obs[idx, 1, start_x, start_y] = AGENT_COLOR
        fixed_obs[idx, 1, goal_x, goal_y] = GOAL_COLOR

    return fixed_obs


def _prepare_base_observation(
        observation_sample: np.ndarray,
        device: torch.device,
        num_scenarios: int
) -> torch.Tensor:
    """
    Prepare the base observation tensor by converting and repeating the sample.

    Args:
        observation_sample (np.ndarray): Initial observation sample
        device (torch.device): Device to place the tensor
        num_scenarios (int): Number of scenarios to create

    Returns:
        torch.Tensor: Repeated observation tensor
    """
    # Convert to float32 tensor and move to specified device
    observation_tensor = torch.tensor(observation_sample, dtype=torch.float32).to(device)

    # Repeat the observation for multiple scenarios
    return observation_tensor.repeat(num_scenarios, 1, 1, 1)


def clip_weights(model, min_val=-1.0, max_val=1.0):
    for param in model.parameters():
        if param.requires_grad:
            param.data.clamp_(min_val, max_val)


def handle_nan_weights(model):
    had_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param.data).any():
            had_nan = True
            print(f"NaN detected in {name}, resetting...")
            # Reset to small random values
            param.data = torch.randn_like(param.data) * 0.01
    return had_nan


def prepare_padding_tensors(cfg: Config, device: torch.device) -> dict[int, torch.Tensor]:
    """
    Prepare padding tensors for multiple sequence lengths.

    Args:
        cfg: Configuration object
        device: Torch device

    Returns:
        Dictionary of padding tensors for different lengths
    """
    # Create padding tensors for various lengths
    padding_tensors = {}

    for pad_length in range(1, cfg.n_actions_in_seq):
        if cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_INT:
            # Padding for integer action representation
            eos_token = cfg.end_of_sequence_token
            padding_tensors[pad_length] = torch.tensor(
                [eos_token for _ in range(pad_length)],
                device=device
            )
        else:
            # Padding for non-integer action representation
            eos_value = ([0.] * (cfg.action_space_size + 1))
            eos_value[cfg.action_space_size] = 1
            padding_tensors[pad_length] = torch.tensor(
                [eos_value for _ in range(pad_length)],
                device=device
            )

    return padding_tensors


def train(cfg: Config, run_name: str, writer: SummaryWriter, log_dir: str):
    assert not (cfg.exclude_decoder_from_computation_graph and not cfg.include_actor_embedding_in_critic_input), \
        "Cannot include actor embedding in critic input when decoder is excluded from computation graph"

    device = get_device(cfg)
    print(f"using device: {device}")

    # env setup
    envs = env_setup(cfg, run_name)

    train_dataset, val_dataset, test_dataset, sub_training_envs = dataset_setup(
        cfg, run_name, max_level=cfg.max_level, start_level=cfg.start_level
    )
    envs.envs[0].unwrapped.start_goal_dataset = train_dataset

    valid_sequences = generate_valid_sequences()
    filtered_valid_sequences = preprocess_valid_sequences(valid_sequences,
                                                          device,
                                                          cfg.n_actions_in_seq,
                                                          cfg.end_of_sequence_token,
                                                          cfg.exploration_sequence_length_constraint)

    all_valid_sequences = preprocess_valid_sequences(valid_sequences,
                                                     device,
                                                     cfg.n_actions_in_seq,
                                                     cfg.end_of_sequence_token,
                                                     exploration_sequence_length_constraint=-1)

    maze_size = envs.single_observation_space.sample()[None].shape[2]
    n_input_channels_critic = envs.single_observation_space.shape[0]
    n_input_channels_actor = envs.single_observation_space.shape[0]
    observation_shape = (maze_size, maze_size)

    # load actor:
    proto_plan_noise = None
    actor_network = Actor(n_input_channels=n_input_channels_actor,
                          n_output_channels=cfg.actor_n_output_channels,
                          observation_shape=observation_shape,
                          noise=proto_plan_noise,
                          linear_layers=cfg.actor_linear_layers,
                          linear_layers_activation=cfg.actor_linear_layers_activation_function,
                          use_batchnorm_linear_layers=cfg.actor_use_batchnorm_linear_layers,
                          num_heads=cfg.n_proto_plan_candidates,
                          pe_embedding_dim=cfg.pe_embedding_dim).to(device)
    target_actor_network = Actor(n_input_channels=n_input_channels_actor,
                                 n_output_channels=cfg.actor_n_output_channels,
                                 observation_shape=observation_shape,
                                 noise=proto_plan_noise,
                                 linear_layers=cfg.actor_linear_layers,
                                 linear_layers_activation=cfg.actor_linear_layers_activation_function,
                                 use_batchnorm_linear_layers=cfg.actor_use_batchnorm_linear_layers,
                                 num_heads=cfg.n_proto_plan_candidates,
                                 pe_embedding_dim=cfg.pe_embedding_dim).to(device)
    target_actor_network.load_state_dict(actor_network.state_dict())
    actor_optimizer = optim.Adam(actor_network.parameters(), lr=cfg.actor_learning_rate, weight_decay=cfg.actor_weight_decay)

    print(f"actor model: {actor_network}")

    # load critic:
    action_seq_dim = (cfg.n_actions_in_seq if cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_INT else cfg.n_actions_in_seq * (cfg.action_space_size + 1)) + (cfg.actor_n_output_channels if cfg.include_actor_embedding_in_critic_input else 0)
    critic_network = Critic(n_input_channels=n_input_channels_critic, observation_shape=observation_shape,
                            action_seq_dim=action_seq_dim,
                            linear_layers=cfg.critic_linear_layers,
                            linear_layers_activation=cfg.critic_linear_layers_activation_function,
                            use_batchnorm_linear_layers=cfg.critic_use_batchnorm_linear_layers,
                            include_actor_embedding_in_critic_input=cfg.include_actor_embedding_in_critic_input).to(device)
    target_critic_network = Critic(n_input_channels=n_input_channels_critic, observation_shape=observation_shape,
                                   action_seq_dim=action_seq_dim,
                                   linear_layers=cfg.critic_linear_layers,
                                   linear_layers_activation=cfg.critic_linear_layers_activation_function,
                                   use_batchnorm_linear_layers=cfg.critic_use_batchnorm_linear_layers,
                                   include_actor_embedding_in_critic_input=cfg.include_actor_embedding_in_critic_input).to(device)
    target_critic_network.load_state_dict(critic_network.state_dict())
    critic_optimizer = optim.Adam(critic_network.parameters(), lr=cfg.critic_learning_rate, weight_decay=cfg.critic_weight_decay)

    print(f"critic model: {critic_network}")

    # load decoder wrapper
    load_pretrained_decoder = cfg.initialize_decoder_from_pretrained or not cfg.train_decoder_end_to_end
    decoder = get_decoder_api(decoder_model_path=cfg.decoder_model_path, decoder_seq_len=cfg.n_actions_in_seq, device=device, maze_n_actions=4,
                              use_gumble_in_decoder=cfg.use_gumble_in_decoder, penalize_cyclic_position_revisits=cfg.penalize_cyclic_position_revisits,
                              deterministic_inference=cfg.deterministic_inference, load_pretrained_weights=load_pretrained_decoder,
                              decoder_n_layer=cfg.decoder_n_layer, decoder_n_head=cfg.decoder_n_head,
                              decoder_n_embd=cfg.decoder_n_embd, decoder_dropout=cfg.decoder_dropout,
                              decoder_use_cuda_graph=cfg.decoder_use_cuda_graph,
                              decoder_soft_token_feedback=cfg.decoder_soft_token_feedback)
    
    # Set decoder to training mode if training end-to-end
    if cfg.train_decoder_end_to_end:
        decoder.model.to(device).train()
        print(f"Decoder mode: GPS-E2E (training end-to-end)")
        print(f"Decoder initialization: {'pretrained' if cfg.initialize_decoder_from_pretrained else 'random'}")
        # Create decoder optimizer
        decoder_optimizer = optim.Adam(decoder.model.parameters(), lr=cfg.decoder_learning_rate, weight_decay=cfg.decoder_weight_decay)
        print(f"decoder optimizer created with lr={cfg.decoder_learning_rate}, weight_decay={cfg.decoder_weight_decay}")
    else:
        decoder.model.to(device).eval()
        decoder_optimizer = None
        print(f"Decoder mode: GPS (frozen pretrained)")

    # generate mapping between sequence to encoder latent space z representation
    sequence_to_embedding: dict[tuple, torch.Tensor] = {}
    if cfg.include_actor_embedding_in_critic_input:
        for s in all_valid_sequences:
            s = s.flatten()
            _, _, _, _, z = decoder.model.forward(s.unsqueeze(0), get_z=True)
            sequence_to_embedding[tuple(s.tolist())] = z.detach()

    # Prepare padding tensors for various lengths
    padding_tensors = prepare_padding_tensors(cfg, device)

    rb = ReplayMemory(capacity=cfg.buffer_size)
    start_time = time.time()

    continue_training = True
    model_path = f"runs/{run_name}/{cfg.exp_name}.model"
    prefix, suffix = model_path.rsplit('/', 1)
    actor_model_path = f"{prefix}/actor"
    critic_model_path = f"{prefix}/critic"

    best_val_mean_reward = float("-inf")
    best_train_mean_reward = float("-inf")
    best_val_mean_success = float("-inf")
    best_train_mean_success = float("-inf")
    best_train_mean_valid_actions_in_sequences_per_episode = float("-inf")
    best_train_mean_unused_actions_in_sequences_per_episode = float("-inf")
    best_train_mean_agent_step_ratio_in_sequences_per_episode = float("-inf")

    best_val_mean_num_decoder_generations = float("-inf")
    best_train_mean_num_decoder_generations = float("-inf")
    best_val_mean_valid_actions_in_sequences_per_episode = float("-inf")
    best_val_mean_unused_actions_in_sequences_per_episode = float("-inf")
    best_val_mean_agent_step_ratio_in_sequences_per_episode = float("-inf")

    best_val_mean_value_global_step = 0
    best_train_mean_value_global_step = 0

    episode_decoder_generations = 0
    verbose_last_global_step = 0
    all_actions = []

    e_noise_decay = LinearNoiseDecay(initial_noise=cfg.start_e, final_noise=cfg.end_e, total_steps=cfg.total_steps_e)
    adaptive_temperature_decay = LinearNoiseDecay(initial_noise=cfg.temperature, final_noise=0.1, total_steps=cfg.total_steps_e * 2)
    rng_exploration = random.Random(cfg.seed)
    rng_sample_from_candidates = random.Random(cfg.seed)
    actor_gradient_norm = SoftAverager()
    critic_gradient_norm = SoftAverager()
    decoder_gradient_norm = SoftAverager()
    smoothed_num_unique_sequences = SoftAverager()

    if cfg.sequence_length_weight > 0:
        n_heads = cfg.n_proto_plan_candidates

        assert len(cfg.min_lengths_per_head) == n_heads
        assert len(cfg.max_lengths_per_head) == n_heads

        # Convert min_lengths and max_lengths to tensors
        min_lengths_tensor = torch.tensor(cfg.min_lengths_per_head, device=device, dtype=torch.int32)  # Shape: (n_heads,)
        max_lengths_tensor = torch.tensor(cfg.max_lengths_per_head, device=device, dtype=torch.int32)  # Shape: (n_heads,)

        # Reshape for broadcasting
        min_lengths_tensor = min_lengths_tensor.view(1, n_heads)  # Shape: (1, n_heads)
        max_lengths_tensor = max_lengths_tensor.view(1, n_heads)  # Shape: (1, n_heads)

    # TRY NOT TO MODIFY: start the game
    obs, obs_info = envs.reset(seed=cfg.seed)

    if cfg.evaluate_actor_diversity_every_k_epochs > 0:
        run_evaluation_k = int(cfg.evaluate_actor_diversity_every_k_epochs)
        fixed_obs_tensor = prepare_fixed_states(obs.copy(), device=device)
        f_name = 'actor_diversity_logs' + f"_{run_name}"
        f_name = f_name + f"_{cfg.slurm_job_id}" if cfg.slurm_job_id else f_name
        log_path = os.path.join(log_dir, f'{f_name}.txt')
        actor_diversity_logger = TrainingLogger(log_path, log_every_k=run_evaluation_k)

    global_step = 0
    episode_reward = 0
    episode_steps = 0

    while global_step < cfg.total_timesteps:
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)

            if cfg.actor_use_batchnorm_linear_layers:
                input = obs_tensor.repeat(2, 1, 1, 1)
                output = actor_network(input)
                actor_proto_plan_emb = output[0]  # Reduce the output back to a single sample
            else:
                actor_proto_plan_emb = actor_network(obs_tensor)

            exploration = False
            if global_step < cfg.learning_starts or rng_exploration.random() < e_noise_decay.get_noise(global_step):
                exploration = True

                if cfg.explore_without_critic:
                    action_seq, action_list_batch = explore_without_critic(filtered_valid_sequences)
                else:
                    action_seq, action_list_batch = explore_with_critic(filtered_valid_sequences, obs_tensor, actor_proto_plan_emb, critic_network,
                                                                        cfg, device, rng_sample_from_candidates)
            else:
                action_list_batch, action_list_proba_batch = decoder.gen_action_seq(
                    gen_input=actor_proto_plan_emb,
                    get_actions_as_one_hot=(cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT))

                batch_q_stack = eval_candidates_with_critic(action_list_batch, action_list_proba_batch, obs_tensor, actor_proto_plan_emb,
                                                            critic_network, cfg)

                action_seq, action_list_batch = select_action(action_list_batch, action_list_proba_batch, batch_q_stack, cfg, rng_sample_from_candidates)

        # Convert action sequence to discrete actions
        action_sequence = action_seq.argmax(-1)
        action_sequence, is_invalid_action = DenseVAE.trim_action_sequence_from_eos_tokens(action_sequence)

        # Store original observation for the whole sequence
        original_obs = obs.copy()
        original_obs_tensor = obs_tensor.clone()

        # Track observations, rewards, and terminations for the sequence
        sequence_obs = [original_obs.copy()]
        sequence_rewards = []
        sequence_terminations = []
        sequence_truncations = []

        # Execute actions one-by-one and record transitions
        current_action_idx = 0
        done = False

        episode_decoder_generations += 1
        all_actions.append(action_sequence)

        # Store the complete action tensor for buffer
        action_to_store = action_seq.argmax(
            -1) if cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_INT else action_seq
        if is_invalid_action:
            action_to_store = action_seq
            # Apply the penalty for invalid actions
            # Note: We'll apply this penalty when we execute steps

        # Process each action in the sequence individually
        while current_action_idx < len(action_sequence.tolist()) and not done:
            current_action = action_sequence[current_action_idx]

            # Execute a single step in the environment
            next_obs, reward, termination, truncation, infos = envs.step(np.array([current_action.cpu()]))

            # Apply penalty for invalid actions if this is the first step
            if current_action_idx == 0 and is_invalid_action:
                reward[0] = -20

            # Increment counters
            global_step += 1
            episode_steps += 1
            episode_reward += reward[0]

            # Record step information
            sequence_obs.append(next_obs.copy())
            sequence_rewards.append(reward[0])
            sequence_terminations.append(termination[0])
            sequence_truncations.append(truncation[0])

            # Check if this step ends the episode
            done = termination[0] or truncation[0]

            # Process buffer updates for this step
            current_obs_tensor = torch.tensor(sequence_obs[-2], dtype=torch.float32, device=device)
            next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device)

            # Handle final observation for truncation
            if truncation[0]:
                final_obs = infos["final_observation"][0]
                final_obs = np.expand_dims(final_obs, axis=0)
                next_obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device)
                # Also update the sequence_obs to maintain consistency
                sequence_obs[-1] = final_obs.copy()

            # Add individual step transition to buffer
            if cfg.push_every_one_step_transition_to_buffer:
                # Create single-step action representation
                single_action = torch.zeros_like(action_to_store)
                single_action[0] = action_to_store[current_action_idx]

                # Pad to maintain fixed length
                pad_length = cfg.n_actions_in_seq - 1
                padded_action = torch.concat((single_action[0].unsqueeze(0), padding_tensors[pad_length]))

                # Compute embedding if needed
                actor_proto_plan_emb_sub_sequence = torch.empty(
                    (cfg.n_proto_plan_candidates, cfg.actor_n_output_channels), device=device)
                if cfg.include_actor_embedding_in_critic_input:
                    action_flatten = padded_action.flatten()
                    sequence_embedding_key = tuple(action_flatten.tolist())
                    if sequence_embedding_key not in sequence_to_embedding:
                        _, _, _, _, z = decoder.model.forward(action_flatten.unsqueeze(0), get_z=True)
                        sequence_to_embedding[sequence_embedding_key] = z.detach()
                    actor_proto_plan_emb_sub_sequence = sequence_to_embedding[sequence_embedding_key]
                    if cfg.n_proto_plan_candidates > 1:
                        actor_proto_plan_emb_sub_sequence = actor_proto_plan_emb_sub_sequence.repeat(
                            cfg.n_proto_plan_candidates, 1)

                # Push to buffer
                rb.push(
                    current_obs_tensor,
                    next_obs_tensor,
                    padded_action,
                    np.array([reward[0]]),
                    np.array([termination[0]]),
                    actor_proto_plan_emb_sub_sequence
                )

            if (cfg.push_sub_sequences_to_buffer_move_end_point and
                    current_action_idx > 0 and
                    current_action_idx < len(action_sequence) - 1 and  # Not the last action (prevents full sequence)
                    current_action_idx % cfg.sub_sequences_min_jump_move_end_point == 0):
                # Create sub-sequence from start to current point
                sub_seq_actions = action_to_store[:current_action_idx + 1]
                pad_length = cfg.n_actions_in_seq - (current_action_idx + 1)
                sub_sequence_actions = torch.concat((sub_seq_actions, padding_tensors[pad_length]))

                # Compute embedding if needed
                actor_proto_plan_emb_sub_sequence = torch.empty(
                    (cfg.n_proto_plan_candidates, cfg.actor_n_output_channels), device=device)
                if cfg.include_actor_embedding_in_critic_input:
                    action_flatten = sub_sequence_actions.flatten()
                    sequence_embedding_key = tuple(action_flatten.tolist())
                    if sequence_embedding_key not in sequence_to_embedding:
                        _, _, _, _, z = decoder.model.forward(action_flatten.unsqueeze(0), get_z=True)
                        sequence_to_embedding[sequence_embedding_key] = z.detach()
                    actor_proto_plan_emb_sub_sequence = sequence_to_embedding[sequence_embedding_key]
                    if cfg.n_proto_plan_candidates > 1:
                        actor_proto_plan_emb_sub_sequence = actor_proto_plan_emb_sub_sequence.repeat(
                            cfg.n_proto_plan_candidates, 1)

                # Calculate cumulative reward for the sub-sequence
                cum_reward = sum(sequence_rewards[:current_action_idx + 1])

                # Push to buffer
                rb.push(
                    original_obs_tensor,
                    next_obs_tensor,
                    sub_sequence_actions,
                    np.array([cum_reward]),
                    np.array([termination[0]]),
                    actor_proto_plan_emb_sub_sequence
                )

            # Update for next step
            obs = next_obs.copy()
            current_action_idx += 1

            # ALGO LOGIC: training - train loop.
            if global_step > cfg.learning_starts:
                data = Transition(*zip(*rb.sample(cfg.batch_size)))
                with torch.no_grad():
                    batch_next_observations = torch.stack(data.next_observations, dim=0).squeeze(
                        1)  # dimension=(batch_size, 3, maze_size, maze_size)

                    target_actor_network.eval()
                    target_critic_network.eval()

                    repeated_batch_next_obs = batch_next_observations.repeat_interleave(cfg.n_proto_plan_candidates,
                                                                                        dim=0)

                    next_state_embeddings = target_actor_network(batch_next_observations)

                    # Generate action sequences and probabilities
                    next_state_actions, next_state_actions_probas = decoder.gen_action_seq(
                        gen_input=next_state_embeddings,
                        get_actions_as_one_hot=(
                                    cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT))

                    # Prepare action input for the critic
                    action_input_for_critic = get_action_format(act_one_hot=next_state_actions,
                                                                action_probas=next_state_actions_probas,
                                                                action_format=cfg.action_seq_representation,
                                                                cfg=cfg)

                    # Repeat batch observations and get Q-values
                    qf_next_target = target_critic_network(repeated_batch_next_obs, next_state_embeddings,
                                                           action_input_for_critic)

                    # Reshape Q-values to (batch_size, n_proto_plan_candidates)
                    qf_next_target = qf_next_target.view(cfg.batch_size, cfg.n_proto_plan_candidates)

                    # Get the best candidate indices and Q-values
                    best_candidate_indices = torch.argmax(qf_next_target, dim=1)
                    qf_next_target = qf_next_target[range(cfg.batch_size), best_candidate_indices]

                    if cfg.min_qf_value is not None:
                        qf_next_target = torch.clip(qf_next_target, min=cfg.min_qf_value, max=cfg.max_qf_value)
                    next_q_value = torch.tensor(np.array(data.rewards), device=device, dtype=torch.float32).squeeze(
                        1) + (
                                           1 - torch.tensor(np.array(data.dones), device=device,
                                                            dtype=torch.float32).squeeze(1)) * cfg.gamma * (
                                       qf_next_target).view(-1)

                    if cfg.verbose > 0 and global_step % cfg.verbose_steps_interval == 0:
                        # Get best action sequences
                        next_state_actions_rep = next_state_actions if cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT else next_state_actions_probas
                        best_next_state_actions = [
                            next_state_actions_rep[
                                i * cfg.n_proto_plan_candidates + best_candidate_indices[i]] for i in
                            range(cfg.batch_size)]

                        next_state_actions = best_next_state_actions

                        unique_next_state_actions = torch.unique(torch.stack(next_state_actions).argmax(-1).cpu(),
                                                                 dim=0)

                        next_state_target_action_list_final = []
                        for actions in unique_next_state_actions:
                            actions, _ = DenseVAE.trim_action_sequence_from_eos_tokens(actions)
                            next_state_target_action_list_final.append(actions.cpu().numpy())

                        unique_next_state_target_action_list_final = set(
                            tuple(sequence) for sequence in next_state_target_action_list_final)
                        unique_next_state_target_action_list_final = [np.array(arr) for arr in
                                                                      unique_next_state_target_action_list_final]

                        # NOTE: next_state_embeddings is (batch, n_heads, embedding_dim), and
                        # dim=2 reduces the *head* axis, not the embedding axis -- so this is
                        # not embedding similarity between states. At n_heads=1 the vectors
                        # have length 1 and cosine degenerates to sign agreement; at n_heads=5
                        # it compares a different quantity entirely, which is why A2 reported
                        # 0.952 against a 1-head baseline of 0.619 (see §9). Kept unchanged so
                        # historical runs stay comparable -- the §8.6 MLP-vs-GPT trajectory
                        # (1.0 -> 0.507 vs 0.706 -> 0.619) is in these units.
                        cosine_similarity_all_pairs = torch.cosine_similarity(next_state_embeddings.unsqueeze(1),
                                                                              next_state_embeddings.unsqueeze(0),
                                                                              dim=2)
                        cosine_similarity_all_pairs_mean = cosine_similarity_all_pairs.mean().item()
                        cosine_similarity_between_first_to_others_mean = cosine_similarity_all_pairs[0].mean().item()

                        # Correct version: pairwise cosine between the per-state embeddings of
                        # head 0, reduced over the embedding axis, with self-pairs excluded.
                        # 1.0 means the actor emits the same embedding for every state. This is
                        # the quantity check_z_influence.py reports (0.7783 on earthy-bird-16),
                        # so the two are directly comparable, and it is comparable across head
                        # counts -- which the metric above is not.
                        head0_embeddings = F.normalize(next_state_embeddings[:, 0, :], p=2, dim=1)
                        head0_similarity = head0_embeddings @ head0_embeddings.T
                        off_diagonal_mask = ~torch.eye(head0_similarity.shape[0], dtype=torch.bool,
                                                       device=head0_similarity.device)
                        actor_embedding_cosine = head0_similarity[off_diagonal_mask].mean().item()
                        print(f"actor embedding cosine (head 0, excl. self): {actor_embedding_cosine}")
                        writer.add_scalar("actor/embedding_cosine", actor_embedding_cosine, global_step)

                        _, counts = torch.unique(batch_next_observations.clone().cpu(), dim=0, return_counts=True)
                        num_same_states = (counts > 1).sum().item()
                        num_unique_sequences = len(unique_next_state_target_action_list_final)
                        # smoothed num unique sequences
                        smoothed_num_unique_sequences.add_value(num_unique_sequences)
                        print(f"num unique states: {cfg.batch_size - num_same_states}")
                        print(f"num unique action sequences: {num_unique_sequences}")
                        print(f"smoothed num unique action sequences: {smoothed_num_unique_sequences.smoothed_value}")
                        print(
                            f"mean length action sequences: {np.mean([len(sequence.tolist()) for sequence in unique_next_state_target_action_list_final])}")
                        print(f"mean cosine similarity all pairs: {cosine_similarity_all_pairs_mean}")
                        print(
                            f"mean cosine similarity first to others: {cosine_similarity_between_first_to_others_mean}")

                        print(
                            f"15 examples of next state unique action sequences:\n{[sequence.tolist() for sequence in unique_next_state_target_action_list_final[:15]]}")
                        writer.add_scalar("sequences/num_unique", smoothed_num_unique_sequences.smoothed_value,
                                          global_step)

                observations = torch.stack(data.observations, dim=0).squeeze(1)
                actor_proto_plan_emb = torch.empty(
                    (cfg.batch_size, cfg.n_proto_plan_candidates, cfg.actor_n_output_channels), device=device)
                if cfg.include_actor_embedding_in_critic_input:
                    actor_proto_plan_emb = torch.stack(data.actor_proto_plan_emb, dim=0)[:, 0:1, :]

                critic_stacked_actions = torch.stack([torch.Tensor(array) for array in data.actions], dim=0) if \
                    cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_INT else \
                    torch.stack([torch.Tensor(array) for array in data.actions], dim=0).flatten(1)
                qf1_a_values = critic_network(observations, actor_proto_plan_emb, critic_stacked_actions).view(-1)
                if cfg.min_qf_value is not None:
                    qf1_a_values = torch.clip(qf1_a_values, min=cfg.min_qf_value, max=cfg.max_qf_value)

                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

                # optimize the model
                critic_optimizer.zero_grad()
                qf1_loss.backward()
                critic_optimizer.step()

                if global_step % cfg.actor_policy_frequency == 0:
                    actor_embeddings = actor_network(observations)

                    if cfg.exclude_decoder_from_computation_graph:
                        with torch.no_grad():
                            decoded_action_one_hot_seq, decoded_action_probas_seq = decoder.gen_action_seq(
                                actor_embeddings,
                                get_actions_as_one_hot=(
                                            cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT),
                                exclude_decoder_from_computation_graph=cfg.exclude_decoder_from_computation_graph)
                            decoded_action_seq = get_action_format(action_format=cfg.action_seq_representation,
                                                                   action_probas=decoded_action_probas_seq,
                                                                   act_one_hot=decoded_action_one_hot_seq,
                                                                   cfg=cfg)
                    else:
                        decoded_action_one_hot_seq, decoded_action_probas_seq = decoder.gen_action_seq(actor_embeddings,
                                                                                                       get_actions_as_one_hot=(
                                                                                                                   cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT),
                                                                                                       exclude_decoder_from_computation_graph=cfg.exclude_decoder_from_computation_graph)
                        decoded_action_seq = get_action_format(action_format=cfg.action_seq_representation,
                                                               action_probas=decoded_action_probas_seq,
                                                               act_one_hot=decoded_action_one_hot_seq,
                                                               cfg=cfg)

                    qf_values = critic_network(observations.repeat_interleave(cfg.n_proto_plan_candidates, dim=0),
                                               actor_embeddings,
                                               decoded_action_seq)
                    if cfg.min_qf_value is not None:
                        qf_values = torch.clip(qf_values, min=cfg.min_qf_value, max=cfg.max_qf_value)

                    # # Scale gradients coming from critic network by a factor (e.g., 0.1)
                    # scaling_factor = 0.1  # Adjust this value as needed
                    # qf_values = scale_grad(qf_values, scaling_factor)

                    # Reshape qf_values to (batch_size, n_heads)
                    qf_values = qf_values.view(observations.shape[0], cfg.n_proto_plan_candidates)

                    # Apply softmax to get weights for each head
                    softmax_weights = F.softmax(qf_values / cfg.temperature, dim=1)

                    # Compute weighted average of Q-values
                    weighted_qf_values = (qf_values * softmax_weights).sum(dim=1)

                    # Compute actor loss
                    actor_loss = -weighted_qf_values.mean()

                    # Add diversity regularization
                    if cfg.diversity_weight > 0:
                        diversity_loss = compute_diversity_loss_with_normalized_embeddings(actor_embeddings)
                        actor_loss += cfg.diversity_weight * diversity_loss

                    # Add invalid actions loss / Add sequence length loss
                    if cfg.invalid_actions_weight > 0 or cfg.sequence_length_weight > 0:
                        # Compute sequence lengths based on the first occurrence of the end-of-sequence token
                        vocab_size = cfg.action_space_size + 1
                        batch_action_sequences = decoded_action_one_hot_seq.view(cfg.batch_size,
                                                                                 cfg.n_proto_plan_candidates,
                                                                                 cfg.n_actions_in_seq, vocab_size)

                        lengths = compute_sequence_lengths_differentiable(batch_action_sequences)

                        if cfg.invalid_actions_weight > 0:
                            # Compute a differentiable approximation of the step function
                            # This will be close to 1 for very small lengths and close to 0 otherwise
                            epsilon = 1e-6
                            zero_length_indicator = torch.exp(-lengths / epsilon)

                            # Sum the indicators to get an approximation of the count of zero-length sequences
                            num_zero_length_sequences = zero_length_indicator.sum()

                            # Update the actor loss
                            actor_loss += cfg.invalid_actions_weight * num_zero_length_sequences

                        if cfg.sequence_length_weight > 0:
                            # Compute penalties
                            penalty_below_min = torch.clamp(min_lengths_tensor - lengths,
                                                            min=0)  # Shape: (batch_size, n_heads)
                            penalty_above_max = torch.clamp(lengths - max_lengths_tensor,
                                                            min=0)  # Shape: (batch_size, n_heads)
                            length_penalty = penalty_below_min + penalty_above_max  # Shape: (batch_size, n_heads)

                            # Aggregate the penalties
                            total_length_penalty = length_penalty.sum()  # Scalar

                            # Update the actor loss
                            actor_loss += cfg.sequence_length_weight * total_length_penalty

                    actor_weights = None
                    if cfg.validate_actor_weights_update:
                        actor_weights = copy.deepcopy(actor_network.state_dict())

                    actor_optimizer.zero_grad()
                    # The decoder sits inside the actor's computation graph when training
                    # end-to-end, so its gradients must be cleared on the same schedule as
                    # the actor's -- otherwise every backward adds to the previous ones and
                    # the optimizer steps along a running sum instead of the current grad.
                    if cfg.train_decoder_end_to_end:
                        decoder_optimizer.zero_grad()
                    actor_loss.backward()

                    # Clip before the gradient-monitoring block below, so the norms it logs
                    # are the ones actually applied.
                    if cfg.actor_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(actor_network.parameters(),
                                                       cfg.actor_grad_clip)
                        if cfg.train_decoder_end_to_end:
                            torch.nn.utils.clip_grad_norm_(decoder.model.parameters(),
                                                           cfg.actor_grad_clip)

                    # Gradient monitoring
                    if global_step > 10 and global_step % 250 == 0:
                        total_norm = 0
                        parameters = [p for p in actor_network.parameters() if p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.norm(2)  # Avoid detach().data
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5

                        # smoothed gradient norm
                        actor_gradient_norm.add_value(total_norm)

                        print(f'Actor Gradient Norm: {total_norm}')
                        print(f'Smoothed Gradient Norm: {actor_gradient_norm.smoothed_value}')
                        writer.add_scalar("gradient/actor_norm", actor_gradient_norm.smoothed_value, global_step)

                        if math.isnan(total_norm) or math.isinf(total_norm):
                            print(f"Warning: Actor Gradient norm is {total_norm}")
                            continue_training = False

                        # Log gradient statistics
                        for name, param in actor_network.named_parameters():
                            if param.grad is not None:
                                grad = param.grad.data
                                print(f'Layer: {name}, Grad stats: '
                                      f'Mean {grad.mean().item():.10f}, '
                                      f'Std {grad.std().item():.10f}, '
                                      f'Min {grad.min().item():.10f}, '
                                      f'Max {grad.max().item():.10f}')

                                # Check for NaN or inf
                                if torch.isnan(grad).any() or torch.isinf(grad).any():
                                    print(f'Actor: NaN or inf detected in gradients of layer: {name}')
                                    continue_training = False

                        total_norm = 0
                        parameters = [p for p in decoder.model.decoder.parameters() if
                                      p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5

                        # smoothed gradient norm
                        decoder_gradient_norm.add_value(total_norm)

                        print(f'Decoder Gradient Norm: {total_norm}')
                        print(f'Smoothed Gradient Norm: {decoder_gradient_norm.smoothed_value}')
                        writer.add_scalar("gradient/decoder_norm", decoder_gradient_norm.smoothed_value, global_step)

                        if math.isnan(total_norm) or math.isinf(total_norm):
                            print(f"Warning: Decoder Gradient norm is {total_norm}.")
                            continue_training = False

                        for name, param in decoder.model.decoder.named_parameters():
                            if param.grad is not None:
                                grad = param.grad.data
                                print(f'Decoder Layer: {name}, Grad stats: '
                                      f'Mean {grad.mean().item():.10f}, '
                                      f'Std {grad.std().item():.10f}, '
                                      f'Min {grad.min().item():.10f}, '
                                      f'Max {grad.max().item():.10f}')

                                # Check for NaN or inf
                                if torch.isnan(grad).any() or torch.isinf(grad).any():
                                    print(f'Decoder: NaN or inf detected in gradients of layer: {name}')
                                    continue_training = False

                        total_norm = 0
                        parameters = [p for p in critic_network.parameters() if p.grad is not None and p.requires_grad]
                        for p in parameters:
                            param_norm = p.grad.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5

                        # smoothed gradient norm
                        critic_gradient_norm.add_value(total_norm)

                        print(f'Critic Gradient Norm: {total_norm}')
                        print(f'Smoothed Gradient Norm: {critic_gradient_norm.smoothed_value}')
                        writer.add_scalar("gradient/critic_norm", critic_gradient_norm.smoothed_value, global_step)

                        if math.isnan(total_norm) or math.isinf(total_norm):
                            print(f"Warning: Critic Gradient norm is {total_norm}.")
                            continue_training = False

                        for name, param in critic_network.named_parameters():
                            if param.grad is not None:
                                grad = param.grad.data
                                print(f'Critic Layer: {name}, Grad stats: '
                                      f'Mean {grad.mean().item():.10f}, '
                                      f'Std {grad.std().item():.10f}, '
                                      f'Min {grad.min().item():.10f}, '
                                      f'Max {grad.max().item():.10f}')

                                # Check for NaN or inf
                                if torch.isnan(grad).any() or torch.isinf(grad).any():
                                    print(f'Critic: NaN or inf detected in gradients of layer: {name}')
                                    continue_training = False

                    # A non-finite gradient makes this update meaningless, so skip the
                    # optimizer steps and unwind the training loop. The bare `return` this
                    # replaces did skip the steps, but it also abandoned the rest of
                    # train() -- including the final test evaluation -- so a run that
                    # destabilised partway through exited 0, reported "Finished", and
                    # produced no test numbers at all despite its best-on-validation
                    # checkpoints already being on disk. Unwinding instead reaches the end
                    # of train(), where that evaluation runs against those checkpoints.
                    if not continue_training:
                        print(f"[GPS] aborting training at global_step={global_step} after a "
                              f"non-finite gradient. Running the final test evaluation on the "
                              f"best-on-validation checkpoints saved so far.", flush=True)
                        break

                    actor_optimizer.step()

                    # Update decoder if training end-to-end
                    if cfg.train_decoder_end_to_end:
                        decoder_optimizer.step()

                    if global_step % 250 == 0:
                        def process_batch(decoded_action_seq, observations, qf_values, softmax_weights, cfg):
                            batch_action_sequences = decoded_action_seq.argmax(-1).view(cfg.batch_size,
                                                                                        cfg.n_proto_plan_candidates,
                                                                                        cfg.n_actions_in_seq)

                            trimmed_sequences = [
                                [DenseVAE.trim_action_sequence_from_eos_tokens(seq)[0] for seq in batch]
                                for batch in batch_action_sequences]

                            sequence_lengths = np.array(
                                [[len(seq) for seq in head_sequences] for head_sequences in trimmed_sequences])
                            average_lengths = np.mean(sequence_lengths, axis=0)

                            unique_tensors, counts = torch.unique(observations.cpu(), dim=0, return_counts=True)
                            unique_obs_indices = torch.nonzero(counts == 1).squeeze(1)[:5].tolist()  # 5 unique samples
                            unique_obs_heads = [trimmed_sequences[i] for i in unique_obs_indices]

                            return {
                                'chosen_actor_head_dist': Counter(qf_values.argmax(1).tolist()),
                                'average_lengths': average_lengths,
                                'unique_count': unique_tensors.shape[0],
                                'unique_softmax_weights': softmax_weights[unique_obs_indices, :].detach().cpu().numpy(),
                                'unique_q_values': qf_values[unique_obs_indices, :].detach().cpu().numpy(),
                                'unique_obs_heads': unique_obs_heads
                            }

                        def print_results(global_step, cfg, results):
                            print(f'======================== global step: {global_step} ========================')
                            print(f"temperature: {cfg.temperature}")
                            print(f"chosen actor head distribution inside batch: {results['chosen_actor_head_dist']}")
                            print(f"action sequence average length per head inside batch: {results['average_lengths']}")
                            print(
                                f"count unique states per buffer {results['unique_count']} out of {cfg.batch_size} in batch")
                            print("5 unique samples' softmax weights:")
                            print(results['unique_softmax_weights'].round(4))
                            print("5 unique samples q values:")
                            print(results['unique_q_values'].round(4))
                            print('================================')
                            print("action sequence samples per head:")
                            for s in results['unique_obs_heads']:
                                print([l.cpu().numpy().tolist() for l in s])
                                print()

                        decoded_action_seq = decoded_action_one_hot_seq if cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT else decoded_action_probas_seq
                        results = process_batch(decoded_action_seq, observations, qf_values, softmax_weights, cfg)
                        print_results(global_step, cfg, results)

                    # Update temperature
                    if cfg.use_adaptive_temperature:
                        cfg.temperature = update_temperature(adaptive_temperature_decay, global_step)

                    if cfg.validate_actor_weights_update and actor_weights is not None:
                        if compare_dicts(actor_weights, actor_network.state_dict()):
                            raise Exception("error - actor's weights should be updated")

                # Update the targets networks
                if global_step % cfg.actor_target_network_frequency == 0:
                    update_target_network(actor_network, target_actor_network, cfg.tau)

                # Update the targets networks
                if global_step % cfg.critic_target_network_frequency == 0:
                    update_target_network(critic_network, target_critic_network, cfg.tau)

                if global_step % cfg.verbose_steps_interval == 0 and global_step % cfg.actor_policy_frequency == 0:
                    writer.add_scalar("losses/critic_values", qf1_a_values.mean().item(), global_step)
                    writer.add_scalar("losses/critic_loss", qf1_loss.item(), global_step)
                    writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                    if cfg.sequence_length_weight > 0:
                        writer.add_scalar("losses/sequence_length_loss", total_length_penalty.item(), global_step)
                    if cfg.diversity_weight > 0:
                        writer.add_scalar("losses/diversity_loss", diversity_loss.item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                if cfg.verbose > 0 and global_step % cfg.actor_policy_frequency == 0 and global_step % min(
                        cfg.train_eval_freq, cfg.total_timesteps) == 0:
                    print("######################")
                    print(f"losses/critic_values={qf1_a_values.mean().item()}")
                    print(f"losses/critic_loss={qf1_loss.item()}")
                    print(f"losses/actor_loss={actor_loss.item()}")
                    if cfg.sequence_length_weight > 0:
                        print(f"losses/sequence_length_loss={total_length_penalty.item()}")
                    if cfg.diversity_weight > 0:
                        print(f"losses/diversity_loss={diversity_loss.item()}")
                    print("######################")

                # evaluation on validation dataset with critic
                best_val_mean_reward, best_val_mean_success, best_val_mean_num_decoder_generations, \
                    best_val_mean_value_global_step, best_val_mean_valid_actions_in_sequences_per_episode, \
                    best_val_mean_unused_actions_in_sequences_per_episode, best_val_mean_agent_step_ratio_in_sequences_per_episode, _ = perform_evaluation(
                    cfg,
                    dataset=val_dataset,
                    run_name=run_name,
                    actor=target_actor_network,
                    critic=target_critic_network,
                    decoder=decoder,
                    device=device,
                    global_step=global_step,
                    writer=writer,
                    eval_on_train_dataset=False,
                    with_critic=True,
                    eval_freq=cfg.val_eval_freq,
                    actor_path=actor_model_path,
                    critic_path=critic_model_path,
                    best_mean_reward=best_val_mean_reward,
                    best_mean_success=best_val_mean_success,
                    best_mean_num_decoder_generations=best_val_mean_num_decoder_generations,
                    best_mean_value_global_step=best_val_mean_value_global_step,
                    best_mean_valid_actions_in_sequences=best_val_mean_valid_actions_in_sequences_per_episode,
                    best_mean_unused_actions_in_sequences=best_val_mean_unused_actions_in_sequences_per_episode,
                    best_mean_gent_step_ratio_in_sequences=best_val_mean_agent_step_ratio_in_sequences_per_episode,
                )

                # evaluation on train dataset with critic
                best_train_mean_reward, best_train_mean_success, best_train_mean_num_decoder_generations, \
                    best_train_mean_value_global_step, best_train_mean_valid_actions_in_sequences_per_episode, \
                    best_train_mean_unused_actions_in_sequences_per_episode, best_train_mean_agent_step_ratio_in_sequences_per_episode, _ = perform_evaluation(
                    cfg,
                    dataset=sub_training_envs,
                    run_name=run_name,
                    actor=target_actor_network,
                    critic=target_critic_network,
                    decoder=decoder,
                    device=device,
                    global_step=global_step,
                    writer=writer,
                    eval_on_train_dataset=True,
                    with_critic=True,
                    eval_freq=cfg.train_eval_freq,
                    actor_path=actor_model_path,
                    critic_path=critic_model_path,
                    best_mean_reward=best_train_mean_reward,
                    best_mean_success=best_train_mean_success,
                    best_mean_num_decoder_generations=best_train_mean_num_decoder_generations,
                    best_mean_value_global_step=best_train_mean_value_global_step,
                    best_mean_valid_actions_in_sequences=best_train_mean_valid_actions_in_sequences_per_episode,
                    best_mean_unused_actions_in_sequences=best_train_mean_unused_actions_in_sequences_per_episode,
                    best_mean_gent_step_ratio_in_sequences=best_train_mean_agent_step_ratio_in_sequences_per_episode,
                )

                if cfg.verbose > 0 and global_step % cfg.actor_policy_frequency == 0 and global_step % min(
                        cfg.train_eval_freq, cfg.total_timesteps) == 0:
                    print("######################")
                    print(f"best val mean reward={best_val_mean_reward}")
                    print(f"best val mean success={best_val_mean_success}")
                    print()
                    print(f"best train mean reward={best_train_mean_reward}")
                    print(f"best train mean success={best_train_mean_success}")
                    print()
                    print(f"best val mean global step={best_val_mean_value_global_step}")
                    print(f"best train mean global step={best_train_mean_value_global_step}")
                    print("######################")

                if cfg.eval_test_dataset_during_training_freq > 1 and global_step % cfg.eval_test_dataset_during_training_freq == 0:
                    eval_test_dataset(test_dataset,
                                      cfg,
                                      writer,
                                      n_input_channels_actor,
                                      n_input_channels_critic,
                                      observation_shape,
                                      decoder,
                                      device,
                                      actor_model_path=actor_model_path,
                                      critic_model_path=critic_model_path,
                                      global_step=global_step)

                if cfg.evaluate_actor_diversity_every_k_epochs > 0 and global_step % run_evaluation_k == 0:
                    with torch.no_grad():
                        current_fixed_embedding = actor_network(fixed_obs_tensor)
                        # Use local names here. This diagnostic decode used to assign to
                        # action_list_batch / action_list_proba_batch, which are the
                        # acting loop's variables -- and because it calls gen_action_seq
                        # without get_actions_as_one_hot, it returns (None, probs), so it
                        # left action_list_batch as None. If the episode happened to end
                        # on the same step, the verbose episode logger below then did
                        # None.argmax(-1) and killed the run. Intermittent by nature: it
                        # needs a step that is a multiple of evaluate_actor_diversity_every_k_epochs
                        # AND an episode ending on it (observed at global_step=650000,
                        # i.e. 13 x 50000, on job 22379902).
                        _, diversity_action_probas = decoder.gen_action_seq(
                            gen_input=current_fixed_embedding)
                        action_seq_per_state = diversity_action_probas.argmax(-1).view(fixed_obs_tensor.shape[0],
                                                                                       cfg.n_proto_plan_candidates,
                                                                                       cfg.n_actions_in_seq)
                        trimmed_sequences = [[DenseVAE.trim_action_sequence_from_eos_tokens(seq)[0] for seq in batch]
                                             for batch in action_seq_per_state]
                        actor_diversity_logger.log_metrics(iteration=global_step,
                                                           trimmed_action_seq_per_state=trimmed_sequences)

            # Log and reset if episode is done
            if done:
                # Log episode stats
                print(
                    f"global_step={global_step}, episodic_length={episode_steps}, "
                    f"episodic_return={episode_reward}, episodic_decisions={episode_decoder_generations}"
                )
                writer.add_scalar("charts/episodic_return", episode_reward, global_step)
                writer.add_scalar("charts/episodic_length", episode_steps, global_step)
                writer.add_scalar("charts/episodic_decisions", episode_decoder_generations, global_step)

                # Verbose logging if configured
                if not exploration and global_step - verbose_last_global_step > cfg.verbose_steps_interval and episode_steps < 30:
                    verbose_last_global_step = global_step

                    action_list_final = []
                    action_list = action_list_batch if cfg.action_seq_representation == ActionSeqRepresentation.ACTION_SEQ_AS_ONE_HOT else action_list_proba_batch
                    for actions in action_list.argmax(-1):
                        actions, _ = DenseVAE.trim_action_sequence_from_eos_tokens(actions)
                        action_list_final.append(actions.cpu().numpy())

                    print(
                        f"final proto plan sequences of candidates: {[sequence.tolist() for sequence in action_list_final]}")
                    print(
                        f"examples of chosen proto plan sequences of current episode:\n{[sequence.tolist() for sequence in all_actions]}")
                    print()

                # Reset episode counters
                episode_reward = 0
                episode_steps = 0
                episode_decoder_generations = 0
                all_actions = []

                # Reset environment
                obs, _ = envs.reset(seed=cfg.seed)
                break

        # Always store the complete sequence transition in the buffer,
        # whether it completed or was interrupted by termination/truncation
        if len(sequence_obs) > 1:
            # Calculate final observation tensor
            final_obs = sequence_obs[-1]
            final_obs_tensor = torch.tensor(final_obs, dtype=torch.float32, device=device)

            # No need to handle truncation here since we already updated sequence_obs
            # in the step loop when processing truncations

            # Calculate cumulative reward
            total_reward = sum(sequence_rewards)

            # Get termination status from the last step
            is_terminated = len(sequence_terminations) > 0 and sequence_terminations[-1]

            # Add the full sequence transition to buffer
            rb.push(
                original_obs_tensor,
                final_obs_tensor,
                action_to_store,
                np.array([total_reward]),
                np.array([is_terminated]),
                actor_proto_plan_emb.squeeze(0)
            )

            # Now handle sub-sequences with moving start point - we do this at the end
            # of the sequence when we have all rewards
            if cfg.push_sub_sequences_to_buffer_move_start_point:
                for start_index in range(1, len(sequence_obs) - 2, cfg.sub_sequences_min_jump_move_start_point):
                    # Create sub-sequence starting at start_index
                    sub_seq_actions = action_to_store[start_index:]

                    # Use the padding tensors approach consistent with the rest of the code
                    # We should pad at the end to maintain the correct sequence length
                    pad_length = start_index

                    # Concat the sub-sequence and padding
                    sub_sequence_actions = torch.cat([sub_seq_actions, padding_tensors[pad_length]], dim=0)

                    # Get the observation at start_index
                    start_obs_tensor = torch.tensor(sequence_obs[start_index], device=device, dtype=torch.float32)

                    # Compute embedding if needed
                    actor_proto_plan_emb_sub_sequence = torch.empty(
                        (cfg.n_proto_plan_candidates, cfg.actor_n_output_channels), device=device)
                    if cfg.include_actor_embedding_in_critic_input:
                        action_flatten = sub_sequence_actions.flatten()
                        sequence_embedding_key = tuple(action_flatten.tolist())
                        if sequence_embedding_key not in sequence_to_embedding:
                            _, _, _, _, z = decoder.model.forward(action_flatten.unsqueeze(0), get_z=True)
                            sequence_to_embedding[sequence_embedding_key] = z.detach()
                        actor_proto_plan_emb_sub_sequence = sequence_to_embedding[sequence_embedding_key]
                        if cfg.n_proto_plan_candidates > 1:
                            actor_proto_plan_emb_sub_sequence = actor_proto_plan_emb_sub_sequence.repeat(
                                cfg.n_proto_plan_candidates, 1)

                    # Calculate cumulative reward from start_index to end
                    cum_reward = sum(sequence_rewards[start_index:])

                    # Push to buffer
                    rb.push(
                        start_obs_tensor,
                        final_obs_tensor,
                        sub_sequence_actions,
                        np.array([cum_reward]),
                        np.array([is_terminated]),
                        actor_proto_plan_emb_sub_sequence
                    )


        # Dedented out of the `len(sequence_obs) > 1` block above so that an abort always
        # unwinds: a sequence that took no environment steps would otherwise skip this
        # check and keep the outer loop running.
        if not continue_training:
            break

    if not cfg.save_model or (cfg.val_eval_freq < 0 and cfg.save_model):
        torch.save(target_actor_network.state_dict(), actor_model_path)
        torch.save(target_critic_network.state_dict(), critic_model_path)
        save_msg = f"actor model saved to {actor_model_path} and critic model saved to {critic_model_path}"
        
        # Save decoder if training end-to-end
        if cfg.train_decoder_end_to_end:
            decoder_model_path = critic_model_path.replace('/critic', '/decoder')
            torch.save(decoder.model.state_dict(), decoder_model_path)
            save_msg += f" and decoder model saved to {decoder_model_path}"
        
        print(save_msg)

    envs.close()

    # eval on test set with critic
    eval_test_dataset(test_dataset,
                      cfg,
                      writer,
                      n_input_channels_actor,
                      n_input_channels_critic,
                      observation_shape,
                      decoder,
                      device,
                      actor_model_path=actor_model_path,
                      critic_model_path=critic_model_path)

    if cfg.track and cfg.evaluate_actor_diversity_every_k_epochs > -1:
        wandb.save(str(actor_diversity_logger.filepath))


def set_seed(
    seed: int, env: Optional[gym.Env] = None, deterministic_torch: bool = False
):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)
    torch.backends.cudnn.deterministic = deterministic_torch


def get_device(cfg: Config) -> torch.device:
    if torch.cuda.is_available() and cfg.cuda:
        return torch.device("cuda")

    if torch.backends.mps.is_available() and cfg.mps:
        return torch.device("mps")

    return torch.device("cpu")


def wandb_init(cfg: Config, run_name: str, log_dir: str ="./wandb_logs") -> tuple[Run, str]:

    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    run = wandb.init(
        project=cfg.wandb_project_name,
        entity=cfg.wandb_entity,
        sync_tensorboard=True,
        config=vars(cfg),
        monitor_gym=True,
        save_code=True,
        settings=wandb.Settings(code_dir="generative")
    )
    run.name = f"{run_name}__{run.name}"

    return run, log_dir


if __name__ == "__main__":
    args = tyro.cli(Config)

    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    assert (not args.mps and not args.cuda) or (args.mps and not args.cuda) or (args.cuda and not args.mps)

    run_name = f"{args.env_id}__{args.exp_name}"

    log_dir = './'
    if args.track:
        run, log_dir = wandb_init(args, run_name)
        run_name = run.name

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    set_seed(seed=args.seed, deterministic_torch=args.torch_deterministic)

    train(args, run_name, writer, log_dir)

    writer.close()
