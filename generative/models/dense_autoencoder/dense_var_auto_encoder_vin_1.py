import itertools
import os
import pickle
import random
from collections import defaultdict, Counter
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Laplace
from torch.nn.utils.rnn import pad_sequence

# Support both `python -m ...` / package imports and running this file directly
# (the __main__ training block below), which would otherwise break the relative import.
try:
    from .mingpt_decoder import GPT, GPTConfig
except ImportError:  # pragma: no cover - fallback for direct script execution
    from mingpt_decoder import GPT, GPTConfig

N_ACTIONS = 4
VAR = 1
seed = 42
DEFAULT_NOISE_LIST = ["normal_0"]
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
# https://medium.com/@rekalantar/variational-auto-encoder-vae-pytorch-tutorial-dce2d2fe0f5f


class TensorCache:
    def __init__(self, max_size=10000):
        self.cache = dict()
        self.max_size = max_size

    @staticmethod
    def get_cache_key(tensor, *args, **kwargs):
        # Convert tensor to list for hashing
        return (tuple(tensor.tolist()),) + args + tuple(sorted(kwargs.items()))

    def __call__(self, fn):
        def wrapped_fn(tensor, *args, **kwargs):
            key = self.get_cache_key(tensor, *args, **kwargs)
            if key in self.cache:
                return self.cache[key]
            result = fn(tensor, *args, **kwargs)
            self.cache[key] = result
            if len(self.cache) > self.max_size:
                self.cache.popitem()
            return result
        return wrapped_fn


class DenseVAE(nn.Module):
    def __init__(self, input_length: int, n_words: int, device: torch.device,
                 variance_for_sample: int = 1, decoder_input_size: int = 16,
                 decoder_n_layer: int = 2, decoder_n_head: int = 4, decoder_n_embd: int = 32,
                 decoder_dropout: Optional[float] = None):
        """
        Args:
            decoder_n_layer / decoder_n_head / decoder_n_embd: size of the Transformer
                decoder trunk. Exposed so the shape can be swept from the command line --
                the autoregressive decode loop costs one trunk pass per token, so n_layer
                is close to a linear knob on decoder wall-clock time.
            decoder_dropout: if given, overrides all three GPT dropout rates at once.
                ``None`` keeps minGPT's own defaults.
        """
        super().__init__()
        self.decoder_input_size = decoder_input_size
        self.device = device
        self.input_length = input_length
        self.n_words = n_words

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_length * n_words, self.decoder_input_size * 2),
            torch.nn.InstanceNorm1d(self.decoder_input_size * 2),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(self.decoder_input_size * 2, self.decoder_input_size),
            torch.nn.InstanceNorm1d(self.decoder_input_size),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(self.decoder_input_size, self.decoder_input_size),
            torch.nn.InstanceNorm1d(self.decoder_input_size),
            torch.nn.Tanh()
        )

        self.mean_layer = nn.Linear(self.decoder_input_size, self.decoder_input_size)
        self.logvar_layer = nn.Linear(self.decoder_input_size, self.decoder_input_size)
        self.norm = torch.distributions.Normal(0, variance_for_sample)

        # Initialize the encoder / mean / logvar weights with the existing strategy.
        # NOTE: this runs *before* the GPT decoder is built so that the Transformer keeps
        # minGPT's own (normal, std=0.02) initialisation rather than the Kaiming scheme.
        self.apply(self._init_weights)

        # Decoder: a small causal Transformer conditioned on the latent z (replaces the
        # previous MLP). z is projected to a prefix token and the action sequence is
        # generated autoregressively. block_size == input_length, vocab_size == n_words.
        gpt_kwargs = dict(
            vocab_size=n_words,
            block_size=input_length,
            decoder_input_size=decoder_input_size,
            n_layer=decoder_n_layer,
            n_head=decoder_n_head,
            n_embd=decoder_n_embd,
        )
        if decoder_dropout is not None:
            gpt_kwargs.update(
                embd_pdrop=decoder_dropout,
                resid_pdrop=decoder_dropout,
                attn_pdrop=decoder_dropout,
            )
        self.gpt_config = GPTConfig(**gpt_kwargs)
        self.decoder = GPT(self.gpt_config)

    def _init_weights(self, module):
        """Initialize weights using Kaiming initialization for LeakyReLU networks."""
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight,
                                     a=0.2,  # slope=0.2 for LeakyReLU(0.2)
                                     mode='fan_in',
                                     nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar

    def decode(self, z: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Teacher-forced decode through the conditional Transformer.

        Args:
            z: latent vector, shape (B, decoder_input_size).
            idx: ground-truth integer tokens for teacher forcing, shape (B, T).

        Returns:
            Vocabulary logits of shape (B, T + 1, n_words). Position 0 is produced from
            the z-prefix token (it predicts the first action); positions 1..T are fed by
            the input tokens.
        """
        logits, _ = self.decoder(z, idx)
        return logits

    def reparameterization(self, mean, var):
        epsilon = self.norm.sample(var.shape).to(self.device)
        z = mean + var * epsilon
        return z

    def forward(self, x: torch.Tensor, get_z: bool = False):
        """Encode a token sequence, sample z, and teacher-force the decoder.

        Args:
            x: integer token indices of shape (B, T) (T == input_length, padded).
            get_z: also return the sampled latent vector.

        Returns:
            (logits, mean, logvar[, z]) where logits has shape (B, T + 1, n_words).
        """
        x = x.long()
        # The encoder still consumes a flattened one-hot view of the sequence.
        enc_in = F.one_hot(x, num_classes=self.n_words).float().flatten(1)
        mean, logvar = self.encode(enc_in)
        z = self.reparameterization(mean, logvar)
        logits = self.decode(z, x)  # teacher forcing on the ground-truth tokens
        if get_z:
            return logits, mean, logvar, z
        return logits, mean, logvar

    @staticmethod
    def convert_action_list_to_one_hot_tensor(action_list: List[int]) -> torch.Tensor:
        seq_ten = torch.Tensor(action_list)
        one_hot = torch.FloatTensor(10, 5).zero_()
        one_hot.scatter_(-1, seq_ten.to(torch.int64).unsqueeze(-1), 1)
        return one_hot

    def get_reconstructed_action_list_with_embedding(self, action_l: List[int]):
        # Feed integer tokens directly; the Transformer decoder is teacher-forced on them.
        tokens = torch.tensor(action_l, dtype=torch.long, device=self.device).unsqueeze(0)  # (1, T)
        logits, _, _, embedding = self.forward(x=tokens, get_z=True)
        # logits[:, t] predicts token t (the z-prefix supplies the +1 shift), so the first
        # T logit positions correspond to the T reconstructed actions.
        reconstructed_action_seq = logits[:, :tokens.size(1), :].argmax(-1).squeeze(0)
        return reconstructed_action_seq, embedding


    @staticmethod
    @TensorCache()
    def trim_action_sequence_from_eos_tokens(actions: torch.Tensor, eos_token: int = 4) -> tuple[torch.Tensor, bool]:
        # find the index where eos_token token appears in the tensor
        eos_token_idx_tensor = (actions == eos_token).nonzero()

        # handle cases where eos_token is the first element, replace it with the subsequent number
        if eos_token_idx_tensor.numel() > 0 and eos_token_idx_tensor[0] == 0:
            subsequent_number = actions[eos_token_idx_tensor[0] + 1]
            actions[eos_token_idx_tensor[0]] = subsequent_number

        # trim the action sequence until the EOS token
        eos_token_idx_tensor = (actions == eos_token).nonzero(as_tuple=True)[0]
        if len(eos_token_idx_tensor) != 0:
            actions = actions[:eos_token_idx_tensor[0]]

        if len(actions) == 0:
            print("invalid - empty action sequence")
            actions = torch.tensor([1])
            return actions, True

        return actions, False


class MazeDataLoaderV2:
    def __init__(self, f_names_list: List[str], augment_more_data: bool = False, use_one_hot: bool = True, padding_val: int = None,
                 ae_input_size: int = 10, merge_all_into_train: bool = False, return_token_indices: bool = True):
        self.f_name_l = f_names_list
        self.bs = 1
        self.use_one_hot = use_one_hot
        # Token-index mode emits a single integer class id per step (for the Transformer's
        # nn.Embedding / CrossEntropyLoss); otherwise we keep the legacy one-hot/scalar width.
        self.return_token_indices = return_token_indices
        self.input_size = 1 if return_token_indices else (5 if use_one_hot else 1)
        self.n_actions = N_ACTIONS
        self.padding = padding_val if padding_val else 4
        self.ae_input_size = ae_input_size
        self.matching_action_mapping = {0: 1, 1: 0, 2: 3, 3: 2}
        self.augment_more_data = augment_more_data
        self.merge_all_into_train = merge_all_into_train

    def convert_action(self, action: int) -> Union[int, float, np.array]:
        if self.return_token_indices:
            # Integer class index (0..n_actions) consumed by nn.Embedding / CrossEntropyLoss.
            return int(action)
        if self.use_one_hot:
            coding = np.zeros(self.n_actions + 1)
            coding[action] = 1.0
        else:
            coding = -1.0
            coding += 0.5 * action
        return coding

    def pad(self, epi_list: list, list_new_size: int) -> list:
        if len(epi_list) != list_new_size:
            epi_list.extend([self.convert_action(self.padding)] * (list_new_size - len(epi_list)))
        return epi_list

    def prepare_list(self, df: pd.DataFrame, cat: str, col: str = 'epi_with_padding', cutoff: Union[int, list] = None,
                     n_samples: int = 2) -> list:
        if (cat == 'train') or (cat == 'validation'):
            chunk_size = self.bs
        elif cat == 'test':
            chunk_size = 1
        else:
            raise Exception("ERROR - wrong category. data category should be train/validation/test")
        series = df[col]
        chunks = list()
        num_chunks = len(series) // chunk_size + (1 if len(series) % chunk_size else 0)
        for i in range(num_chunks):
            chunk = series[i * chunk_size:(i + 1) * chunk_size].tolist()
            if cutoff:
                if isinstance(cutoff, list):
                    chosen_cut = np.random.choice(cutoff)
                else:
                    chosen_cut = cutoff
                initials = [[(random.randint(0, max(len(c) - chosen_cut, 0))) for b in range(n_samples)] for c in chunk]
                chunk = [[c[initial: initial + chosen_cut] for initial in initials[idx]] for idx, c in enumerate(chunk)]
            chunk = [self.convert_action(act) for act in chunk[0]]
            chunk = self.pad(chunk, self.ae_input_size)
            if self.return_token_indices:
                # (seq_len,) integer tokens; stacking a batch yields (B, seq_len).
                chunk_torch = torch.tensor(chunk, dtype=torch.long)
            else:
                chunk_arr = np.array(chunk)
                chunk_torch = torch.from_numpy(chunk_arr).type(torch.FloatTensor)
            chunks.append(chunk_torch)
        return chunks

    @staticmethod
    def split_into_train_test_val(df: pd.DataFrame, bin_col_name: str, train_ratio: Union[float, int] = 0.7,
                                  val_ratio: Union[float, int] = 0.15
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        :param df:
        :param bin_col_name: should be 'bins' if we want fixed length sequences or 'max_len' if we want different
        length sequences
        :param train_ratio:
        :param val_ratio:
        :return:
        """
        df_train, df_val, df_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        for b_size in df[bin_col_name].unique().tolist():
            slice_df = df[df[bin_col_name] == b_size]
            slice_df = slice_df.sample(frac=1).reset_index(drop=True)
            if train_ratio < 1:
                df_len = slice_df.shape[0]
                temp_slide_df_train = slice_df.iloc[:int(df_len * train_ratio), :]
                temp_slide_df_val = slice_df.iloc[int(train_ratio * df_len):int(df_len * (train_ratio + val_ratio)), :]
                temp_slide_df_test = slice_df.iloc[int((train_ratio + val_ratio) * df_len):, :]
            else:
                temp_slide_df_train = slice_df.iloc[:train_ratio, :]
                temp_slide_df_val = slice_df.iloc[train_ratio:train_ratio + val_ratio, :]
                temp_slide_df_test = slice_df.iloc[train_ratio + val_ratio:, :]

            if df_train.empty:
                df_train = temp_slide_df_train
            else:
                df_train = pd.concat([df_train, temp_slide_df_train], ignore_index=True)

            if df_val.empty:
                df_val = temp_slide_df_val
            else:
                df_val = pd.concat([df_val, temp_slide_df_val], ignore_index=True)

            if df_test.empty:
                df_test = temp_slide_df_test
            else:
                df_test = pd.concat([df_test, temp_slide_df_test], ignore_index=True)

        return df_train, df_val, df_test

    def get_train_val_test_data_eos_version(self, reserve_two_words: bool = False):
        episodes_l = []
        for f_name in self.f_name_l:
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'dense_autoencoder/db', f_name), 'rb'
                      ) as fp:
                cur_episode_l = pickle.load(fp)
                episodes_l.extend(cur_episode_l)
        sorted_episode_l = sorted(episodes_l, key=lambda x: len(x), reverse=True)
        # remove duplicates:
        sorted_episode_l.sort()
        sorted_episode_l = list(epi for epi, _ in itertools.groupby(sorted_episode_l))
        if isinstance(sorted_episode_l[0][0], tuple):  # True == we have (state, action) and we want action
            sorted_episode_l = [[a[1] + 2 * reserve_two_words for a in b] for b in
                                sorted_episode_l]  # if reserve_two_words == True then all words are shifted by 2 (0 --> 2, 1--> 3..)
        else:
            sorted_episode_l = [[a + 2 * reserve_two_words for a in b] for b in sorted_episode_l]

        episodes_df = pd.DataFrame({'episode': sorted_episode_l, 'len': [len(a) for a in sorted_episode_l]})
        episodes_df = episodes_df[(episodes_df.len >= 1) & (episodes_df.len <= 10)]

        col_name = 'len'
        train_df, val_df, test_df = self.split_into_train_test_val(df=episodes_df, bin_col_name=col_name)
        if self.merge_all_into_train:
            train_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

        if self.augment_more_data:
            ten_sequence_size = train_df[train_df.len == self.ae_input_size].shape[0]
            for action_seq_length in range(1, self.ae_input_size):
                n_missing_seq = ten_sequence_size - train_df[train_df.len == action_seq_length].shape[0]
                n_times_multiple = (n_missing_seq // train_df[train_df.len == action_seq_length].shape[0])
                if n_times_multiple == 0:
                    sampled_data_df = train_df[train_df.len == action_seq_length].sample(n=n_missing_seq)
                else:
                    sampled_data_df = pd.concat([train_df[train_df.len == action_seq_length]] * n_times_multiple)
                    sampled_data_df = pd.concat([sampled_data_df, train_df[train_df.len == action_seq_length].sample(
                        n=n_missing_seq - sampled_data_df.shape[0])])
                if action_seq_length == 1:  # first index
                    total_sample_df = sampled_data_df
                else:
                    total_sample_df = pd.concat([sampled_data_df, total_sample_df], ignore_index=True)
            train_df = pd.concat([train_df, total_sample_df], ignore_index=True)

        train_torch = self.prepare_list(train_df, cat='train', n_samples=1, col='episode')
        val_torch = self.prepare_list(val_df, cat='validation', n_samples=1, col='episode')
        test_torch = self.prepare_list(test_df, cat='test', col='episode')

        if self.augment_more_data and not self.return_token_indices:
            # This soft-label jitter (writing floats into the one-hot tensor) only makes
            # sense for the one-hot representation, not for integer token indices.
            updated_train_torch = []
            for action_seq, action_seq_len in zip(train_torch, train_df.len):
                if action_seq_len <= 5:  # augment data for short sequences only
                    if random.randint(0, self.ae_input_size) > 3:
                        # Get the indices where the tensor is equal to 1
                        indices = (action_seq == 1).nonzero(as_tuple=False)
                        # Update the first k indices with random floats between 0.9 and 1.0
                        for action_seq_length in range(min(action_seq_len, len(indices))):
                            action_seq[indices[action_seq_length][0], indices[action_seq_length][1]] = random.uniform(0.9, 1.0)
                updated_train_torch.append(action_seq)
            train_torch = updated_train_torch
        if self.merge_all_into_train:
            return train_torch, None, None, torch.Tensor(train_df.len), None, None
        return train_torch, val_torch, test_torch, torch.Tensor(train_df.len), torch.Tensor(val_df.len), torch.Tensor(test_df.len)

    def inverse_transform_sequence(self, sequence: torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        # unflatten seq:
        squeezed = sequence.reshape(-1, self.input_size)
        act_list = squeezed.argmax(1)
        return squeezed, act_list

    @staticmethod
    def transform_sequence(sequence: torch.tensor) -> torch.tensor:
        return torch.nn.functional.one_hot(sequence, num_classes=5)


def get_model_performance_on_set(seq_set, eval_model, c_mzl, seq_set_len, extract_embedding: bool = False):
    n_errors = 0
    n_diff_elements_l = []
    total_loss = 0
    embedding_map_l = []
    for seq, seq_len in zip(seq_set, seq_set_len):
        fl_seq = seq.flatten()
        with torch.no_grad():
            if extract_embedding:
                recon_seq, mean_seq, log_var_seq, _, embedding = eval_model(fl_seq.unsqueeze(0), True)
            else:
                recon_seq, mean_seq, log_var_seq, _ = eval_model(fl_seq.unsqueeze(0), False)
        recon_one_hot, recon_act_seq = c_mzl.inverse_transform_sequence(recon_seq)
        _, act_seq = c_mzl.inverse_transform_sequence(seq)
        if extract_embedding:
            embedding_map_l.append((act_seq.tolist(), embedding))
        n_diff_elements = int((True ^ torch.eq(recon_act_seq, act_seq)).sum())  # number of different actions in sequence
        n_diff_elements_l.append(n_diff_elements)
        if n_diff_elements > 0:
            n_errors += 1
        seq_loss = loss_function(fl_seq.unsqueeze(0), recon_seq, mean_seq, log_var_seq, desired_var=VAR, x_seq_len=seq_len)
        total_loss += seq_loss
    if extract_embedding:
        return total_loss / len(seq_set), n_errors, np.mean(n_diff_elements_l), embedding_map_l
    return total_loss/len(seq_set), n_errors, np.mean(n_diff_elements_l)#, n_diff_elements_l


def combine_lists_iteratively(list1, list2, k):
    combined_list = [
        element
        for i in range(0, max(len(list1), len(list2)), k)
        for element in list1[i:i+k] + list2[i:i+k]
    ]
    return combined_list


def get_noise_per_cat(noise_cat: str, actor_n_channels: int) -> torch.Tensor:
    if noise_cat == 'normal_0.9':
        return torch.FloatTensor(actor_n_channels).normal_(mean=0.9, std=1)
    if noise_cat == 'normal_-0.9':
        return torch.FloatTensor(actor_n_channels).normal_(mean=-0.9, std=1)
    if noise_cat == 'normal_0-3':
        return torch.FloatTensor(actor_n_channels).normal_(mean=0, std=3)
    if noise_cat == 'normal_0':
        return torch.FloatTensor(actor_n_channels).normal_(mean=0, std=1)
    if noise_cat == 'uniform':
        return torch.FloatTensor(actor_n_channels).uniform_(-1, 1)
    if noise_cat == 'bernoulli':
        return torch.FloatTensor(actor_n_channels).bernoulli_(0.5)
    if noise_cat == 'exponential':
        return torch.FloatTensor(actor_n_channels).exponential_(lambd=1.0)
    if noise_cat == 'laplace':
        return Laplace(0, 1).sample([actor_n_channels])


def evaluate_k_seq_in_decoder(decoder, action_db_path: str, n_samples: int = 10000, decoder_input_size: int = 16, compress_prints: bool = False,
                              noise_cat_list: List[str] = DEFAULT_NOISE_LIST) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        action_d = pd.read_pickle(action_db_path, compression='gzip')
        for noise_cat in noise_cat_list:
            action_seq_l = []
            action_seq_d = defaultdict(list)
            coverage_dict = defaultdict(float)
            print(f'============ {noise_cat} ============')
            for i in range(n_samples):
                rand_noise = get_noise_per_cat(noise_cat=noise_cat, actor_n_channels=decoder_input_size).to(device)
                if isinstance(decoder, DenseVAE):  # during decoder training
                    rand_action_seq = decoder.decoder(rand_noise.unsqueeze(0)).reshape(-1, 10, 5).argmax(-1)
                    action_sequence, _ = decoder.trim_action_sequence_from_eos_tokens(rand_action_seq.squeeze(0))
                else:  # outside of training - decoder is wrapped in a ActorGen tool
                    rand_action_seq = decoder.gen_action_seq(rand_noise.unsqueeze(0))[1].argmax(-1)
                    action_sequence, _ = decoder.model.trim_action_sequence_from_eos_tokens(rand_action_seq[0])
                action_seq_l.append(action_sequence)
                action_seq_len = action_sequence.size(0)
                action_seq_d[action_seq_len].append(str(action_sequence.tolist()))
            action_seq_size_l = [len(a) for a in action_seq_l]

            for action_seq_len, action_seq_list in action_d.items():
                curr_set = set(action_seq_d[action_seq_len])
                counter = 0
                for action_seq in action_seq_list:
                    if action_seq in curr_set:
                        counter += 1
                coverage = counter/len(action_seq_list)
                coverage_dict[action_seq_len] = np.round(coverage, 3)
                if not compress_prints:
                    print(f"size: {action_seq_len}, coverage: {coverage}")
            if compress_prints:
                print(f"valid coverage: {coverage_dict}")
            print(f"total length distribution: {[(k, np.round(v/n_samples, 3)) for (k,v) in sorted(Counter(action_seq_size_l).items())]}")


def loss_function(logits, targets, mean, log_var, desired_var, label_smoothing=0.1):
    """Autoregressive reconstruction (cross-entropy) + KLD for the Transformer VAE.

    Args:
        logits: decoder output of shape (B, T + 1, vocab). Position 0 is produced from the
            z-prefix token; positions 1..T are produced from the teacher-forced tokens.
        targets: ground-truth integer tokens of shape (B, T).
        mean, log_var: latent posterior parameters, shape (B, latent_dim).
        desired_var: target prior variance for the KLD term.
        label_smoothing: cross-entropy label smoothing.

    The +1 autoregressive shift is supplied by the z-prefix sitting at position 0: the
    logit at position t predicts token t, so the first T logit positions are aligned with
    all T targets (equivalently, feeding [z, x_0..x_{T-1}] predicts [x_0..x_{T-1}]).
    """
    seq_len = targets.size(1)
    shifted_logits = logits[:, :seq_len, :]  # (B, T, vocab)
    reproduction_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.size(-1)),
        targets.reshape(-1),
        label_smoothing=label_smoothing,
        reduction='sum',
    )
    KLD = -0.5 * torch.sum(1 + log_var - (1 / desired_var) * (mean.pow(2) + log_var.exp()) - np.log(desired_var))
    return reproduction_loss + KLD


if __name__ == '__main__':

    f_names = [
        'action_seq_db_1_to_10_2_act_in_seq_v1.pickle',
    ]
    merge_all_into_train = True
    mzl = MazeDataLoaderV2(f_names_list=f_names, augment_more_data=False, merge_all_into_train=merge_all_into_train)
    train_iter, _, _, train_seq_len, _, _ = mzl.get_train_val_test_data_eos_version(reserve_two_words=False)
    # Model Initialization
    model = DenseVAE(input_length=10, n_words=5, variance_for_sample=VAR, device=torch.device("cpu"))

    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    epochs = 20000

    batch_size = 32
    outputs = []
    losses = []
    batch_losses = []
    train_data_l = []
    val_data_l = []
    train_len_tuple_iter = [(seq, seq_len) for seq, seq_len in zip(train_iter, train_seq_len)]
    short_train_len_tuple_iter = [(seq, seq_len) for seq, seq_len in train_len_tuple_iter if seq_len <= 5]
    long_train_len_tuple_iter = [(seq, seq_len) for seq, seq_len in train_len_tuple_iter if seq_len > 5]
    for epoch in range(epochs):
        overall_loss = 0
        random.Random(4).shuffle(short_train_len_tuple_iter)
        random.Random(4).shuffle(long_train_len_tuple_iter)
        short_long_train_len_tuple_iter = combine_lists_iteratively(short_train_len_tuple_iter, long_train_len_tuple_iter, batch_size // 2)
        batch_idx = 0
        batch_loss = 0

        c_train_iter = [a[0] for a in short_long_train_len_tuple_iter]
        c_train_seq_len = [a[1] for a in short_long_train_len_tuple_iter]
        for i in range(0, len(c_train_iter), batch_size):
            batch_idx += 1
            optimizer.zero_grad()
            image_batch_l = train_iter[i:i+batch_size]
            image_len_batch_l = c_train_seq_len[i:i + batch_size]
            image_batch_tensor = torch.stack(image_batch_l)
            flatten_batch = image_batch_tensor.flatten(1)

            reconstructed_batch, mean, log_var, _ = model(flatten_batch)

            loss = loss_function(flatten_batch, reconstructed_batch, mean, log_var, image_len_batch_l,
                                 desired_var=VAR)

            overall_loss += loss.item()

            # Storing the losses in a list for plotting
            losses.append(loss)
            loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            model.eval()
            print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))

            train_loss, train_n_errors, train_n_errors_in_set = get_model_performance_on_set(seq_set=train_iter, eval_model=model, c_mzl=mzl,
                                                                                             seq_set_len=train_seq_len)
            print(
                f'epoch: {epoch}/{epochs}. loss in train set: {train_loss}, # wrong sequences: {train_n_errors}, avg errors in seq: {train_n_errors_in_set}')
            train_data_l.append((train_loss, train_n_errors, train_n_errors_in_set))

            if epoch % 200 == 0:
                evaluate_k_seq_in_decoder(decoder=model, action_db_path='db/action_sequence_dict_by_sequence_length_entire_db.pkl.gzip',
                                          n_samples=10000,
                                          decoder_input_size=model.decoder_input_size, compress_prints=True)
            model.train()
        # if epoch % 1000 == 0:
            # torch.save(model.state_dict(), f'middle_denseAE_generic_2act_seq_VAE_v7_bs={batch_size}_epochs={epoch}-{epochs}_lr={lr}_end_size_16_var_{VAR}_leakyrelu_normalized_by_seq_len_without_aug_instance_norm.pt')
        # if epoch >= epochs-1:
            # torch.save(model.state_dict(), f'denseAE_generic_2act_seq_VAE_v7_bs={batch_size}_epochs={epoch}-{epochs}_lr={lr}_end_size_16_var_{VAR}_leakyrelu_normalized_by_seq_len_without_aug_instance_norm.pt')




