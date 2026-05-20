from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, List
import re
import torch
import transformers
from torch.utils.data import Dataset
from transformers import Seq2SeqTrainer
import torch.nn.functional as F
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TRANSFORMERS_NO_FLAX"] = "1"  # 可选

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

ACTION_LIST = ["ETWT", "NTST", "ELWL", "NLSL"]
_ACTION_PATTERN = re.compile(r"<signal>(.*?)</signal>")


def _extract_action(response_text: str) -> int:
    """Extract action index from response text by parsing <signal> tags."""
    if not response_text:
        return 0
    matches = _ACTION_PATTERN.findall(response_text)
    if not matches:
        return 0
    signal = matches[-1].strip()
    try:
        return ACTION_LIST.index(signal)
    except ValueError:
        return 0


class ScoreDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data):
        super(ScoreDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return dict(input_ids=self.data[i])

def _single_tokenize(text, tokenizer):
    toked = tokenizer(
            text,
            return_tensors="pt"
        )

    return toked['input_ids'][0]

@dataclass
@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    beta: float = field(default=0.15)
    length_penalty: float = field(default=1)
    gap: float = field(default=0)
    seed: int = field(
        default=1
    )

    temperature: float = field(
        default=1
    )

    # --- Stage 2: action-aware loss hyperparameters ---
    action_balance_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for action-balance entropy regularizer (0=off)."},
    )
    global_action_weights: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated per-action weights for ETWT,NTST,ELWL,NLSL, e.g. '0.5,1.0,1.0,2.0'."
        },
    )
    overconf_penalty_weight: float = field(
        default=0.0,
        metadata={"help": "Weight for over-confidence penalty on score distribution (0=off)."},
    )


@dataclass
class DataCollatorForSupervisedDatasetMix(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        idxs = []
        all_scores = []
        input_ids = []
        labels = []
        action_labels = []
        for idx, ins in enumerate(instances):
            ins = ins['input_ids']
            query = ins['query']
            response = ins['response']
            score = ins['score']
            all_scores.append(score)
            idxs.append(idx)

            query_input_ids = _single_tokenize(query, self.tokenizer)
            query_target = torch.LongTensor([IGNORE_INDEX] * query_input_ids.shape[0])

            res_input_ids = _single_tokenize(response + self.tokenizer.eos_token, self.tokenizer)  # eos here
            input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
            labels.append(torch.cat((query_target, res_input_ids), dim=0))

            action_labels.append(_extract_action(response))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
            action_labels=torch.LongTensor(action_labels),
        )


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        idxs = []
        all_scores = []
        input_ids = []
        labels = []
        action_labels = []
        for idx, ins in enumerate(instances):
            ins = ins['input_ids']
            query = ins['query']
            responses = ins['responses']
            scores = ins['scores']
            all_scores.append(scores)
            idxs.append([idx] * len(scores))

            query_input_ids = _single_tokenize(query, self.tokenizer)
            query_target = torch.LongTensor([IGNORE_INDEX] * query_input_ids.shape[0])
            for res in responses:
                res_input_ids = _single_tokenize(res + self.tokenizer.eos_token, self.tokenizer)  # eos here
                input_ids.append(torch.cat((query_input_ids, res_input_ids), dim=0))
                labels.append(torch.cat((query_target, res_input_ids), dim=0))

                action_labels.append(_extract_action(res))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            labels=labels,
            idxs=torch.LongTensor(idxs),
            scores=torch.FloatTensor(all_scores),
            action_labels=torch.LongTensor(action_labels),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, train_data, test_data, mix=False) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = ScoreDataset(data=train_data)
    test_dataset = ScoreDataset(data=test_data)
    if mix:
        data_collator = DataCollatorForSupervisedDatasetMix(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return dict(train_dataset=train_dataset, eval_dataset=test_dataset, data_collator=data_collator)


local_rank = None
class RankTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._global_action_weights_tensor: Optional[torch.Tensor] = None
        self._resolve_global_action_weights()

    def _resolve_global_action_weights(self):
        raw = getattr(self.args, "global_action_weights", None)
        if raw is None or not str(raw).strip():
            self._global_action_weights_tensor = None
            return
        parts = [p.strip() for p in str(raw).split(",")]
        if len(parts) != len(ACTION_LIST):
            raise ValueError(
                f"global_action_weights must have {len(ACTION_LIST)} values, got {len(parts)}"
            )
        values = [float(p) for p in parts]
        self._global_action_weights_tensor = torch.tensor(values, dtype=torch.float32)

    def gather_logits_labels(self, logits, labels):
        mask = (labels != -100).float()
        new_logits = logits.clone()  # B x S x hidsize
        new_labels = labels.clone()
        new_labels[labels == -100] = 0  # B x S
        output = torch.gather(new_logits, dim=-1, index=new_labels.unsqueeze(-1)).squeeze(-1)
        output = output * mask  # B * L
        return output

    def get_score(self, logit_label, labels):
        mask = (labels != -100).float()
        length = mask.sum(-1)
        scores = logit_label.sum(-1) / (length ** self.args.length_penalty)
        return scores

    def aft_rank_loss(self, scores, idxs, rw_scores, action_labels=None):
        """
        Ranking loss with optional per-action equal weighting.

        When action_labels is provided, pairwise loss terms are grouped by the
        ground-truth action of each positive sample.  Each action contributes
        equally to the final loss regardless of its sample count in the batch.
        """
        num_actions = len(ACTION_LIST)

        sorted_indexs = rw_scores.argsort(dim=-1, descending=True)
        sorted_rw_scores = rw_scores[sorted_indexs]
        sorted_scores = scores[sorted_indexs]
        sorted_actions = action_labels[sorted_indexs] if action_labels is not None else None
        sorted_idxs = idxs[sorted_indexs]

        positive_lower_boundary = 10000

        # Accumulators
        if sorted_actions is not None:
            per_action_loss_terms: List[List[torch.Tensor]] = [[] for _ in range(num_actions)]
        else:
            global_tmp = 0.0

        for i in range(0, len(sorted_rw_scores) - 1):
            pos_scores = sorted_scores[i]
            pos_rw_score = sorted_rw_scores[i].item()
            neg_start = None

            positive_lower_boundary = min(pos_scores.item(), positive_lower_boundary)
            for kk in range(i + 1, len(sorted_rw_scores)):
                if sorted_rw_scores[kk].item() < pos_rw_score:
                    neg_start = kk
                    break
            if neg_start is None:
                continue

            pair_loss = 0.0
            for neg_score in sorted_scores[neg_start:]:
                pair_loss += torch.exp((neg_score - pos_scores) / self.args.temperature)
                pair_loss += torch.exp(
                    2 * positive_lower_boundary - 2 * self.args.beta - pos_scores.item() - neg_score
                )

            if sorted_actions is not None:
                pos_action = sorted_actions[i].item()
                pos_action = max(0, min(pos_action, num_actions - 1))
                per_action_loss_terms[pos_action].append(pair_loss)
            else:
                global_tmp += pair_loss

        if sorted_actions is not None:
            # Per-action equal weighting: average per-sample loss within each action,
            # then average across actions (each action contributes equally).
            total_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
            valid_actions = 0
            for a in range(num_actions):
                if not per_action_loss_terms[a]:
                    continue
                w = (
                    self._global_action_weights_tensor[a].to(scores.device)
                    if self._global_action_weights_tensor is not None
                    else 1.0
                )
                # Mean of per-positive-sample losses for this action
                per_sample_losses = [
                    torch.log(1.0 + term) for term in per_action_loss_terms[a]
                ]
                action_mean = torch.stack(per_sample_losses).mean()
                total_loss = total_loss + w * action_mean
                valid_actions += 1

            if valid_actions == 0:
                return torch.tensor(0.0, device=scores.device, requires_grad=True)
            return total_loss / max(1, valid_actions)
        else:
            if global_tmp != 0:
                loss = torch.log(1 + global_tmp)
            else:
                loss = torch.tensor(0.0, device=scores.device, requires_grad=True)
            return loss

    def _action_balance_regularizer(self, scores, action_labels, idxs):
        """
        Entropy-based regularizer on the per-action distribution of top-ranked
        predictions.  Penalises any action dominating the batch's predictions.
        """
        num_actions = len(ACTION_LIST)
        device = scores.device

        unique_queries = idxs.unique()
        pred_counts = torch.zeros(num_actions, device=device)

        for qid in unique_queries:
            q_mask = (idxs == qid)
            if q_mask.sum() == 0:
                continue
            q_scores = scores[q_mask]
            q_actions = action_labels[q_mask]
            best_local = q_scores.argmax().item()
            best_action = q_actions[best_local].item()
            pred_counts[best_action] += 1

        total = pred_counts.sum()
        if total < 2:
            return torch.tensor(0.0, device=device)

        probs = pred_counts / total
        eps = 1e-8
        entropy = -(probs * torch.log(probs + eps)).sum()
        max_entropy = torch.log(
            torch.tensor(num_actions, device=device, dtype=torch.float32)
        )
        return max_entropy - entropy  # 0 when uniform, >0 when imbalanced

    def _overconf_penalty(self, scores, action_labels, idxs):
        """
        Penalise when any action's mean predicted score is significantly above
        the overall mean — a signal of systematic over-confidence.
        """
        num_actions = len(ACTION_LIST)
        device = scores.device

        per_action_sum = torch.zeros(num_actions, device=device)
        per_action_count = torch.zeros(num_actions, device=device)

        for i in range(len(scores)):
            a = action_labels[i].item()
            a = max(0, min(a, num_actions - 1))
            per_action_sum[a] += scores[i]
            per_action_count[a] += 1

        mask = per_action_count > 0
        if mask.sum() < 2:
            return torch.tensor(0.0, device=device)

        per_action_mean = per_action_sum[mask] / per_action_count[mask]
        overall_mean = per_action_mean.mean()
        excess = torch.relu(per_action_mean - overall_mean)
        return excess.sum()

    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(input_ids=inputs.get('input_ids'), attention_mask=inputs.get('attention_mask'))[0]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = inputs.get("labels")[..., 1:].contiguous()
        logits = F.log_softmax(shift_logits, dim=-1)
        logit_label = self.gather_logits_labels(logits, shift_labels)
        scores = self.get_score(logit_label, shift_labels)

        action_labels = inputs.get("action_labels")
        # Flatten rw_scores and idxs: the non-mix collator produces [B, R] tensors
        # while model scores and action_labels are flat [B*R].
        rw_scores = inputs.get("scores")
        if rw_scores.dim() > 1:
            rw_scores = rw_scores.reshape(-1)
        else:
            rw_scores = rw_scores.squeeze()
        idxs = inputs.get("idxs")
        if idxs.dim() > 1:
            idxs = idxs.reshape(-1)

        # 1. Core ranking loss (action-aware with per-action equal weighting)
        alignment_loss = self.aft_rank_loss(
            scores, idxs, rw_scores, action_labels
        )
        loss = alignment_loss

        # 2. Action-balance entropy regularizer (on predicted actions)
        if self.args.action_balance_weight > 0 and action_labels is not None:
            loss = loss + self.args.action_balance_weight * self._action_balance_regularizer(
                scores, action_labels, idxs
            )

        # 3. Over-confidence penalty (on score distribution)
        if self.args.overconf_penalty_weight > 0 and action_labels is not None:
            loss = loss + self.args.overconf_penalty_weight * self._overconf_penalty(
                scores, action_labels, idxs
            )

        return (loss, scores) if return_outputs else loss
