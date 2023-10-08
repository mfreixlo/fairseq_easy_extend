
from IPython.core.async_helpers import _AsyncSyntaxErrorVisitor
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass

from dataclasses import dataclass, field
from fairseq.logging import metrics
from fairseq import utils
import math
import sacremoses

from sacrebleu.metrics import CHRF
import sacrebleu
from sacrebleu import corpus_bleu as _corpus_bleu

import torch.nn.functional as F
import torch

import wandb

def sentence_bleu(hypothesis, reference):
  bleu = _corpus_bleu(hypothesis, reference)
  for i in range(1, 4):
      bleu.counts[i] += 1
      bleu.totals[i] += 1
  bleu = sacrebleu.BLEU.compute_bleu(
      bleu.counts,
      bleu.totals,
      bleu.sys_len,
      bleu.ref_len,
      smooth_method="exp",
  )
  return bleu.score


def eval_metric(metric, hyps, ref):
    if metric == "bleu":
      score = sentence_bleu(hyps, [ref])
      # We want the Bleu score to be in [0, 1] and sentence_bleu is in [0, 100]
      score /= 100
      # score = BLEU().corpus_score(hyps, [ref]).score
    elif metric == "chrf":
      score = CHRF().corpus_score(hyps, [ref]).score

    return score


def detokenize(sentence):
    detok = sacremoses.MosesDetokenizer()

    return(detok.detokenize(sentence.strip().split(" "))
            .replace(" @", "")
            .replace("@ ", "")
            .replace(" =", "=")
            .replace("= ", "=")
            .replace(" – ", "–")
        )


@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu",
                                       metadata={"help": "sentence level metric"})


@register_criterion("custom_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
  def __init__(self, task, sentence_level_metric):
    super().__init__(task)
    self.metric = sentence_level_metric
    self.tgt_dict = task.tgt_dict

  def forward(self, model, sample, reduce=True):
    """Compute the loss for the given sample.
    Returns a tuple with three elements:
    1) the loss
    2) the sample size, which is used as the denominator for the gradient
    3) logging outputs to display while training
    """
    nsentences, ntokens = sample["nsentences"], sample["ntokens"]

    # B x T
    src_tokens, src_lengths = (
        sample["net_input"]["src_tokens"],
        sample["net_input"]["src_lengths"],
    )
    tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]


    outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
    #get loss only on tokens, not on lengths
    outs = outputs["word_ins"].get("out", None)
    masks = outputs["word_ins"].get("mask", None)

    loss, repetition = self._compute_loss(outs, tgt_tokens, masks)
    
    # NOTE:
    # we don't need to use sample_size as denominator for the gradient
    # here sample_size is just used for logging
    sample_size = 1
    logging_output = {
        "loss": loss.detach(),
        "ntokens": ntokens,
        "nsentences": nsentences,
        "sample_size": sample_size,
        "repetition": repetition
    }

    return loss, sample_size, logging_output


  def _compute_repetition(self, output_sentence, target_sentence):
    
    rep_out = 0
    prev_word = output_sentence[0]
    for word in output_sentence[1:]:
      if prev_word == word:
        rep_out += 1
      prev_word = word

    rep_target = 0
    prev_word = target_sentence[0]
    for word in target_sentence[1:]:
      if prev_word == word:
        rep_target += 1
      prev_word = word
    
    repetition = rep_out - rep_target

    return repetition



  def _compute_loss(self, outputs, targets, masks=None):
    """
    outputs: batch x len x d_model
    targets: batch x len
    masks:   batch x len
    """
    
    batch_size, seq_len, vocab_size = outputs.shape
    device = outputs.get_device()

    with torch.no_grad():
      logits = F.softmax(outputs, dim=-1).view(-1, vocab_size)

      sample_idx = torch.multinomial(logits, 1, \
        replacement=True).view(batch_size, seq_len)


    sampled_sentence_string = self.tgt_dict.string(sample_idx, bpe_symbol="@@")
    target_sentence_string = self.tgt_dict.string(targets, bpe_symbol="@@")

    repetition = []
    reward = []
    with torch.no_grad():
      for sampled_sentence, target_sentence in zip(sampled_sentence_string.split('\n'), target_sentence_string.split('\n')):
        target_sentence = target_sentence.replace(' <pad>', '')
        sampled_sentence = detokenize(sampled_sentence)
        target_sentence = detokenize(target_sentence)
        reward.append(eval_metric(self.metric, sampled_sentence.split(' '), target_sentence.split(' ')))
        repetition.append(self._compute_repetition(sampled_sentence.split(' '), target_sentence.split(' ')))
      repetition = torch.Tensor(repetition).mean()

    reward = torch.Tensor(reward).unsqueeze(dim=-1)
    reward = reward.repeat(1, seq_len)
    reward = reward.to(device)
    #reward = 1 - reward.to(device)

    #padding mask, do not remove
    if masks is not None:
      outputs = outputs[masks]
      reward, sample_idx = reward[masks], sample_idx[masks]

    wandb.log({"reward": reward.mean()})

    log_probs = F.log_softmax(outputs, dim = -1)
    log_probs_of_samples = torch.gather(log_probs, 1, sample_idx.unsqueeze(dim=-1))
    log_probs_of_samples = log_probs_of_samples.squeeze()

    loss = - log_probs_of_samples * reward
    # loss = - log_probs_of_samples * reward * repetition

    loss = loss.mean()

    return loss, repetition

  @staticmethod
  def reduce_metrics(logging_outputs) -> None:
    """Aggregate logging outputs from data parallel training."""
    sample_size =  utils.item(sum(log.get("sample_size", 0) for log in logging_outputs))
    loss_sum = utils.item(sum(log.get("loss", 0) for log in logging_outputs))

    repetition = [log.get("repetition", 0) for log in logging_outputs]
    rep_len = len(repetition)

    wandb.log({"repetition": sum(repetition)/rep_len, "loss": loss_sum/ math.log(2)})

    metrics.log_scalar(
        "loss", loss_sum / math.log(2), sample_size, round=3
    )
    metrics.log_scalar("repetition", sum(repetition) / rep_len)



