import torch
from transformers import T5Tokenizer, GPT2Tokenizer, AutoModelForCausalLM, pipeline, AutoTokenizer, T5ForConditionalGeneration
import hashlib
import random
import numpy as np
from tqdm import tqdm
import math
import warnings

class NecessaryAndSufficientWatermark_EncDec():
    def __init__(self, gamma, z=4):
        self.gamma = gamma
        self.z = z

    @torch.no_grad()
    def _initialize(self, model, max_length, num_beams, num_batch, alpha, forced_bos_token_id):
        self.vocab_size = model.config.vocab_size
        self.eos_token_id = model.config.eos_token_id
        self.num_beams = num_beams
        self.device = model.device
        self.num_batch = num_batch
        self.alpha = alpha
        self.max_length = max_length
        
        self.g_max = math.ceil(self.gamma * (max_length-1) + self.z * math.sqrt(self.gamma * (1 - self.gamma) * (max_length-1)))
    
        # table[t][g] denotes the hypothesis and score where the number of wards is t and the number of green words is g.
        self.table = [[None for _ in range(self.g_max+1)] for _ in range(max_length+1)]
        self.score_table = [[-math.inf for _ in range(self.g_max+1)] for _ in range(max_length+1)] 
        self.table[0][0] = torch.LongTensor([[model.config.decoder_start_token_id, forced_bos_token_id] for _ in range(self.num_batch)]).to(self.device)
        self.score_table[0][0] = 0.0

        # sentences such that the last words is EOS.
        self.complete_hypothesis = [[] for _ in range(self.num_batch)]
        self.complete_hypothesis_scores = [[] for _ in range(self.num_batch)]

        self.mask_generator = torch.Generator()


    @torch.no_grad()
    def _range_green(self, t, ref_length):
        tmp = min(1, self.gamma + self.z*math.sqrt(self.gamma*(1-self.gamma)/(ref_length-1))) * (t-1)
        min_g = min(math.ceil(max(0, tmp - self.alpha)), self.g_max)
        max_g = min(math.floor(min(tmp + self.alpha, self.g_max)), t-1)
        return range(min_g, max_g+1)

    
    @torch.no_grad()
    def _print_table(self):
        for i in range(len(self.table)):
            print(self.table[i])

    @torch.no_grad()
    def _generate_masks(self, last_word_id, gamma=0.5):
        self.mask_generator.manual_seed(last_word_id.item() % (2**32 - 1))
        uniform_tensor = torch.randn(self.vocab_size, generator=self.mask_generator)
        green_mask = (uniform_tensor <= torch.sort(uniform_tensor)[0][int(self.vocab_size*self.gamma)-1]).float().to(self.device)
        red_mask = 1.0 - green_mask

        return green_mask, red_mask
            
        
    @torch.no_grad()            
    def _generate_masks_from_hyp(self, hyp, gamma=0.5):
        
        green_masks, red_masks = [], []
        
        for batch_id in range(self.num_batch):
            last_word_id = hyp[batch_id][-1]
            green_mask, red_mask = self._generate_masks(last_word_id)
            green_masks.append(green_mask.view(1, -1))
            red_masks.append(red_mask.view(1, -1))
            
        return torch.cat(green_masks), torch.cat(red_masks)

    
    @torch.no_grad()    
    def _threshold_z(self, num_words):
        # There exists the case where the first generated words is EOS.
        if num_words == 1:
            return math.inf 
        return self.gamma * (num_words-1) + self.z * math.sqrt(self.gamma * (1 - self.gamma) * (num_words-1))

    
    @torch.no_grad()    
    def _update_table(self, t):
        for g in range(min(t, self.g_max)+1):
            if self.table[t][g] is None:
                continue
            
            # the shape is num_batch * (num_beam * vocab_size)
            all_scores = torch.cat(self.score_table[t][g], dim=1)        
            sorted_id = torch.argsort(all_scores, descending=True, dim=-1)
            beams = self.table[t][g]
                        
            new_batch_beams, new_batch_beam_scores = [], []
            
            for batch_id in range(self.num_batch):
                
                new_beams, new_beam_scores = [], []
                
                i = 0
                while len(new_beams) < self.num_beams: 
                
                    word_id = sorted_id[batch_id][None, i, None] % self.vocab_size
                    beam_id = int(sorted_id[batch_id, i] / self.vocab_size)
                    score = all_scores[batch_id, sorted_id[batch_id, i]]
                     
                    generated_sentence = torch.cat([beams[beam_id][None, batch_id], word_id], dim=-1)
                
                    if word_id == self.eos_token_id:
                        if g >= self._threshold_z(t):
                            self.complete_hypothesis[batch_id].append(generated_sentence)
                            self.complete_hypothesis_scores[batch_id].append(score.item())
                    else:
                        new_beams.append(generated_sentence)
                        new_beam_scores.append(score.item())                
                    i += 1

                new_batch_beams.append(new_beams)
                new_batch_beam_scores.append(new_beam_scores)

            self.table[t][g] = torch.cat([torch.cat(hyp).view(self.num_beams, 1, -1) for hyp in new_batch_beams], dim=1)
            self.score_table[t][g] = torch.tensor(new_batch_beam_scores).T.to(self.device) 

            
    @torch.no_grad()
    def _add_table(self, hypothesis, scores, num_words, num_greens):

        if self.table[num_words][num_greens] is None:
            self.table[num_words][num_greens] = [hypothesis]
            self.score_table[num_words][num_greens] = [scores]
        else:
            self.table[num_words][num_greens].append(hypothesis)
            self.score_table[num_words][num_greens].append(scores)

        
    @torch.no_grad()
    def generate(self, model, input_ids, forced_bos_token_id, alpha=math.inf, num_beams=1, max_length=100):
    
        # Current our code only supports the case where the batch size is one. 
        #assert input_ids.shape[0] == 1
        
        self._initialize(model, max_length, num_beams, input_ids.shape[0], alpha, forced_bos_token_id)
        
        if self.alpha == math.inf:
            ref_length = max_length
        else:
            # estimate the length of generated texts.
            output_wo_watermark = model.generate(input_ids, max_length=100, forced_bos_token_id=forced_bos_token_id, num_beams=num_beams, length_penalty=0.0)
            ref_length = output_wo_watermark.shape[1]
            if ref_length == 1:
                warnings.warn("The generated text without watermarks is only single token.")
                ref_length = 2
                
        for t in range(0, max_length):

            if t > 0:                
                for g in self._range_green(t, ref_length):
                    
                    if g < self.g_max:                        
                        for beam_id in range(num_beams):
                            hyp, hyp_score = self.table[t][g][beam_id], self.score_table[t][g][beam_id].view(-1, 1)

                            output = model(input_ids, decoder_input_ids=hyp)                        
                            next_token_probs = torch.softmax(output.logits[:, -1, :], dim=-1)
                            green_mask, red_mask = self._generate_masks_from_hyp(hyp) 
                            
                            # The case where the red word is generated.
                            scores = torch.log(red_mask * next_token_probs) + hyp_score
                            self._add_table(hyp, scores, t+1, g)
                            
                            # The case where the green word is generated.
                            scores = torch.log(green_mask * next_token_probs) + hyp_score
                            self._add_table(hyp, scores, t+1, g+1)
                        
                    else: # g == self.g_max
                        for beam_id in range(num_beams):
                            hyp, hyp_score = self.table[t][g][beam_id], self.score_table[t][g][beam_id].view(-1, 1)

                            output = model(input_ids, decoder_input_ids=hyp)
                            next_token_probs = torch.softmax(output.logits[:, -1, :], dim=-1)
                            
                            # The case where the red/green word is generated.
                            scores = torch.log(next_token_probs) + hyp_score
                            self._add_table(hyp, scores, t+1, g)
                            
            else: # t == 0
                # The number of green words is also 0.
                g = 0 

                hyp, hyp_score = self.table[t][g], self.score_table[t][g]
                output = model(input_ids, decoder_input_ids=hyp)
                
                next_token_probs = torch.softmax(output.logits[:, -1, :], dim=-1)
                scores = torch.log(next_token_probs) + hyp_score
                
                self._add_table(hyp, scores, t+1, g)
                    
            # Select the top-(num_beams) hypothesis.
            self._update_table(t+1)
            
            if self.is_stop(t+1):
                break
            
        results = []
        for batch_id in range(self.num_batch):
            if self.is_stop(t+1):
                scores_of_complete_sentences = [score for score in self.complete_hypothesis_scores[batch_id]]
                best_beam_id = scores_of_complete_sentences.index(max(scores_of_complete_sentences))
                results.append(self.complete_hypothesis[batch_id][best_beam_id].view(-1))
            else:
                scores = [score[batch_id].item() for score in self.score_table[-1][-1]]

                if len(self.complete_hypothesis[batch_id]) == 0:
                    best_beam_id = scores.index(max(scores))
                    results.append(self.table[-1][-1][best_beam_id][batch_id].view(-1))
                else:
                    scores_of_complete_sentences = [score for score in self.complete_hypothesis_scores[batch_id]]

                    if max(scores) > max(scores_of_complete_sentences):
                        best_beam_id = scores.index(max(scores))
                        results.append(self.table[-1][-1][best_beam_id][batch_id].view(-1))
                    else:
                        best_beam_id = scores_of_complete_sentences.index(max(scores_of_complete_sentences))
                        results.append(self.complete_hypothesis[batch_id][best_beam_id].view(-1))                
                
        return results 

    @torch.no_grad()
    def is_stop(self, t):
        if len(self.complete_hypothesis_scores[0]) == 0:
            return False
        else:
            max_hypothesis_score = max(torch.cat([score[:, 0] for score in self.score_table[t] if (type(score) is not float)]))
            max_complete_hypothesis_score = max(self.complete_hypothesis_scores[0])
            return max_complete_hypothesis_score > max_hypothesis_score
    
    @torch.no_grad()
    def compute_z_score(self, output_ids):
        counter = 0
        # Assume that the initial word is PAD, and remove this word.
        length = len(output_ids[0])-1 
        
        for t in range(len(output_ids[0])-1):
            green_mask, _ = self._generate_masks(output_ids[0][t])
            
            if green_mask[output_ids[0][t+1]] == 1.0:
                counter += 1
                
        return (counter - self.gamma*length) / math.sqrt(self.gamma * (1-self.gamma) * length), counter/length
