import torch
from transformers import T5Tokenizer, GPT2Tokenizer, AutoModelForCausalLM, pipeline, AutoTokenizer, T5ForConditionalGeneration
import hashlib
import random
import numpy as np
from tqdm import tqdm
import math
from util import *

def soft_watermark(model, input_ids, num_beams=2, max_length=100, gamma=0.5, delta=1.0, forced_bos_token_id=None, encoder_decoder_model=False):

    if encoder_decoder_model:
        if forced_bos_token_id is None:
            beams = [torch.LongTensor([[model.config.decoder_start_token_id]]).to(model.device) for _ in range(num_beams)]
        else:
            beams = [torch.LongTensor([[model.config.decoder_start_token_id, forced_bos_token_id]]).to(model.device) for _ in range(num_beams)]            
    else:
        beams = [torch.LongTensor([[]]).to(model.device) for _ in range(num_beams)]
        
    beam_scores = [0.0 for _ in range(num_beams)]

    complete_beams = []
    complete_beam_scores = []
    
    with torch.no_grad():
        for t in tqdm(range(max_length)):
        
            tmp = []
            
            if t == 0:
                
                if encoder_decoder_model:
                    output = model(input_ids, decoder_input_ids=beams[0])
                else:
                    output = model(torch.cat([input_ids, beams[0]], dim=1))
                
                next_token_logits = output.logits[0, -1, :]
                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                next_token_probs = next_token_probs 
                tmp.append(torch.log(next_token_probs))
            else:
                for i in range(num_beams):
                    
                    if encoder_decoder_model:
                        output = model(input_ids, decoder_input_ids=beams[i])
                    else:
                        output = model(torch.cat([input_ids, beams[i]], dim=1))

                    next_token_logits = output.logits[0, -1, :]

                    # add wartermark constraints.
                    mask = generate_mask(model.config.vocab_size, beams[i][0][-1], gamma=gamma).to(model.device)
                    next_token_probs = torch.softmax(next_token_logits + delta * mask, dim=-1)
                    next_token_probs = next_token_probs
                    tmp.append(torch.log(next_token_probs) + beam_scores[i])
  
        
            all_probs = torch.cat(tmp)
            sorted_id = torch.argsort(all_probs, descending=True, dim=-1)
            
            new_beam_scores = []
            new_beams = []
        
            i = 0
            while len(new_beams) < num_beams:
                word_id = sorted_id[None, i, None] % model.config.vocab_size
                beam_id = int(sorted_id[i] / model.config.vocab_size)
                score = all_probs[sorted_id[i]]
                generated_sentence = torch.cat([beams[beam_id], word_id], dim=-1)
                
                if word_id == model.config.eos_token_id:
                    complete_beams.append(generated_sentence)
                    complete_beam_scores.append(score)
                else:
                    new_beams.append(generated_sentence)
                    new_beam_scores.append(score)                
                i += 1

            beam_scores = new_beam_scores
            beams = new_beams

    if len(complete_beams) == 0:
        return beams[beam_scores.index(max(beam_scores))]        
    else: # len(complete_beams) > 0:
        if max(beam_scores) > max(complete_beam_scores):
            return beams[beam_scores.index(max(beam_scores))]        
        else:
            return complete_beams[complete_beam_scores.index(max(complete_beam_scores))]


def adaptive_soft_watermark(model, input_ids, num_bos_tokens, num_beams=2, max_length=100, gamma=0.5, delta_list=[4,6,8,10,12], forced_bos_token_id=None, encoder_decoder_model=False, Z=4):

    min_idx = 0
    max_idx = len(delta_list)-1

    # idx whose score exceeds Z.
    valid_z_score = []
    valid_output_ids = []

    # idx whose score does not exceed Z.    
    invalid_z_score = []
    invalid_output_ids = []
    
    while min_idx <= max_idx:
        middle_idx = (min_idx + max_idx) // 2
        delta = delta_list[middle_idx]
        output_ids = soft_watermark(model, input_ids, num_beams=num_beams, max_length=max_length, gamma=gamma, delta=delta, forced_bos_token_id=forced_bos_token_id, encoder_decoder_model=encoder_decoder_model)

        if output_ids.shape[1] == 1:
            return output_ids
        
        z_score = compute_z_score(output_ids, model.config.vocab_size, gamma, num_bos_tokens)

        if z_score >= Z:
            max_idx = middle_idx - 1
            valid_z_score.append(z_score)
            valid_output_ids.append(output_ids)
        else:
            min_idx = middle_idx + 1
            invalid_z_score.append(z_score)
            invalid_output_ids.append(output_ids)
            
    if len(valid_z_score)>0:
        return valid_output_ids[valid_z_score.index(min(valid_z_score))]
    else:
        return invalid_output_ids[invalid_z_score.index(max(invalid_z_score))]
