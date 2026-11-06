import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from loguru import logger
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import recall_score



import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from tqdm import tqdm
import numpy as np

def test_acc(model, data_loader, metric='acc', return_proba=False):
    model.eval()
    preds = []
    labels = []
    probas = []
    true_zeros = 0
    true_ones = 0
    predicted_true_zeros = 0
    predicted_true_ones = 0

    with torch.no_grad():
        #for input_ids, batch_labels, attention_masks in tqdm(data_loader):
        for input_ids, batch_labels, attention_masks,sentence_count in tqdm(data_loader):
            outputs = model(input_ids, attention_mask=attention_masks)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[-1] if type(outputs) == list else outputs
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)
            labels.extend(batch_labels.reshape(-1).cpu().numpy())

            # 统计 label=0 和 label=1 的准确率
            true_zeros += np.sum(batch_labels.cpu().numpy() == 0)
            predicted_true_zeros += np.sum((batch_preds == 0) & (batch_labels.cpu().numpy() == 0))
            true_ones += np.sum(batch_labels.cpu().numpy() == 1)
            predicted_true_ones += np.sum((batch_preds == 1) & (batch_labels.cpu().numpy() == 1))
            

            if logits.shape[1] == 1:
                # Binary classification using sigmoid
                probabilities = torch.sigmoid(logits).cpu().numpy().flatten()  # shape [B]
            else:
                # Multi-class (e.g., 2-class) classification using softmax
                probabilities = F.softmax(logits, dim=1).cpu().numpy()[:, 1]  # shape [B]
            probas.extend(probabilities.tolist())  # always [B]


    # 基础分类指标
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')#macro/weighted/micro



    human_recall = recall_score(labels, preds, labels=[0,1], pos_label=0, zero_division=0)
    machine_recall = recall_score(labels, preds, labels=[0,1], pos_label=1, zero_division=0)
    recall = (human_recall + machine_recall) / 2.0

    zero_acc = predicted_true_zeros / true_zeros if true_zeros > 0 else 0.0
    one_acc  = predicted_true_ones  / true_ones  if true_ones  > 0 else 0.0

    probas_np = np.asarray(probas)
    positive_probs = probas_np if probas_np.ndim == 1 else probas_np[:, 1]

    if len(np.unique(labels)) < 2:
        auroc = float("nan")
    else:
        auroc = roc_auc_score(labels, positive_probs)





    return float(acc), zero_acc, one_acc, f1, recall, human_recall, machine_recall,auroc






def fourier_high_statistic_by_dynamic_ratio(
    magnitude, attention_masks, input_ids,sentence_counts, tokenizer, high_ratio=0.3
):

    device = magnitude.device
    B, F = magnitude.size()

    # 全局均值而非局部
    seq_lengths = attention_masks.sum(dim=1).float()      # [B]

    T_avg = int(seq_lengths.mean().item())
    S_avg = max(1, int(sentence_counts.float().mean().item()))
    

    #print('T_avg:',T_avg)
    # 构建统一 cutoff 掩码
    low_end = max(1, int(F / T_avg))
    base_mid = F / S_avg
    tail_span = F - low_end - base_mid
    raw_mid_end = int(base_mid + tail_span * (1 - high_ratio))
    raw_mid_end = max(low_end + 1, min(raw_mid_end, F - 1))

    # 构造统一掩码 [F]
    low_mask = torch.zeros(F, device=device)
    mid_mask = torch.zeros(F, device=device)
    high_mask = torch.zeros(F, device=device)
    low_mask[:low_end] = 1.0
    mid_mask[low_end:raw_mid_end] = 1.0
    high_mask[raw_mid_end:] = 1.0

    # 扩展到 [B, F]
    low_masks = low_mask.unsqueeze(0).expand(B, -1)
    mid_masks = mid_mask.unsqueeze(0).expand(B, -1)
    high_masks = high_mask.unsqueeze(0).expand(B, -1)

    #eps = 1e-6  # 防止除零

    avg_low = (magnitude * low_masks).sum() / (low_masks.sum())
    avg_mid = (magnitude * mid_masks).sum() / (mid_masks.sum())
    avg_high = (magnitude * high_masks).sum() / (high_masks.sum())

    return avg_high, avg_mid, avg_low




def fourier_high_statistic_by_dynamic_ratio2(
    embed,labels, attention_masks, input_ids,sentence_counts, tokenizer, high_ratio=0.3
):
    device = embed.device
    freq_data = torch.fft.rfft(embed, dim=1)       # [B, F]
    magnitude = torch.abs(freq_data)                # [B, F]
            # [B, F]
    B, F = magnitude.size()

    # 全局均值而非局部
    seq_lengths = attention_masks.sum(dim=1).float()      # [B]

    T_avg = int(seq_lengths.mean().item())
    S_avg = max(1, int(sentence_counts.float().mean().item()))
    

    low_end = max(1, int(F / T_avg))
    base_mid = F / S_avg
    tail_span = F - low_end - base_mid
    raw_mid_end = int(base_mid + tail_span * (1 - high_ratio))
    raw_mid_end = max(low_end + 1, min(raw_mid_end, F - 1))

    # 构造统一掩码 [F]
    low_mask = torch.zeros(F, device=device)
    mid_mask = torch.zeros(F, device=device)
    high_mask = torch.zeros(F, device=device)
    low_mask[:low_end] = 1.0
    mid_mask[low_end:raw_mid_end] = 1.0
    high_mask[raw_mid_end:] = 1.0

    # 扩展到 [B, F]
    low_masks = low_mask.unsqueeze(0).expand(B, -1)
    mid_masks = mid_mask.unsqueeze(0).expand(B, -1)
    high_masks = high_mask.unsqueeze(0).expand(B, -1)


    mask_1 = (labels == 1).unsqueeze(1).expand_as(magnitude).float()
    mask_0 = (labels == 0).unsqueeze(1).expand_as(magnitude).float()

    def safe_mean(val, mask):
        return (val * mask).sum() / (mask.sum())

    high_1 = safe_mean(magnitude * high_masks, mask_1)
    high_0 = safe_mean(magnitude * high_masks, mask_0)
    mid_1  = safe_mean(magnitude * mid_masks,  mask_1)
    mid_0  = safe_mean(magnitude * mid_masks,  mask_0)
    low_1  = safe_mean(magnitude * low_masks,  mask_1)
    low_0  = safe_mean(magnitude * low_masks,  mask_0)
    


    return [high_1, high_0], [mid_1, mid_0], [low_1, low_0]


def compute_statistics_all_mean(train_loader_all, pretrained_model, tokenizer, device, prominence=0.3, sub_batch_size=500):
    pretrained_model.eval()

    all_embed = []
    all_attention = []
    all_labels = []
    all_input_ids = []
    all_sent_counts = []

    with torch.no_grad():
        for input_ids_all, labels_all, attention_masks_all, sentence_counts_all in train_loader_all:
            input_ids_all = input_ids_all.to(device)
            attention_masks_all = attention_masks_all.to(device)
            labels_all = labels_all.to(device)
            sentence_counts_all = sentence_counts_all.to(device)

            total = input_ids_all.size(0)
            for start in range(0, total, sub_batch_size):
                end = min(start + sub_batch_size, total)
                input_ids = input_ids_all[start:end]
                attention_masks = attention_masks_all[start:end]
                labels = labels_all[start:end]
                sent_counts = sentence_counts_all[start:end]

                outputs = pretrained_model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
                pooled = outputs.hidden_states[-1][:, 0, :].float()  # [B, D]

                all_embed.append(pooled.cpu())
                all_attention.append(attention_masks.cpu())
                all_labels.append(labels.cpu())
                all_input_ids.append(input_ids.cpu())
                all_sent_counts.append(sent_counts.cpu())

    all_embed = torch.cat(all_embed, dim=0).to(device)
    all_attention = torch.cat(all_attention, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    all_input_ids = torch.cat(all_input_ids, dim=0).to(device)
    all_sent_counts = torch.cat(all_sent_counts, dim=0).to(device)

    freq_data = torch.fft.rfft(all_embed, dim=1)  # [N, F]
    magnitude = torch.abs(freq_data)              # [N, F]

    mag_1 = magnitude[all_labels == 1]
    ids_1 = all_input_ids[all_labels == 1]
    attn_1 = all_attention[all_labels == 1]
    sc_1 = all_sent_counts[all_labels == 1]

    mag_0 = magnitude[all_labels == 0]
    ids_0 = all_input_ids[all_labels == 0]
    attn_0 = all_attention[all_labels == 0]
    sc_0 = all_sent_counts[all_labels == 0]



    if mag_1.size(0) > 0:
        h1, m1, l1 = fourier_high_statistic_by_dynamic_ratio(mag_1, attn_1, ids_1, sc_1, tokenizer, high_ratio=prominence)
    else:
        h1 = m1 = l1 = torch.tensor(0.0, device=device)

    if mag_0.size(0) > 0:
        h0, m0, l0 = fourier_high_statistic_by_dynamic_ratio(mag_0, attn_0, ids_0, sc_0, tokenizer, high_ratio=prominence)
    else:
        h0 = m0 = l0 = torch.tensor(0.0, device=device)

    high_mean = [h1, h0]
    mid_mean = [m1, m0]
    low_mean = [l1, l0]

    return high_mean, mid_mean, low_mean





def fourier_high_statistic_by_dynamic_ratio_batch(embed, labels, attention_masks,sentence_count,high_mean_global, mid_mean_global, high_mean_base, mid_mean_base, high_ratio=0.3):
    
    device = embed.device
    freq_data = torch.fft.rfft(embed, dim=1)       # [B, F]
    magnitude = torch.abs(freq_data)                # [B, F]
    phase = torch.angle(freq_data)                  # [B, F]
    B, F = magnitude.size()

    seq_lengths = attention_masks.sum(dim=1)        # [B]


    # S_avg = max(sentence_counts.float().min().item())

    
    perturbed_embed = torch.zeros_like(embed)
    enhanced_freq_list = []
    enhanced_mag_list = []
    
    for b in range(B):
        T_b = int(seq_lengths[b].item())
        S_b = int(sentence_count[b].item())

        # print('S_b:',int(S_b))
        # print('T_b:',int(T_b))
        
        low_end = max(1, int(F / T_b))  
        
        base_mid = F / max(S_b, 1)               
        tail_span = F - low_end - base_mid        
        raw_mid_end = int(base_mid + tail_span * (1 - high_ratio))

        raw_mid_end = max(low_end + 1, min(raw_mid_end, F - 1))  


        low_mask = torch.zeros(F, device=device); low_mask[:low_end] = 1.0
        mid_mask = torch.zeros(F, device=device); mid_mask[low_end:raw_mid_end] = 1.0
        high_mask = torch.zeros(F, device=device); high_mask[raw_mid_end:] = 1.0


        mag_b = magnitude[b]  # [F]
        high_freq = mag_b * high_mask
        mid_freq = mag_b * mid_mask
        low_freq = mag_b * low_mask

        def safe_div(g, x, device):
            g = torch.tensor(g, device=device) if not isinstance(g, torch.Tensor) else g
            x = torch.tensor(x, device=device) if not isinstance(x, torch.Tensor) else x
            return g / x if (torch.isfinite(x) and x > 0 and torch.isfinite(g) and g > 0) else torch.tensor(1.0, device=device)
        
        if labels[b] == 1:
            mh = safe_div(high_mean_global[0], high_mean_base[0], device)
            mm = safe_div(mid_mean_global[0],  mid_mean_base[0],  device)
        else:
            mh = safe_div(high_mean_global[1], high_mean_base[1], device)
            mm = safe_div(mid_mean_global[1],  mid_mean_base[1], device)



        enhanced_mag = mh * high_freq + mm * mid_freq

          
        enhanced_mag_list.append(enhanced_mag.unsqueeze(0))

        perturbed_freq = enhanced_mag * torch.exp(1j * phase[b])  # [F]
        perturbed_embed[b] = torch.fft.irfft(perturbed_freq, n=embed.size(1), dim=0)
        enhanced_freq_list.append(perturbed_freq.unsqueeze(0))
        
    enhanced_freq = torch.cat(enhanced_freq_list, dim=0)
    enhanced_mag = torch.cat(enhanced_mag_list, dim=0)  
    
    return perturbed_embed, enhanced_freq, enhanced_mag 






def moving_average(x, window_size):

    padding = window_size // 2
    weight = torch.ones(1, 1, window_size, device=x.device) / window_size  # [1, 1, W]
    x = x.unsqueeze(1)  # [B, 1, F]
    x = F.pad(x, (padding, padding), mode='reflect') 
    smoothed = F.conv1d(x, weight).squeeze(1)  # [B, F]
    return smoothed


import torch
import torch.nn.functional as F_nn

import numpy as np
from scipy.ndimage import uniform_filter1d




def margin_freq_loss(magnitude, labels, method='mse', smooth=True, window_size=7, normalize=True):

    if magnitude.ndim == 3 and magnitude.size(1) == 1:
        magnitude = magnitude.squeeze(1)
    elif magnitude.ndim != 2:
        raise ValueError(f"Expected shape [B, F], but got {magnitude.shape}")

    B, F = magnitude.shape


    # if smooth:
    #     magnitude = moving_average(magnitude, window_size)

    # if normalize:
    #     min_vals = magnitude.min(dim=1, keepdim=True).values
    #     max_vals = magnitude.max(dim=1, keepdim=True).values
    #     magnitude = (magnitude - min_vals) / (max_vals - min_vals + 1e-8)




    mag1 = magnitude.unsqueeze(1)  # [B, 1, F]
    mag2 = magnitude.unsqueeze(0)  # [1, B, F]

    if method == 'mae':
        dist = torch.abs(mag1 - mag2).mean(dim=-1)  # [B, B]
    elif method == 'mse':
        dist = ((mag1 - mag2) ** 2).mean(dim=-1)
    else:
        raise ValueError("Only 'mae' or 'mse' are supported")
    
    dist = dist / (dist.max().detach() + 1e-8)

    label_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)
    pos_mask = label_matrix.float() - torch.eye(B, device=magnitude.device)  
    neg_mask = 1.0 - label_matrix.float()                   

    margin = (dist * pos_mask).max().detach() 


    pos_term = (dist * pos_mask).sum(-1) / (pos_mask.sum(-1) + 1e-6)
    neg_term = (F_nn.relu(margin - dist) * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-6)
    freq_margin_loss = (pos_term + neg_term).mean()

    return freq_margin_loss





def train_common_machine_mix_all(model, optimizer, data_loader,tokenizer, high_mean_global, mid_mean_global,low_mean_global, log_steps=100,
                                 pre_model=None, shift11=0, scl_reg=2.0, loss_type='ce', 
                                 save_steps=-1, save_dir=None, step_counter=0, max_grad_norm=1.0, type_base1=1,device=0):


    device = 'cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu'


    model.train()
    if pre_model is not None:
        pre_model.eval()

    running_loss = 0
    step_count = 0
    
    total_T = 0.0
    total_S = 0.0
    batch_count = 0

    for input_ids, labels, attention_masks,sentence_count in data_loader:
        
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        attention_masks = attention_masks.to(device)
        sentence_count = sentence_count.to(device)
        
        mask_1_base = labels == 1
        mask_0_base = labels == 0
        
        input_ids_1 = input_ids[mask_1_base]
        input_ids_0 = input_ids[mask_0_base]


        if hasattr(model, 'centers') or (hasattr(model, 'module') and hasattr(model.module, 'centers')):  # LMCL loss version
            outputs = model(input_ids, attention_mask=attention_masks, label=labels)
        else:
            outputs = model(input_ids, attention_mask=attention_masks, output_hidden_states=True)
            
        if hasattr(outputs,'logits'):
            logits = outputs.logits
        else:
            logits = outputs


        data_base = outputs.hidden_states[-1][:, 0, :]
        
        seq_lengths = attention_masks.sum(dim=1)  # [B]
        
        T_avg = seq_lengths.float().mean().item()
        S_avg = sentence_count.float().mean().item()
        window_now = max(T_avg / S_avg, 5)


            

        high_mean_base, mid_mean_base,low_mean_base = fourier_high_statistic_by_dynamic_ratio2(data_base,labels, attention_masks, input_ids, sentence_count, tokenizer, high_ratio=type_base1)

        combined_data,enhanced_freq,enhanced_mag= fourier_high_statistic_by_dynamic_ratio_batch(data_base, labels, attention_masks,sentence_count,high_mean_global, mid_mean_global,high_mean_base, mid_mean_base,high_ratio=type_base1)
        combined_labels = labels 
            


        if combined_data.dim() == 2:
            combined_data = combined_data.unsqueeze(1) 

        logits_combined = model.module.classifier(combined_data)
        
        loss_fct = CrossEntropyLoss()
        
        loss = loss_fct(logits_combined, combined_labels)
        

        
        if loss_type == 'mae':
            magnitude = torch.abs(enhanced_freq)
            loss += scl_reg * margin_freq_loss(magnitude, combined_labels, method=loss_type,window_size=int(window_now))
            


        elif loss_type == 'mse':
            freq_data = torch.fft.rfft(combined_data, dim=1)
            magnitude = torch.abs(freq_data)
            loss +=scl_reg* margin_freq_loss(magnitude, combined_labels,method=loss_type) 

        if shift11 > 0 and pre_model is not None:
            dist = torch.sum(torch.abs(torch.cat(
                [p.view(-1) for n, p in model.named_parameters() if 'bert' in n.lower()]) - torch.cat(
                [p.view(-1) for n, p in pre_model.named_parameters() if
                 'bert' in n.lower()])) ** 2).item()
            loss += shift11 * dist

        running_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        if (step_count + 1) % log_steps == 0:
            logger.info("step {}, running loss = {}".format(step_count + 1, running_loss))
            running_loss = 0

        step_count += 1
        step_counter += 1

        if save_steps != -1 and save_dir is not None:
            if step_counter % save_steps == 0:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), '{}/step{}_model.pt'.format(save_dir, step_counter))
                logger.info("model saved to {}/step{}_model.pt".format(save_dir, step_counter))

    return step_counter


