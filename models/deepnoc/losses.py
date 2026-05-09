"""
Loss functions for deepNoC multi-output training - Đã tối ưu
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNoCLoss(nn.Module):
    """
    Multi-task loss cho deepNoC.
    Ưu tiên mạnh nhiệm vụ chính (profile NoC).
    """
    
    def __init__(self, 
                 noc_weight: float = 2.0,          # Tăng trọng số cho NoC
                 peak_weight: float = 0.1,
                 locus_weight: float = 0.15,
                 profile_mix_weight: float = 0.2):
        super().__init__()
        self.noc_weight = noc_weight
        self.peak_weight = peak_weight
        self.locus_weight = locus_weight
        self.profile_mix_weight = profile_mix_weight
        
        self.mse = nn.MSELoss(reduction='mean')
        self.ce = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, outputs: dict, targets: dict) -> dict:
        losses = {}
        total = 0.0
        
        # ==================== 1. Profile NoC (TASK CHÍNH) ====================
        if 'profile_noc' in outputs and 'profile_noc' in targets:
            noc_target = targets['profile_noc'] - 1          # 0-indexed
            losses['noc'] = self.ce(outputs['profile_noc'], noc_target)
            total += self.noc_weight * losses['noc']
        
        # ==================== 2. Peak proportion allelic (MSE) ====================
        if 'peak_prop_allelic' in outputs and 'peak_prop_allelic' in targets:
            # Chỉ tính trên các peak thực sự có height > 0 (không tính padding)
            pred = outputs['peak_prop_allelic']
            tgt = targets['peak_prop_allelic']
            
            # Tạo mask: nơi có peak (height feature index 26 > 0)
            # Giả sử outputs và targets có shape [B, 24, 50, 1]
            height_mask = targets.get('peak_height_mask', None)
            if height_mask is not None:
                mask = height_mask.view(-1)
                if mask.any():
                    losses['peak_prop'] = self.mse(pred.view(-1)[mask], tgt.view(-1)[mask])
                    total += self.peak_weight * losses['peak_prop']
            else:
                # Fallback nếu chưa có mask
                losses['peak_prop'] = self.mse(pred, tgt)
                total += self.peak_weight * losses['peak_prop']
        
        # ==================== 3. Peak number of alleles (CE) ====================
        if 'peak_n_alleles' in outputs and 'peak_n_alleles' in targets:
            pred = outputs['peak_n_alleles'].view(-1, 21)
            tgt = targets['peak_n_alleles'].view(-1)
            
            # Chỉ tính loss trên vị trí có peak thực (tgt != padding value)
            # Giả sử padding = -1 hoặc 0
            mask = tgt >= 0
            if mask.any():
                losses['peak_nall'] = self.ce(pred[mask], tgt[mask])
                total += self.peak_weight * losses['peak_nall']
        
        # ==================== 4. Locus mixture proportions (MSE) ====================
        if 'locus_mix_props' in outputs and 'locus_mix_props' in targets:
            losses['locus_mix'] = self.mse(
                outputs['locus_mix_props'], 
                targets['locus_mix_props']
            )
            total += self.locus_weight * losses['locus_mix']
        
        # ==================== 5. Locus number of alleles (CE) ====================
        if 'locus_n_alleles' in outputs and 'locus_n_alleles' in targets:
            pred = outputs['locus_n_alleles'].view(-1, 20)
            tgt = targets['locus_n_alleles'].view(-1)
            mask = tgt >= 0
            if mask.any():
                losses['locus_nall'] = self.ce(pred[mask], tgt[mask])
                total += self.locus_weight * losses['locus_nall']
        
        # ==================== 6. Profile mixture proportions (MSE) ====================
        if 'profile_mix_props' in outputs and 'profile_mix_props' in targets:
            losses['profile_mix'] = self.mse(
                outputs['profile_mix_props'], 
                targets['profile_mix_props']
            )
            total += self.profile_mix_weight * losses['profile_mix']
        
        losses['total'] = total
        return losses


class NoCOnlyLoss(nn.Module):
    """Loss đơn giản chỉ cho profile NoC"""
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets - 1)