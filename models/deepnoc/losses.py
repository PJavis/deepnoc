"""
Loss functions for deepNoC multi-output training.

From the paper (Section 2.5):
- MSE for: peak proportion allelic, locus mixture proportions, profile mixture proportions
- Categorical cross-entropy for: peak n_alleles, locus n_alleles, profile NoC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepNoCLoss(nn.Module):
    """
    Combined multi-output loss for deepNoC.
    
    Weights can be adjusted to prioritize the NoC classification output.
    """
    
    def __init__(self, noc_weight=1.0, peak_weight=0.2, locus_weight=0.3,
                 profile_mix_weight=0.3):
        super().__init__()
        self.noc_weight = noc_weight
        self.peak_weight = peak_weight
        self.locus_weight = locus_weight
        self.profile_mix_weight = profile_mix_weight
        
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, outputs: dict, targets: dict) -> dict:
        """
        Compute all losses.
        
        Args:
            outputs: dict from DeepNoC.forward()
            targets: dict with matching keys
        
        Returns:
            dict with individual losses and 'total' loss
        """
        losses = {}
        total = 0.0
        
        # 1. Profile NoC (main output) — always required
        if 'profile_noc' in targets:
            # targets['profile_noc']: [B] with values 1-10 → convert to 0-indexed
            noc_target = targets['profile_noc'] - 1  # 0-indexed for CE
            losses['noc'] = self.ce(outputs['profile_noc'], noc_target)
            total += self.noc_weight * losses['noc']
        
        # 2. Peak proportion allelic (MSE)
        if 'peak_prop_allelic' in targets:
            losses['peak_prop'] = self.mse(
                outputs['peak_prop_allelic'],
                targets['peak_prop_allelic']
            )
            total += self.peak_weight * losses['peak_prop']
        
        # 3. Peak number of alleles (CE)
        if 'peak_n_alleles' in targets:
            # Reshape for CE: [B*24*50, 21] vs [B*24*50]
            pred = outputs['peak_n_alleles'].view(-1, 21)
            tgt = targets['peak_n_alleles'].view(-1)
            # Only compute loss on non-zero targets (where there are actual peaks)
            mask = tgt >= 0
            if mask.any():
                losses['peak_nall'] = self.ce(pred[mask], tgt[mask])
                total += self.peak_weight * losses['peak_nall']
        
        # 4. Locus mixture proportions (MSE)
        if 'locus_mix_props' in targets:
            losses['locus_mix'] = self.mse(
                outputs['locus_mix_props'],
                targets['locus_mix_props']
            )
            total += self.locus_weight * losses['locus_mix']
        
        # 5. Locus number of alleles (CE)
        if 'locus_n_alleles' in targets:
            pred = outputs['locus_n_alleles'].view(-1, 20)
            tgt = targets['locus_n_alleles'].view(-1)
            mask = tgt >= 0
            if mask.any():
                losses['locus_nall'] = self.ce(pred[mask], tgt[mask])
                total += self.locus_weight * losses['locus_nall']
        
        # 6. Profile mixture proportions (MSE)
        if 'profile_mix_props' in targets:
            losses['profile_mix'] = self.mse(
                outputs['profile_mix_props'],
                targets['profile_mix_props']
            )
            total += self.profile_mix_weight * losses['profile_mix']
        
        losses['total'] = total
        return losses


class NoCOnlyLoss(nn.Module):
    """Simple cross-entropy loss for NoC classification only."""
    
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: [B, num_classes]
            targets: [B] with values 1-indexed (1-10)
        """
        return self.ce(logits, targets - 1)  # Convert to 0-indexed