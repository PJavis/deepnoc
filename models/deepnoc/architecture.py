"""
deepNoC neural network architecture.

Based on Taylor & Humphries (2024): 16 layers from input to profile NoC output,
with secondary outputs at peak, locus, and profile levels providing explainability.
Secondary outputs are fed back into the main branch.

Input: [batch, 24 loci, 50 peaks, 89 features]
Outputs:
  1. peak_prop_allelic:    [batch, 24, 50, 1]     - proportion of peak that is allelic
  2. peak_n_alleles:       [batch, 24, 50, 21]    - number of alleles at position (0-20)
  3. locus_mix_props:      [batch, 24, 10]         - mixture proportions per locus
  4. locus_n_alleles:      [batch, 24, 20]         - number of alleles per locus (1-20)
  5. profile_mix_props:    [batch, 10]              - overall mixture proportions
  6. profile_noc:          [batch, 10]              - number of contributors (1-10)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.constants import NUM_LOCI, MAX_PEAKS_PER_LOCUS, NUM_FEATURES_PER_PEAK


class PeakProcessor(nn.Module):
    """Process peaks within each locus using 1D convolutions."""
    
    def __init__(self, in_features=89, hidden=128):
        super().__init__()
        # Conv1D across peaks dimension (treat features as channels)
        self.conv1 = nn.Conv1d(in_features, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden)
    
    def forward(self, x):
        """
        x: [batch * 24, 89, 50] (features as channels, peaks as sequence)
        returns: [batch * 24, hidden, 50]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x


class PeakOutputHead(nn.Module):
    """Secondary outputs at the peak level."""
    
    def __init__(self, in_features=128):
        super().__init__()
        # Proportion allelic: per-peak scalar [0, 1]
        self.prop_allelic = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        # Number of alleles: per-peak classification (0 to 20)
        self.n_alleles = nn.Sequential(
            nn.Conv1d(in_features, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 21, kernel_size=1),
        )
    
    def forward(self, x):
        """
        x: [batch * 24, hidden, 50]
        returns: prop_allelic [batch*24, 1, 50], n_alleles [batch*24, 21, 50]
        """
        prop = self.prop_allelic(x)
        n_all = self.n_alleles(x)
        return prop, n_all


class LocusProcessor(nn.Module):
    """Aggregate peak information to locus level."""
    
    def __init__(self, in_features=128 + 1 + 21, hidden=128):
        super().__init__()
        # After concatenating peak outputs back, process across peaks
        self.conv1 = nn.Conv1d(in_features, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden)
        
        # Aggregate peaks → locus representation
        self.pool = nn.AdaptiveMaxPool1d(1)  # [hidden, 50] → [hidden, 1]
        
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
    
    def forward(self, x):
        """
        x: [batch * 24, in_features, 50]
        returns: [batch * 24, hidden] (locus representation)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)  # [batch*24, hidden]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class LocusOutputHead(nn.Module):
    """Secondary outputs at the locus level."""
    
    def __init__(self, in_features=128):
        super().__init__()
        # Mixture proportions per locus: [10] summing to 1
        self.mix_props = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=-1),
        )
        # Number of alleles per locus: classification (1 to 20)
        self.n_alleles = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
        )
    
    def forward(self, x):
        """
        x: [batch * 24, hidden]
        returns: mix_props [batch*24, 10], n_alleles [batch*24, 20]
        """
        return self.mix_props(x), self.n_alleles(x)


class ProfileProcessor(nn.Module):
    """Aggregate locus information to profile level."""
    
    def __init__(self, in_features=128 + 10 + 20, hidden=256):
        super().__init__()
        # Process across loci (treat 24 loci as sequence)
        self.conv1 = nn.Conv1d(in_features, hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden)
        
        # Aggregate loci → profile representation
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 128)
    
    def forward(self, x):
        """
        x: [batch, in_features, 24]
        returns: [batch, 128]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ProfileOutputHead(nn.Module):
    """Final outputs at the profile level."""
    
    def __init__(self, in_features=128):
        super().__init__()
        # Mixture proportions: [10] summing to 1
        self.mix_props = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
            nn.Softmax(dim=-1),
        )
        # NoC: classification (1 to 10)
        self.noc = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        """
        x: [batch, hidden]
        returns: mix_props [batch, 10], noc_logits [batch, 10]
        """
        return self.mix_props(x), self.noc(x)


class DeepNoC(nn.Module):
    """
    deepNoC: Deep learning system for NoC assignment.
    
    Architecture follows Figure 1 of Taylor & Humphries (2024):
    - Peak processing → peak outputs → feedback into main branch
    - Locus aggregation → locus outputs → feedback into main branch
    - Profile aggregation → profile outputs (mixture props + NoC)
    
    Total ~16 layers from input to profile NoC output.
    """
    
    def __init__(self, peak_hidden=128, locus_hidden=128, profile_hidden=256,
                 num_classes=10):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Stage 1: Peak processing (layers 1-3)
        self.peak_processor = PeakProcessor(
            in_features=NUM_FEATURES_PER_PEAK, hidden=peak_hidden
        )
        self.peak_output = PeakOutputHead(peak_hidden)
        
        # Stage 2: Locus processing (layers 4-9)
        # Input: peak features + peak outputs concatenated
        self.locus_processor = LocusProcessor(
            in_features=peak_hidden + 1 + 21,  # peak feat + prop_allelic + n_alleles
            hidden=locus_hidden,
        )
        self.locus_output = LocusOutputHead(locus_hidden)
        
        # Stage 3: Profile processing (layers 10-16)
        # Input: locus features + locus outputs concatenated
        self.profile_processor = ProfileProcessor(
            in_features=locus_hidden + 10 + 20,  # locus feat + mix_props + n_alleles
            hidden=profile_hidden,
        )
        self.profile_output = ProfileOutputHead(128)  # ProfileProcessor outputs 128
    
    def forward(self, x):
        """
        Args:
            x: [batch, 24, 50, 89] - input tensor
        
        Returns dict of outputs:
            peak_prop_allelic:    [batch, 24, 50, 1]
            peak_n_alleles:       [batch, 24, 50, 21]
            locus_mix_props:      [batch, 24, 10]
            locus_n_alleles:      [batch, 24, 20]
            profile_mix_props:    [batch, 10]
            profile_noc:          [batch, 10]  (logits)
        """
        batch_size = x.shape[0]
        
        # ====== Stage 1: Peak Processing ======
        # Reshape: [batch, 24, 50, 89] → [batch*24, 89, 50]
        x_peaks = x.view(batch_size * NUM_LOCI, MAX_PEAKS_PER_LOCUS,
                         NUM_FEATURES_PER_PEAK)
        x_peaks = x_peaks.permute(0, 2, 1)  # [B*24, 89, 50]
        
        peak_feat = self.peak_processor(x_peaks)  # [B*24, hidden, 50]
        
        # Peak outputs
        peak_prop, peak_nall = self.peak_output(peak_feat)
        # peak_prop: [B*24, 1, 50], peak_nall: [B*24, 21, 50]
        
        # ====== Feedback peak outputs into main branch ======
        # Concatenate: [B*24, hidden+1+21, 50]
        peak_combined = torch.cat([peak_feat, peak_prop, peak_nall], dim=1)
        
        # ====== Stage 2: Locus Processing ======
        locus_feat = self.locus_processor(peak_combined)  # [B*24, locus_hidden]
        
        # Locus outputs
        locus_mix, locus_nall = self.locus_output(locus_feat)
        # locus_mix: [B*24, 10], locus_nall: [B*24, 20]
        
        # ====== Feedback locus outputs into main branch ======
        locus_combined = torch.cat([locus_feat, locus_mix, locus_nall], dim=1)
        # [B*24, locus_hidden+10+20]
        
        # Reshape for profile processing: [B, features, 24]
        locus_combined = locus_combined.view(batch_size, NUM_LOCI, -1)
        locus_combined = locus_combined.permute(0, 2, 1)  # [B, features, 24]
        
        # ====== Stage 3: Profile Processing ======
        profile_feat = self.profile_processor(locus_combined)  # [B, 128]
        
        # Profile outputs
        profile_mix, profile_noc = self.profile_output(profile_feat)
        
        # ====== Reshape outputs ======
        # Peak outputs: [B*24, C, 50] → [B, 24, 50, C]
        peak_prop_out = peak_prop.permute(0, 2, 1).view(batch_size, NUM_LOCI,
                                                         MAX_PEAKS_PER_LOCUS, 1)
        peak_nall_out = peak_nall.permute(0, 2, 1).view(batch_size, NUM_LOCI,
                                                          MAX_PEAKS_PER_LOCUS, 21)
        
        # Locus outputs: [B*24, C] → [B, 24, C]
        locus_mix_out = locus_mix.view(batch_size, NUM_LOCI, 10)
        locus_nall_out = locus_nall.view(batch_size, NUM_LOCI, 20)
        
        return {
            'peak_prop_allelic': peak_prop_out,
            'peak_n_alleles': peak_nall_out,
            'locus_mix_props': locus_mix_out,
            'locus_n_alleles': locus_nall_out,
            'profile_mix_props': profile_mix,
            'profile_noc': profile_noc,
        }
    
    def predict_noc(self, x):
        """Convenience method: get NoC prediction only."""
        with torch.no_grad():
            outputs = self.forward(x)
            probs = F.softmax(outputs['profile_noc'], dim=-1)
            return probs.argmax(dim=-1) + 1  # 1-indexed NoC
    
    def predict_noc_probs(self, x):
        """Get full NoC probability distribution."""
        with torch.no_grad():
            outputs = self.forward(x)
            return F.softmax(outputs['profile_noc'], dim=-1)


class DeepNoCSimple(nn.Module):
    """
    Simplified deepNoC without secondary outputs.
    
    For quick baseline comparison - just the main branch
    from input to NoC classification.
    """
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        # Per-peak processing
        self.peak_net = nn.Sequential(
            nn.Conv1d(89, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # [128, 50] → [128, 1]
        )
        
        # Per-locus (across 24 loci)
        self.locus_net = nn.Sequential(
            nn.Conv1d(128, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # [256, 24] → [256, 1]
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        """
        x: [batch, 24, 50, 89]
        returns: logits [batch, num_classes]
        """
        B = x.shape[0]
        
        # Process each locus's peaks
        x = x.view(B * 24, 50, 89).permute(0, 2, 1)  # [B*24, 89, 50]
        x = self.peak_net(x).squeeze(-1)  # [B*24, 128]
        
        # Reshape to loci sequence
        x = x.view(B, 24, 128).permute(0, 2, 1)  # [B, 128, 24]
        x = self.locus_net(x).squeeze(-1)  # [B, 256]
        
        return self.classifier(x)


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    model = DeepNoC()
    print(f"DeepNoC parameters: {count_parameters(model):,}")
    
    x = torch.randn(4, 24, 50, 89)
    outputs = model(x)
    
    print("\nOutput shapes:")
    for name, tensor in outputs.items():
        print(f"  {name}: {tensor.shape}")
    
    # Test simple version
    simple = DeepNoCSimple()
    print(f"\nDeepNoCSimple parameters: {count_parameters(simple):,}")
    out = simple(x)
    print(f"  output: {out.shape}")