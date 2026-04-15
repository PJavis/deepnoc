"""
Constants for the GlobalFiler profiling kit and utility data.
"""

# 24 loci in the GlobalFiler kit (order matters for one-hot encoding)
GLOBALFILER_LOCI = [
    "D3S1358", "vWA", "D16S539", "CSF1PO", "TPOX",
    "Y-Indel", "AMEL", "D8S1179", "D21S11", "D18S51",
    "DYS391", "D2S441", "D19S433", "TH01", "FGA",
    "D22S1045", "D5S818", "D13S317", "D7S820", "SE33",
    "D10S1248", "D1S1656", "D12S391", "D2S1338",
]

LOCUS_TO_IDX = {locus: i for i, locus in enumerate(GLOBALFILER_LOCI)}

# Alternative names that may appear in GeneMapper CSVs
LOCUS_ALIASES = {
    "Yindel": "Y-Indel",
    "Y Indel": "Y-Indel",
    "AMEL": "AMEL",
    "Amelogenin": "AMEL",
    "VWA": "vWA",
    "vwa": "vWA",
    "TH01": "TH01",
    "THO1": "TH01",
    "TPOX": "TPOX",
    "CSF1PO": "CSF1PO",
    "FGA": "FGA",
    "SE33": "SE33",
}

NUM_LOCI = 24
MAX_PEAKS_PER_LOCUS = 50
NUM_FEATURES_PER_PEAK = 89
MAX_NOC = 10

# Normalization constants (from paper Section 2.4)
ALLELE_NORM = 100.0
SIZE_NORM = 100.0
HEIGHT_NORM = 33000.0
LOCUS_PEAK_NORM = 100.0
PROFILE_PEAK_NORM = 1000.0

# Approximate allele frequencies for GlobalFiler loci
# (Australian Caucasian population from Taylor et al. [32])
# These are simplified/approximate - replace with actual frequencies for production use
# Format: {locus: {allele: frequency}}
# Using common alleles with approximate frequencies
DEFAULT_ALLELE_FREQ = 0.01  # Default for unknown alleles

# Stutter ratios (approximate expected values for GlobalFiler)
EXPECTED_BACK_STUTTER_RATIO = 0.10    # -1 repeat
EXPECTED_DBL_BACK_STUTTER_RATIO = 0.01  # -2 repeat
EXPECTED_FWD_STUTTER_RATIO = 0.03     # +1 repeat
EXPECTED_PT2_STUTTER_RATIO = 0.005    # +/- 0.2 repeat (incomplete repeats)

# Dye channels in GlobalFiler
DYE_CHANNELS = {
    "B": ["D3S1358", "vWA", "D16S539", "CSF1PO", "TPOX"],          # Blue
    "G": ["Y-Indel", "AMEL", "D8S1179", "D21S11", "D18S51", "DYS391"],  # Green
    "Y": ["D2S441", "D19S433", "TH01", "FGA"],                      # Yellow
    "R": ["D22S1045", "D5S818", "D13S317", "D7S820", "SE33"],       # Red
    "P": ["D10S1248", "D1S1656", "D12S391", "D2S1338"],             # Purple
}