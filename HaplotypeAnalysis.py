
import sys
!{sys.executable} -m pip install -U scikit-learn
from sklearn.impute import KNNImputer
import sklearn
print(sklearn.__version__)
import os
import numpy as np
import pandas as pd
import allel
import matplotlib.pyplot as plt
import sys
!{sys.executable} -m pip install matplotlib seaborn networkx
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
print("All plotting libraries imported successfully")
import gzip

import os, sys

# Save the ORIGINAL directory where the notebook/script started
PROJECT_ROOT = os.getcwd()

chrom_num = int(input("Enter chromosome number (e.g., 1): "))

BASE_OUTDIR = os.path.join(PROJECT_ROOT, "results")
CHR_OUTDIR = os.path.join(BASE_OUTDIR, f"chr{chrom_num}")

os.makedirs(CHR_OUTDIR, exist_ok=True)

# Change into chromosome-specific directory
os.chdir(CHR_OUTDIR)

print("=" * 60)
print(f"Running FULL PIPELINE for chr{chrom_num}")
print(f"Project root       : {PROJECT_ROOT}")
print(f"Chromosome folder  : {CHR_OUTDIR}")
print("=" * 60)


# USER SETTINGS

VCF_PATH = fr"your_vcf_path\your_vcf_file_name{chrom_num}.vcf" ### (update your own vcf path and name,my recommendation please not change {chrom_num} this otherwise you have to change everywhere the the chromosome number will come.)


# FUNCTIONS

def open_vcf(path):
    """Open .vcf or .vcf.gz automatically"""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    else:
        return open(path, "r")

###Step 1 — VCF allele inspection

def check_vcf_alleles(vcf_path):
    total_sites = 0
    biallelic_sites = 0
    multiallelic_sites = 0

    with open_vcf(vcf_path) as f:
        for line in f:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 5:
                continue

            ref = fields[3]
            alt = fields[4]

            # skip missing ALT
            if alt == ".":
                continue

            total_sites += 1

            # ALT alleles separated by comma
            alt_alleles = alt.split(",")

            if len(alt_alleles) == 1:
                biallelic_sites += 1
            else:
                multiallelic_sites += 1

    # Decide type
    if multiallelic_sites == 0:
        file_type = "PURE BI-ALLELIC VCF"
    elif biallelic_sites == 0:
        file_type = "PURE MULTI-ALLELIC VCF"
    else:
        file_type = "MIXED VCF (contains both bi-allelic + multi-allelic sites)"

    # Print results
    print("\n=============================")
    print("VCF ALLELE TYPE SUMMARY")
    print("=============================")
    print(f"VCF file           : {vcf_path}")
    print(f"Total variant sites: {total_sites}")
    print(f"Biallelic sites     : {biallelic_sites}")
    print(f"Multiallelic sites  : {multiallelic_sites}")
    print("-----------------------------")
    print(f"VCF TYPE: {file_type}")
    print("=============================\n")

    # Return results if needed
    return {
        "total": total_sites,
        "biallelic": biallelic_sites,
        "multiallelic": multiallelic_sites,
        "type": file_type
    }


# RUN

check_vcf_alleles(VCF_PATH)


# USER SETTINGS

VCF_PATH = fr"your_vcf_path\your_vcf_file_name{chrom_num}.vcf" ### (update your own vcf path and name,my recommendation please not change {chrom_num} this otherwise you have to change everywhere the the chromosome number will come.)

OUT_IMPUTED_VCF = f"imputed_knn_chr{chrom_num}.vcf"
OUT_PHASED_VCF = "give_your_own name_phased.vcf.gz" ### (update your own phased vcf name)


MIN_MAF = 0.05
MAX_MISSING_VARIANT = 0.1
MAX_MISSING_SAMPLE = 0.2
HWE_P = 1e-6
KNN_K = 5

MAX_DISTANCE_BP = 50000
BIN_SIZE_BP = 2000


# Beagle settings
BEAGLE_JAR = r"beagle file patha\beagle.27Feb25.75f.jar"  ### (update your own beagle path and name)
JAVA_EXE = "java file path\java.exe"   ### (update your own java path and name)



#READ VCF

print("Reading VCF...")
callset = allel.read_vcf(
    VCF_PATH,
    fields=["variants/CHROM", "variants/POS", "variants/ID", "variants/REF", "variants/ALT", "calldata/GT"]
)

chrom = callset["variants/CHROM"]
pos   = callset["variants/POS"]
ids   = callset["variants/ID"]
ref   = callset["variants/REF"]
alt   = callset["variants/ALT"]

gt = allel.GenotypeArray(callset["calldata/GT"])


import allel
import numpy as np
import pandas as pd


###Step 2 — Split multiallelic sites (SPLIT MULTIALLELIC → MULTIPLE BIALLELIC MARKERS)


def split_multiallelic_to_biallelic(chrom, pos, ids, ref, alt, gt):
    """
    Split multiallelic variants into multiple biallelic variants.
    Each ALT allele becomes one biallelic marker (0 vs that ALT).
    
    Returns:
      chrom2, pos2, ids2, ref2, alt2, gt2 (GenotypeArray)
    """

    chrom2, pos2, ids2, ref2, alt2 = [], [], [], [], []
    gt_list = []

    n_variants = gt.shape[0]

    for i in range(n_variants):
        alt_alleles = alt[i]

        # if missing ALT info
        if alt_alleles is None or len(alt_alleles) == 0 or alt_alleles[0] is None:
            continue

        # number of ALT alleles at this site
        n_alt = len(alt_alleles)

        # genotype array for this variant (samples, ploidy)
        g = gt[i].values  # shape (n_samples, 2)

        # loop through each ALT allele -> create a separate biallelic record
        for a in range(1, n_alt + 1):  # ALT allele index in VCF is 1..n_alt

            # create biallelic genotype:
            # 0 stays 0
            # allele == a becomes 1
            # other ALT alleles (not a) become missing (-1)
            gb = g.copy()

            # mark "other ALT alleles" as missing
            other_alt_mask = (gb > 0) & (gb != a)
            gb[other_alt_mask] = -1

            # recode selected ALT allele index -> 1
            gb[gb == a] = 1

            # keep REF=0 as is
            # missing stays -1

            # if all missing -> skip
            if np.all(gb < 0):
                continue

            # append
            chrom2.append(chrom[i])
            pos2.append(pos[i])

            id_i = ids[i]
            if id_i is None or id_i == "":
                id_i = "."
            ids2.append(str(id_i) + f"_ALT{a}")

            ref2.append(ref[i])
            alt2.append([alt_alleles[a-1]])  # store correct ALT for this split marker

            gt_list.append(gb)

    gt2 = allel.GenotypeArray(np.array(gt_list, dtype=np.int16))
    return (np.array(chrom2), np.array(pos2), np.array(ids2),
            np.array(ref2), np.array(alt2, dtype=object), gt2)


# APPLY THE SPLIT FUNCTION

print("Splitting multiallelic sites into biallelic markers...")
chrom, pos, ids, ref, alt, gt = split_multiallelic_to_biallelic(chrom, pos, ids, ref, alt, gt)

# Convert genotype → alt allele count (0/1/2, missing=-1)
gn = gt.to_n_alt().astype(float)
gn[gn < 0] = -1

n_variants, n_samples = gn.shape
print(f"Loaded {n_variants} markers after splitting (biallelic+split multiallelic) and {n_samples} samples")


###Step 3 — Variant QC (MAF, Missing, HWE)

gn = gt.to_n_alt().astype(float)
gn[gn < 0] = np.nan

variant_missing = np.isnan(gn).mean(axis=1)
alt_counts = np.nansum(gn, axis=1)
non_missing = np.sum(~np.isnan(gn), axis=1)
maf = np.minimum(alt_counts/(2*non_missing), 1-alt_counts/(2*non_missing))

obs_het = np.nansum(gn == 1, axis=1)
obs_hom_ref = np.nansum(gn == 0, axis=1)
obs_hom_alt = np.nansum(gn == 2, axis=1)
n = obs_hom_ref + obs_het + obs_hom_alt
p = (2*obs_hom_ref + obs_het)/(2*n)
q = 1-p
exp_het = 2*n*p*q
chi2 = (obs_het-exp_het)**2/(exp_het+1e-9)
from scipy.stats import chi2 as chi2_dist
hwe_p = 1 - chi2_dist.cdf(chi2, df=1)

variant_keep = (variant_missing <= MAX_MISSING_VARIANT) & (maf >= MIN_MAF) & (hwe_p >= HWE_P)

gt = gt.compress(variant_keep, axis=0)
print(f"Variants passing QC: {variant_keep.sum()} / {n_variants}")

# Apply variant QC ONCE and keep everything aligned

chrom = chrom[variant_keep]
pos   = pos[variant_keep]
ids   = ids[variant_keep]
ref   = ref[variant_keep]
alt   = alt[variant_keep]
gn    = gn[variant_keep, :]

print(f"Variants passing QC: {gt.shape[0]}")


###STEP 4 — Sample QC

sample_missing = np.isnan(gn).mean(axis=0)
sample_keep = sample_missing <= MAX_MISSING_SAMPLE
gt = gt.compress(sample_keep, axis=1)
gn = gn[:, sample_keep]

###Step 5 — KNN genotype imputation

print("Running sklearn KNN imputation...")
gn_knn = gn.copy()
gn_knn[gn_knn < 0] = np.nan

imputer = KNNImputer(n_neighbors=KNN_K, weights="distance")
gn_imp = imputer.fit_transform(gn_knn)   # returns float

# KNN can give values like 0.3, 1.6 etc → round to nearest genotype class
gn_imp = np.rint(gn_imp).astype(int)
gn_imp = np.clip(gn_imp, 0, 2)

print("KNN Imputation complete.")

# helper function: convert 0/1/2 genotype into VCF GT string

def altcount_to_gt(x):
    if x == 0:
        return "0/0"
    if x == 1:
        return "0/1"
    if x == 2:
        return "1/1"
    return "0/0"




###Step 6 — Write imputed VCF(UNPHASED)
print("Writing imputed VCF with correct FORMAT header...")

with open(VCF_PATH, "r") as fin, open(OUT_IMPUTED_VCF, "w") as fout:

    # copy header
    header_lines = []
    for line in fin:
        if line.startswith("#CHROM"):
            break
        header_lines.append(line)

    # Ensure FORMAT GT header exists
    has_gt_format = any(l.startswith("##FORMAT=<ID=GT") for l in header_lines)
    if not has_gt_format:
        header_lines.append(
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
        )

    # write header lines
    for l in header_lines:
        fout.write(l)

    # write column header
    fout.write(line)


    # write variant records
    
    for i in range(len(pos)):

        chrom_i = f"chr{chrom_num}"   
        pos_i   = str(pos[i])

        id_i = ids[i]
        if id_i is None or id_i == "":
            id_i = "."
        id_i = str(id_i)

        ref_i = str(ref[i])

        alt_arr = alt[i]
        if alt_arr is None or len(alt_arr) == 0:
            alt_i = "."
        else:
            alt_i = str(alt_arr[0])

        gt_fields = [altcount_to_gt(int(x)) for x in gn_imp[i]]

        if len(gt_fields) != gn_imp.shape[1]:
            raise ValueError(f"GT count mismatch at SNP {i}")

        row = [chrom_i, pos_i, id_i, ref_i, alt_i, ".", "PASS", ".", "GT"] + gt_fields
        fout.write("\t".join(row) + "\n")

print("Done:", OUT_IMPUTED_VCF)


import gzip, shutil

gz_path = OUT_IMPUTED_VCF + ".gz"
with open(OUT_IMPUTED_VCF, "rb") as f_in, gzip.open(gz_path, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)

print("GZ created:", gz_path)


###Step 7 — Phasing with Beagle

import subprocess, os, glob

# FULL PATHS (important)
IMPUTED_VCF_GZ = os.path.abspath(OUT_IMPUTED_VCF + ".gz")

OUT_PREFIX = os.path.abspath(f"imputed_knn_phased_chr{chrom_num}")

cmd = [
    JAVA_EXE, "-jar", BEAGLE_JAR,
    f"gt={IMPUTED_VCF_GZ}",
    f"out={OUT_PREFIX}",
    "nthreads=4"
]


print("Running Beagle command:\n", " ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True)

print("Return code:", result.returncode)

# Print errors if any
if result.stdout:
    print("STDOUT (first 2000 chars):\n", result.stdout[:2000])
if result.stderr:
    print("STDERR (first 4000 chars):\n", result.stderr[:4000])

# After run, detect output
candidates = glob.glob(OUT_PREFIX + "*.vcf*")

print("Output candidates:", candidates)

if len(candidates) == 0:
    raise FileNotFoundError("Beagle did not output phased VCF. See STDERR above.")

OUT_PHASED_VCF = candidates[0]
print("Using phased VCF:", OUT_PHASED_VCF)


#READ PHASED VCF

print("Reading phased VCF...")
phased = allel.read_vcf(
    OUT_PHASED_VCF,
    fields=["variants/POS", "calldata/GT"]
)

pos_phase = phased["variants/POS"]
gt_phase = allel.GenotypeArray(phased["calldata/GT"])

###Step 8 — Convert phased genotypes to haplotypes (2 haplotypes per sample)

# shape = (variants, samples, ploidy)
h = gt_phase.to_haplotypes()  # shape (variants, 2*samples)

print("Haplotype matrix shape:", h.shape)


###Step 9 —haplotype LD

def haplo_ld(h1, h2):
    """
    LD from haplotypes:
    haplotypes are binary allele copies {0,1} on phased chromosomes.
    """
    h1 = np.asarray(h1)
    h2 = np.asarray(h2)

    pA = h1.mean()
    pB = h2.mean()

    pAB = np.mean((h1 == 1) & (h2 == 1))

    D = pAB - pA * pB

    # r²
    denom = pA * (1 - pA) * pB * (1 - pB)
    r2 = (D**2 / denom) if denom > 0 else np.nan

    # D'
    if D >= 0:
        Dmax = min(pA*(1-pB), (1-pA)*pB)
    else:
        Dmax = min(pA*pB, (1-pA)*(1-pB))

    Dprime = D / Dmax if Dmax > 0 else np.nan

    return D, Dprime, r2


###Step 10 — LD decay

print("Computing haplotype LD decay...")
distances, r2_values = [], []

M = h.shape[0]
for i in range(M):
    for j in range(i+1, M):
        dist = abs(pos_phase[j] - pos_phase[i])
        if dist > MAX_DISTANCE_BP:
            break

        _, _, r2 = haplo_ld(h[i], h[j])
        if np.isfinite(r2):
            distances.append(dist)
            r2_values.append(r2)

print("Pairs collected:", len(r2_values))

decay_df = pd.DataFrame({"distance_bp": distances, "r2": r2_values})
bins = np.arange(0, MAX_DISTANCE_BP + BIN_SIZE_BP, BIN_SIZE_BP)
decay_df["bin"] = pd.cut(decay_df["distance_bp"], bins=bins)
ld_decay = decay_df.groupby("bin")["r2"].mean().reset_index()
ld_decay["distance_mid"] = [x.mid for x in ld_decay["bin"]]

ld_decay.to_csv(f"LD_decay_haplo_chr{chrom_num}.csv", index=False)

plt.figure(figsize=(8,6))
plt.plot(ld_decay["distance_mid"], ld_decay["r2"], marker="o")
plt.xlabel("Distance (bp)")
plt.ylabel("Mean r² (haplotype LD)")
plt.title(f"LD Decay Curve ( Haplotype LD) chr{chrom_num}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"LD_decay_haplo_chr{chrom_num}.png", dpi=300)
plt.show()


###Step 11 — Haplotype blocks

"""
A block is defined where strong LD is observed for most pairs.
Strong LD means CI bounds of D' are high.
We approximate using D' >= 0.8 and r² >= 0.2 thresholds.
"""

STRONG_DPRIME = 0.8
MIN_R2 = 0.2
PAIR_COVERAGE = 0.95  # 95% pairs must be strong LD

print("Calling haplotype blocks")

def haplotype_blocks(h, pos, strong_d=0.8, min_r2=0.2, coverage=0.95):
    blocks = []
    start = 0
    M = h.shape[0]

    while start < M:
        end = start
        best_end = start

        while end < M:
            # Compute LD for all pairs in [start,end]
            strong_pairs = 0
            total_pairs = 0

            for i in range(start, end+1):
                for j in range(i+1, end+1):
                    total_pairs += 1
                    _, dprime, r2 = haplo_ld(h[i], h[j])
                    if np.isfinite(dprime) and np.isfinite(r2):
                        if dprime >= strong_d and r2 >= min_r2:
                            strong_pairs += 1

            if total_pairs == 0:
                frac = 1.0
            else:
                frac = strong_pairs / total_pairs

            if frac >= coverage:
                best_end = end
                end += 1
            else:
                break

        blocks.append((start, best_end))
        start = best_end + 1

    return blocks

blocks = haplotype_blocks(h, pos_phase, STRONG_DPRIME, MIN_R2, PAIR_COVERAGE)

block_info = []
for b, (s, e) in enumerate(blocks, start=1):
    block_info.append({
        "Block": b,
        "Start_SNP": s,
        "End_SNP": e,
        "Start_BP": pos_phase[s],
        "End_BP": pos_phase[e],
        "Length_BP": pos_phase[e] - pos_phase[s],
        "Num_SNPs": e - s + 1
    })

block_df = pd.DataFrame(block_info)
block_df.to_csv(f"haplotype_blocks_chr{chrom_num}.csv", index=False)
print(block_df)

plt.figure(figsize=(10,4))
for _, row in block_df.iterrows():
    plt.plot([row["Start_BP"], row["End_BP"]], [row["Block"], row["Block"]], linewidth=6)

plt.xlabel("Genomic position (bp)")
plt.ylabel("Block")
plt.title(f"Haplotype Blocks chr{chrom_num}")
plt.tight_layout()
plt.savefig(f"Haplotype_blocks_chr{chrom_num}.png", dpi=300)
plt.show()

### Step 12 — Haplotype counts and frequencies

import pandas as pd

haplotype_results = []

n_samples = gt_phase.shape[1]

for block_id, (s, e) in enumerate(blocks, start=1):
    block_gt = gt_phase[s:e+1, :, :]

    hap_list = []

    for sample in range(n_samples):
        hap1 = block_gt[:, sample, 0].astype(int)
        hap2 = block_gt[:, sample, 1].astype(int)

        # - HAP1 -
        hap1_code = "HAP_" + "-".join(map(str, hap1))

        hap1_bases = []
        for i, allele in enumerate(hap1):
            if allele == 0:
                hap1_bases.append(ref[s + i])
            elif allele == 1:
                hap1_bases.append(alt[s + i][0])
            else:
                hap1_bases.append("N")

        hap1_seq = "HAP_" + "-".join(hap1_bases)

        # - HAP2 -
        hap2_code = "HAP_" + "-".join(map(str, hap2))

        hap2_bases = []
        for i, allele in enumerate(hap2):
            if allele == 0:
                hap2_bases.append(ref[s + i])
            elif allele == 1:
                hap2_bases.append(alt[s + i][0])
            else:
                hap2_bases.append("N")

        hap2_seq = "HAP_" + "-".join(hap2_bases)

        hap_list.append((hap1_code, hap1_seq))
        hap_list.append((hap2_code, hap2_seq))

    # Count haplotypes
    hap_df = pd.DataFrame(hap_list, columns=["Haplotype", "Haplotype_seq"])

    hap_counts = hap_df.value_counts().reset_index(name="Count")
    hap_counts["Frequency"] = hap_counts["Count"] / len(hap_list)

    for _, row in hap_counts.iterrows():
        haplotype_results.append({
            "Block": block_id,
            "Haplotype": row["Haplotype"],
            "Haplotype_seq": row["Haplotype_seq"],
            "Count": int(row["Count"]),
            "Frequency": row["Frequency"]
        })

# Create dataframe
haplotype_df = pd.DataFrame(haplotype_results)

# Save output
output1 = f"haplotypes_per_block_chr{chrom_num}.csv"
haplotype_df.to_csv(output1, index=False)

print("Haplotype counts complete:", output1)


# Process directly 

df = haplotype_df.copy()

# Convert Frequency
df["Frequency"] = pd.to_numeric(df["Frequency"], errors="coerce")

# Drop invalid
df = df.dropna(subset=["Frequency"])

#Keep max Frequency per Block

df_max = df.loc[df.groupby("Block")["Frequency"].idxmax()]
df_max = df_max.sort_values("Block").reset_index(drop=True)

# Filter Haplotype

def valid_haplotype(x):
    x = str(x)
    
    if x.startswith("HAP_"):
        x = x.replace("HAP_", "")
    
    if x in ["0", "1"]:
        return False
    
    if x.count('0') == 1 or x.count('1') == 1:
        return False
    
    return True

df_filtered = df_max[df_max['Haplotype'].apply(valid_haplotype)]


# Remove unwanted column

df_filtered = df_filtered.drop(columns=["Haplotype"], errors='ignore')

# Save SECOND output

output2 = f"final_filtered_chr{chrom_num}.csv"
df_filtered.to_csv(output2, index=False)

print("Step 2 complete:", output2)
print(df_filtered.head())

###Step 13 — Recombination landscape

print("Computing recombination landscape using haplotype LD...")

window, step = 50_000, 25_000
hotspot_pos, mean_ld = [], []

M = len(pos_phase)

for start in range(pos_phase.min(), pos_phase.max(), step):
    end = start + window
    idx = np.where((pos_phase >= start) & (pos_phase < end))[0]

    if len(idx) < 5:
        continue

    r2_vals = []

    for a in range(len(idx)):
        i = idx[a]
        for b in range(a + 1, len(idx)):
            j = idx[b]

            _, _, r2 = haplo_ld(h[i], h[j])
            if np.isfinite(r2):
                r2_vals.append(r2)

    if len(r2_vals) > 0:
        mean_ld.append(np.mean(r2_vals))
        hotspot_pos.append(start + window / 2)

plt.figure(figsize=(8,5))
plt.plot(hotspot_pos, mean_ld, 'o-')
plt.xlabel("Genomic position (bp)")
plt.ylabel("Mean r² (haplotype LD)")
plt.title(f"Recombination Landscape chr{chrom_num} (LD-based)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"recombination_chr{chrom_num}.png", dpi=300)
plt.close()

print("Recombination landscape plot saved.")


###Step 14 — LD heatmap (r² matrix)

MAX_SNPS_HEATMAP = 300   # change 300 / 500 / 1000 (higher = slower)
M_heat = min(MAX_SNPS_HEATMAP, h.shape[0])

print(f"Computing haplotype LD r² matrix for first {M_heat} SNPs...")

ld_r2 = np.full((M_heat, M_heat), np.nan, dtype=float)

for i in range(M_heat):
    ld_r2[i, i] = 1.0
    for j in range(i + 1, M_heat):
        _, _, r2 = haplo_ld(h[i], h[j])   # haplotype LD
        ld_r2[i, j] = r2
        ld_r2[j, i] = r2

# labels using positions
labels = [f"{pos_phase[i]}" for i in range(M_heat)]

ld_df = pd.DataFrame(ld_r2, index=labels, columns=labels)
ld_df.to_csv(f"LD_Haplotype_matrix_chr{chrom_num}.csv")

print("LD matrix saved: LD_haplotype_matrix.csv")

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(ld_df, cmap="viridis", square=True)
plt.title(f"Haplotype LD Heatmap chr{chrom_num}(r²)")
plt.tight_layout()
plt.savefig(f"LD_haplotype_heatmap_chr{chrom_num}.png", dpi=300)
plt.show()

###Step 15 — SNP density across chromosome

window = 100_000  # 100 kb
bins = np.arange(pos.min(), pos.max()+window, window)
counts, edges = np.histogram(pos, bins=bins)

mid = (edges[:-1] + edges[1:]) / 2

plt.figure(figsize=(10,4))
plt.plot(mid, counts)
plt.xlabel("Genomic position (bp)")
plt.ylabel("SNP count per 100kb")
plt.title(f"SNP density across chromosome chr{chrom_num}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"SNP_density_chr{chrom_num}_100kb.png", dpi=300)
plt.show()

###Step 16 — LD scatter plot (haplotype LD)

plt.figure(figsize=(7,5))
plt.scatter(distances, r2_values, s=5, alpha=0.3)
plt.xlabel("Distance (bp)")
plt.ylabel("r²")
plt.title(f"LD scatter chr{chrom_num} (haplotype LD)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"LD_scatter_pairs_chr{chrom_num}.png", dpi=300)
plt.show()


###Step 17 — Haplotype-block distributions

plt.figure(figsize=(7,5))
plt.hist(block_df["Length_BP"], bins=30)
plt.xlabel("Block length (bp)")
plt.ylabel("Number of blocks")
plt.title(f"Haplotype block length distribution chr{chrom_num}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"block_length_hist_chr{chrom_num}.png", dpi=300)
plt.show()
###14)Haplotype block SNP count distribution
plt.figure(figsize=(7,5))
plt.hist(block_df["Num_SNPs"], bins=30)
plt.xlabel("SNPs per block")
plt.ylabel("Number of blocks")
plt.title(f"Haplotype block SNP count distribution chr{chrom_num}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"block_snp_count_hist_chr{chrom_num}.png", dpi=300)
plt.show()

###Step 18 — Haplotype diversity across chromosome

plt.figure(figsize=(10,4))
lengths = block_df["Length_BP"].values

for _, row in block_df.iterrows():
    plt.plot([row["Start_BP"], row["End_BP"]], [row["Block"], row["Block"]], linewidth=6)

plt.xlabel("Genomic position (bp)")
plt.ylabel("Block ID")
plt.title(f"Haplotype blocks across chromosome chr{chrom_num}")
plt.tight_layout()
plt.savefig(f"Haplotype_blocks_map_chr{chrom_num}.png", dpi=300)
plt.show()

#Haplotype diversity per block

hap_count_per_block = haplotype_df.groupby("Block")["Haplotype"].nunique()

plt.figure(figsize=(8,4))
plt.plot(hap_count_per_block.index, hap_count_per_block.values, marker="o")
plt.xlabel("Block")
plt.ylabel("Unique haplotypes")
plt.title(f"Haplotype diversity per block chr{chrom_num}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"haplotypes_per_block_chr{chrom_num}.png", dpi=300)
plt.show()

#Major haplotype frequency per block


major_freq = haplotype_df.groupby("Block")["Frequency"].max()

plt.figure(figsize=(8,4))
plt.plot(major_freq.index, major_freq.values, marker="o")
plt.xlabel("Block")
plt.ylabel("Major haplotype frequency")
plt.title(f"Major haplotype frequency per block chr{chrom_num}")
plt.ylim(0,1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"major_haplotype_frequency_chr{chrom_num}.png", dpi=300)
plt.show()


###Step 19 — Shannon entropy of haplotypes

entropy_list = []
for block, grp in haplotype_df.groupby("Block"):
    p = grp["Frequency"].values
    H = -(p * np.log(p + 1e-12)).sum()
    entropy_list.append((block, H))

entropy_df = pd.DataFrame(entropy_list, columns=["Block", "ShannonEntropy"])

plt.figure(figsize=(8,4))
plt.plot(entropy_df["Block"], entropy_df["ShannonEntropy"], marker="o")
plt.xlabel("Block")
plt.ylabel("Shannon entropy")
plt.title(f"Haplotype diversity (Shannon entropy) per block chr{chrom_num}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"haplotype_entropy_per_block_chr{chrom_num}.png", dpi=300)
plt.show()


###Step 20 — FAST EXTENDED HAPLOTYPE HOMOZYGOSITY (EHH)


print("Computing FAST Extended Haplotype Homozygosity (EHH)...")

EHH_CUTOFF = 0.05        # stop when EHH decays below this
MAX_WINDOW_SNPS = 500    # max SNPs scanned on each side


def compute_ehh_fast(haps, core_index, mask, pos_array):
    """
    Fast EHH calculation:
    - No haplotype string building
    - Early stopping
    - Window limited
    Complexity: O(W × N)
    """

    M = haps.shape[0]
    core_pos = pos_array[core_index]

    distances = []
    ehh_vals = []

    # scan left and right
    for direction in [-1, 1]:

        same = np.ones(mask.sum(), dtype=np.float32)
        core_alleles = haps[core_index, mask]

        for step in range(1, MAX_WINDOW_SNPS + 1):
            idx = core_index + direction * step
            if idx < 0 or idx >= M:
                break

            current = haps[idx, mask]
            same *= (current == core_alleles)

            ehh = np.mean(same)

            if ehh < EHH_CUTOFF:
                break

            distances.append(abs(pos_array[idx] - core_pos))
            ehh_vals.append(ehh)

    return np.array(distances), np.array(ehh_vals)



# EXAMPLE: PLOT ONE CORE SNP

core = len(h) // 2
core_alleles = h[core]

d_der, ehh_der = compute_ehh_fast(h, core, core_alleles == 1, pos_phase)
d_anc, ehh_anc = compute_ehh_fast(h, core, core_alleles == 0, pos_phase)
plt.figure(figsize=(6,4))
plt.plot(d_der, ehh_der, label="Derived allele")
plt.plot(d_anc, ehh_anc, label="Ancestral allele")
plt.xlabel("Distance from core SNP (bp)")
plt.ylabel("EHH")
plt.title(f"EHH chr{chrom_num}")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"EHH_chr{chrom_num}.png", dpi=300)
plt.show()



###Step 21 — iHS CALCULATION


print("Computing iHS selection scan...")

ihs_raw = []

for i in range(h.shape[0]):

    core_alleles = h[i]

    derived_mask = core_alleles == 1
    ancestral_mask = core_alleles == 0

    # require enough chromosomes
    if derived_mask.sum() < 5 or ancestral_mask.sum() < 5:
        ihs_raw.append(np.nan)
        continue

    d1, e1 = compute_ehh_fast(h, i, derived_mask, pos_phase)
    d2, e2 = compute_ehh_fast(h, i, ancestral_mask, pos_phase)

    if len(d1) < 2 or len(d2) < 2:
        ihs_raw.append(np.nan)
        continue

    i1 = np.trapezoid(e1, d1)
    i2 = np.trapezoid(e2, d2)

    if i1 > 0 and i2 > 0:
        ihs_raw.append(np.log(i1 / i2))
    else:
        ihs_raw.append(np.nan)

ihs_raw = np.array(ihs_raw)



# CORRECT FREQUENCY-BIN STANDARDIZATION

print("Standardizing iHS using frequency bins...")

freq = np.mean(h, axis=1)
bins = np.linspace(0, 1, 21)  # 20 bins

ihs_std = np.full_like(ihs_raw, np.nan)

for b in range(len(bins) - 1):
    mask = (freq >= bins[b]) & (freq < bins[b+1])

    if np.sum(mask) < 10:
        continue

    mean_bin = np.nanmean(ihs_raw[mask])
    std_bin = np.nanstd(ihs_raw[mask])

    if std_bin == 0 or np.isnan(std_bin):
        continue

    ihs_std[mask] = (ihs_raw[mask] - mean_bin) / std_bin



# PLOT iHS


plt.figure(figsize=(10,4))
plt.scatter(pos_phase, ihs_std, s=8)
plt.axhline(2, linestyle='--')
plt.axhline(-2, linestyle='--')
plt.xlabel("Genomic position (bp)")
plt.ylabel("Standardized iHS")
plt.title(f"iHS chr{chrom_num}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"iHS_chr{chrom_num}.png", dpi=300)
plt.show()

print("FAST EHH and frequency-standardized iHS completed.")




# Step 22 — Nucleotide Diversity (π)


print("Calculating nucleotide diversity (π)...")

ac = gt_phase.count_alleles()

# Genome-wide π
pi = allel.sequence_diversity(pos_phase, ac)
print("Genome-wide π:", pi)

# Sliding window π
window_size = 100_000
step = 50_000

pi_vals, pi_windows, _, _ = allel.windowed_diversity(
    pos_phase,
    ac,
    size=window_size,
    step=step
)

# Compute window midpoints correctly
pi_mid = pi_windows.mean(axis=1)

plt.figure(figsize=(10,4))
plt.plot(pi_mid, pi_vals)
plt.title(f"Nucleotide Diversity (π) Across Chromosome chr{chrom_num}")
plt.xlabel("Position (bp)")
plt.ylabel("π")
plt.tight_layout()
plt.savefig(f"nucleotide_diversity_pi_chr{chrom_num}.png", dpi=300)
plt.show()

###Step 23 —  Tajima’s D


print("Calculating Tajima’s D...")

# Genome-wide Tajima's D
tajd = allel.tajima_d(ac)
print("Genome-wide Tajima's D:", tajd)

# Sliding window Tajima's D
taj_vals, taj_windows, _ = allel.windowed_tajima_d(
    pos_phase,
    ac,
    size=window_size,
    step=step
)

taj_mid = taj_windows.mean(axis=1)

plt.figure(figsize=(10,4))
plt.plot(taj_mid, taj_vals)
plt.axhline(0, linestyle='--')
plt.title(f"Tajima’s D Across chr{chrom_num}")
plt.xlabel("Position (bp)")
plt.ylabel("Tajima’s D")
plt.tight_layout()
plt.savefig(f"tajimas_d_chr{chrom_num}.png", dpi=300)
plt.show()

###Step 24 —  FST (Population Differentiation)


print("Calculating FST...")

n_samples = gt_phase.shape[1]
pop1 = list(range(0, n_samples // 2))
pop2 = list(range(n_samples // 2, n_samples))

ac1 = gt_phase.count_alleles(subpop=pop1)
ac2 = gt_phase.count_alleles(subpop=pop2)

# Genome-wide FST
num, den = allel.hudson_fst(ac1, ac2)
fst = np.sum(num) / np.sum(den)
print("Genome-wide FST:", fst)

# Sliding window FST (returns 3 values)
fst_vals, fst_windows, _ = allel.windowed_hudson_fst(
    pos_phase,
    ac1,
    ac2,
    size=window_size,
    step=step
)

# Compute midpoint correctly

fst_mid = fst_windows.mean(axis=1)
plt.figure(figsize=(10,4))
plt.plot(fst_mid, fst_vals)
plt.title(f"FST Across chr{chrom_num}")
plt.xlabel("Position (bp)")
plt.ylabel("FST")
plt.tight_layout()
plt.savefig(f"fst_plot_chr{chrom_num}.png", dpi=300)
plt.show()

print("All done!")