import ray
import time
import sys
import re
import zlib
import random
from statistics import mean, stdev
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit.Chem import SaltRemover

# 1. MAXIMIZE RECURSION
sys.setrecursionlimit(50000)

# ---------------------------------------------------------
# 1. RANDOM BRANCHED ALKANE GENERATOR (The Control)
# ---------------------------------------------------------
def random_branched_alkane(n_carbons):
    if n_carbons < 1: return ""
    mol = Chem.RWMol()
    idx0 = mol.AddAtom(Chem.Atom(6))
    for i in range(1, n_carbons):
        targets = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6 and a.GetDegree() < 4]
        if not targets: break
        attach_to = random.choice(targets)
        new_idx = mol.AddAtom(Chem.Atom(6))
        mol.AddBond(attach_to, new_idx, Chem.BondType.SINGLE)
    mol = mol.GetMol()
    Chem.SanitizeMol(mol)
    mol = Chem.AddHs(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def generate_alkane_control_set(n=20, size_range=(20,40)):
    alkanes = []
    while len(alkanes) < n:
        c_count = random.randint(*size_range)
        smi = random_branched_alkane(c_count)
        if smi and Chem.MolFromSmiles(smi):
            alkanes.append(smi)
    return alkanes

# ---------------------------------------------------------
# 2. THE DATASET (Reconstructed)
# ---------------------------------------------------------
# Taxol (Paclitaxel)
TAXOL = "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C"
# ATP
ATP = "Nc1ncnc2c1ncn2[C@@H]3O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]3O"
# Vancomycin (Another Complex Natural Product)
VANCOMYCIN = "CC1C(C(CC(O1)OC2C(C(C(OC2OC3=C4C=C5C=C3OC6=C(C=C(C=C6)C(C(C(=O)NC(C(=O)NC5C(=O)NC7C8=CC(=C(C=C8)O)C9=C(C=C(C=C9O)O)C(NC(=O)C(C(C1=CC(=C(O4)C=C1)Cl)O)NC7=O)C(=O)O)CC(=O)N)CC(=O)O)O)O)Cl)CO)O)O)(C)N"

TARGETS = {
    "Dead_Simple": [
        "CO", "CCO", "CCC", "CCCC", "CC(C)C", "c1ccccc1", "C1CC1", "C1CCCC1",
        "CC=O", "CC(=O)O", "CCOC", "CN", "CCl", "CBr", "CI", "CS", "C#N", "C=O",
        "CC#N", "C1=CC=NC=C1", "CCCO", "CC(C)O", "CCOCC", "CC(C)CCO", "C1CCCCC1",
        "c1ccncc1", "c1ccsc1", "c1ccoc1", "CC(=O)OC", "CCOC(=O)C", "CCN", "CCS",
        "CS(=O)C", "CC(=O)Cl", "CC(=O)Br", "C1CO1", "C1CS1", "C1CN1", "CC(=N)N",
        "CC(=O)OC(C)C", "CC(=O)N(C)C", "CC(C)=O", "C=CC", "C=C", "C#CC", "C1=CCC=C1",
        "CC(=C)C", "C=C(C)C", "CC#CC", "COc1ccccc1", "Cc1ccccc1", "c1ccccc1O",
        "c1ccccc1N", "c1ccccc1Cl", "c1ccccc1C=O", "c1ccccc1C(=O)O", "C1=CC=CC=N1",
        "C1=CN=CC=N1", "C1=NC=NC=N1", "O=C1C=CC(=O)N1", "O=C1NC(=O)NC1=O",
        "C1CCC(=O)NC1=O", "CC1=CC=CC=C1O", "CC1=CC=CC=C1N", "CC1=CC=CC=C1C(=O)O",
        "CC(=O)NC1=CC=CC=C1", "CC(C)CC(=O)O", "CC(C)CCN", "CC(C)CO", "CCCCCC",
        "CCCCCCCC", "CCCCCCCCCC", "C1CCCCCC1", "C1CCCCCCCC1", "c1ccccc1C",
        "c1ccccc1CC", "c1ccccc1CCC", "c1ccccc1CO", "c1ccccc1CN", "c1ccccc1C#N",
        "c1ccccc1C(=O)OC", "c1ccccc1C(=O)Cl", "c1ccccc1C(=O)Br", "c1ccccc1C(=O)I",
        "c1ccccc1C(=O)OC1CCCCC1", "c1ccccc1C(=O)N(C)C", "c1ccccc1C(=O)NC1CCCCC1",
        "c1ccccc1C(=O)NC1CC1", "c1ccccc1C(=O)NC1CCOCC1", "c1ccccc1C(=O)NC1CCCC1",
        "c1ccccc1C(=O)NC1CCCC1=O", "c1ccccc1C(=O)NC1CCCCC1=O", "c1ccccc1C(=O)NC1CCCCC1O",
        "c1ccccc1C(=O)NC1CCCCC1N", "c1ccccc1C(=O)NC1CCCCC1Cl", "c1ccccc1C(=O)NC1CCCCC1Br",
        "c1ccccc1C(=O)NC1CCCCC1I", "c1ccccc1C(=O)NC1CCCCC1C(=O)O",
        "c1ccccc1C(=O)NC1CCCCC1C(=O)OC", "c1ccccc1C(=O)NC1CCCCC1C(=O)Cl",
        "c1ccccc1C(=O)NC1CCCCC1C(=O)Br", "c1ccccc1C(=O)NC1CCCCC1C(=O)I",
        "c1ccccc1C(=O)NC1CCCCC1C(=O)NC1CCCCC1", "c1ccccc1C(=O)NC1CCCCC1C(=O)NC1CC1",
        "c1ccccc1C(=O)NC1CCCCC1C(=O)NC1CCOCC1", "c1ccccc1C(=O)NC1CCCCC1C(=O)NC1CCCC1",
        "c1ccccc1C(=O)NC1CCCCC1C(=O)NC1CCCC1=O",
    ],
    "Bio_Basic": [
        # amino acids (20 standard + some derivatives)
        "N[C@@H](C)C(=O)O", "N[C@@H](CO)C(=O)O", "N[C@@H](Cc1ccccc1)C(=O)O",
        "N[C@@H](CCCNC(=N)N)C(=O)O", "N[C@@H](Cc1c[nH]cn1)C(=O)O",
        "N[C@@H](Cc1ccc(O)cc1)C(=O)O", "N[C@@H](CC(N)=O)C(=O)O",
        "N[C@@H](CS)C(=O)O", "N[C@@H](CC(C)C)C(=O)O", "N[C@@H](C(C)C)C(=O)O",
        "N[C@@H](CC(=O)O)C(=O)O", "N[C@@H](CCCCN)C(=O)O",
        "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O", "N[C@@H](Cc1cnc[nH]1)C(=O)O",
        # sugars (monosaccharides)
        "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O",  # glucose
        "C([C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O", # galactose
        "C([C@@H]1[C@H]([C@@H]([C@H](CO1)O)O)O)O",         # fructose (furanose)
        "C([C@@H]1[C@@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)CO", # mannose
        "C1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O",           # ribose (furanose)
        "C1[C@H]([C@@H]([C@H]([C@H](O1)CO)O)O)O",          # deoxyribose
        # nucleotides / nucleosides
        "Nc1ncnc2c1ncn2[C@@H]3O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]3O", # ATP
        "CC(=O)N[C@@H]1[C@H](O)O[C@H](CO)[C@@H](O)[C@@H]1O", # N-acetylglucosamine
        "C1=CC(=O)NC(=O)C=C1",      # uracil
        "O=C1C=CN(C2O[C@H](CO)[C@@H](O)[C@H]2O)C(=O)N1", # cytidine
        "O=C1NC(=O)C=C1",           # thymine
        "C1=NC2=C(N1)C(=O)NC(=O)N2", # xanthine
        # lipids (fatty acids)
        "CCCCCCCCCCCCCCCC(=O)O",  # palmitic acid
        "CCCCCCCCCCCCCCCCCC(=O)O", # stearic acid
        "CCCCCCCCCCCCCCCCCCCC(=O)O", # arachidic acid
        "CCCC/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O", # arachidonic acid
        "CCCCCCCC/C=C\\CCCCCCCC(=O)O", # oleic acid
        "CCCCCCCC/C=C\\C/C=C\\CCCCCCCC(=O)O", # linoleic acid
        "CC/C=C\\C/C=C\\C/C=C\\C/C=C\\C/C=C\\CCCC(=O)O", # EPA
        # vitamins / cofactors
        "CC1=C(C=C(C=C1)C)CC2=C(C(=O)C(=C(O2)C)C=CC(=C(C)C(=O)C(=C(C)C)C(=O)C(C(C)C)O)C)O", # vitamin E
        "CC(C)CCCC(C)CCCC(C)CCCC1(C)CCC2(C)CCC3(C)C(=CC(=O)C4C3(C)CCC(O)C4(C)C)C2C1", # vitamin D3
        "CC1=NC2=C(C(=O)N1)C(=O)NC(=O)N2", # riboflavin (simplified)
        "C1=NC(=C2C(=O)NC(N)=NC2=N1)C3=CC=CC=C3", # folic acid (simplified)
        # neurotransmitters / hormones
        "C1=CC(=C(C=C1C[C@@H](C(=O)O)N)O)O", # L-DOPA
        "C1=CC(=CC=C1CCN)O", # dopamine
        "C1=CC(=CC=C1C[C@H](N)C(=O)O)O", # tyrosine
        "C[C@@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O", # testosterone
        "C[C@]12CC[C@H]3[C@@H](CC[C@@H]4C[C@@H](O)CC[C@]34C)[C@@H]1CC[C@H]2C(=O)CO", # cortisone
        "C[C@H](CCCC(C)C)[C@H]1CC[C@@]2([H])[C@]3([H])CC=C4C[C@@H](O)CC[C@]4(C)[C@@]3([H])CC[C@]12C", # cholesterol
        # other bioactives
        "CC(=O)OC[C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O", # 1-O-acetylglucose
        "CC(=O)O[C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO", # glucose pentaacetate
        "CC(C)CC1=NC2=C(C(=O)N1)C(=O)NC(=O)N2", # riboflavin simplified
        "COc1cc2c(c(OC)c1OC)-c1ccc(=O)c(OC)cc1[C@@H](NC(C)=O)CC2", # colchicine
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", # caffeine
        "O=C1C=CN(C2O[C@H](CO)[C@@H](O)[C@H]2O)C(=O)N1", # cytidine (again)
        # Fill to 100 with more standard biochemicals:
        "C[C@@H](O)[C@@H](CO)N", # threonine
        "N[C@@H](Cc1ccc(I)cc1)C(=O)O", "N[C@@H](Cc1cc(Br)c(O)c(Br)c1)C(=O)O",
        "N[C@@H](Cc1cc(F)c(O)c(F)c1)C(=O)O", "N[C@@H](Cc1cc(Cl)c(O)c(Cl)c1)C(=O)O",
        "CC(=O)N[C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO", # N-acetylglucosamine
        "CC(=O)N[C@@H]1[C@H](O)[C@@H](O)[C@@H](O)[C@H]1O", # N-acetylmannosamine
        "CC(=O)N[C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@@H]1O", # N-acetylgalactosamine
        "CCCCCCCCCCCC(=O)O", "CCCCCCCCCC(=O)O", "CCCCCCCC(=O)O", "CCCC(=O)O",
        "CCCCCCCCCCCCCCCCCCCCCC(=O)O", "CC/C=C\\C/C=C\\C/C=C\\CCCCCC(=O)O",
        "CCCCCC/C=C\\C/C=C\\CCCC(=O)O",
        # peptides (di- and tripeptides)
        "CC(C)C[C@H](N)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",
        "CC(C)C[C@H](N)C(=O)N[C@@H](C)C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",
        "CC(C)C[C@H](N)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",
        "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)CO", # sorbitol
        "C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)C(=O)O", # gluconic acid
        "C1=NC2=C(C(=O)N1)C(=O)NC(=O)N2", # xanthine (repeat OK)
        "CC(=O)N[C@@H]1[C@H](O)[C@@H](O)[C@H](O)[C@H](O1)CO", # N-acetylglucosamine
        "CC(C)C[C@H](N)C(=O)O", "N[C@@H](CCCNC(=N)N)C(=O)O",
        "N[C@@H](CC1=CC=CC=C1)C(=O)O", "N[C@@H](CC1=CNC=N1)C(=O)O",
        "N[C@@H](CC(=O)O)C(=O)O", "N[C@@H](CCC(=O)O)C(=O)O",
        "N[C@@H](CC(N)=O)C(=O)O", "N[C@@H](CCC(N)=O)C(=O)O",
        "N[C@@H](CS)C(=O)O", "N[C@@H](CO)C(=O)O",
        "N[C@@H](C(C)C)C(=O)O", "N[C@@H](C(C)CC)C(=O)O",
        "N[C@@H](CC1=CC=C(O)C=C1)C(=O)O", "N[C@@H](CC1=CNC2=CC=CC=C12)C(=O)O",
        "N[C@@H](CCCCN)C(=O)O", "N[C@@H](C)C(=O)O",
        "N1=CN=C2N=CNC2=C1", "O=C1N=C(N)NC=C1", "O=C1NC(=O)NC=C1",
        "O=C1NC(=O)C=C1", "C1=CN=C(N)NC1=O", "CC1=CN=C(N)NC1=O", "CC1=CNC(=O)NC1=O"
    ],
    "Complex_Natural_Products": [
        "CC1=C2[C@H](C(=O)[C@@]3([C@H](C[C@@H]4[C@]([C@H]3[C@@H]([C@@](C2(C)C)(C[C@@H]1OC(=O)[C@@H]([C@H](C5=CC=CC=C5)NC(=O)C6=CC=CC=C6)O)O)OC(=O)C7=CC=CC=C7)(CO4)OC(=O)C)O)C)OC(=O)C",
        "CO[C@H]1C[C@H]2CC[C@@H](C)[C@@](O)(O2)C(=O)C(=O)N3CC[C@H]4[C@@H]5[C@H](O)C[C@@H](C[C@H]5C(=O)[C@H](C)/C=C(C)/C[C@@H](C)C(=O)C[C@@H]1OC)OC(=O)[C@@H]4[C@H](C)C3",
        "C1=CC=C2C(=C1)C3=C4[C@H]5[C@@H]6[C@H](C[C@@H]7N(C6)CC[C@@]57CO4)C(=O)N3C2",
        "CC1([C@@H](N2[C@H](S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C",
        "CC1=C(N2[C@@H]([C@@H](C2=O)NC(=O)CCCCCN)SC1)C(=O)O",
        "CC[C@H]1OC(=O)[C@H](C)[C@@H](O)[C@H]1O[C@H]2[C@@H](O)[C@@H](N(C)C)C[C@H](O[C@@H]3[C@H](O)[C@H](O[C@@H]4O[C@H](C)[C@@H](O)[C@H](O)[C@@H]4O)[C@H](O3)C)O2",
        "C1=CC(=CC=C1[C@H]2C(=O)N[C@H](C(=O)N[C@H]3C(=O)N[C@H](C(=O)N[C@H](C(=O)N[C@H]2C(=O)N3)C4=CC(=C(C=C4)O)Cl)C5=CC(=C(C=C5)O)O)C6=CC(=C(C=C6)O)Cl)O",
        "COC1=CC2=C(C=CN=C2C=C1)[C@H](O)[C@@H]3C[C@@H]4CCN3C[C@@H]4C=C",
        "CN1CC[C@]23[C@@H]4Oc5c2c(C[C@@H]1[C@@H]3C=C[C@@H]4O)ccc5O",
        "COC1=C(OC)C=C(C=C1)C(=O)O[C@@H]2[C@H]3[C@@H]4N(CC3)CC[C@@]2([C@@H]4OC)OC(=O)c5cc(OC)c(OC)c(OC)c5",
        "C[C@H](CCCC(C)C)[C@H]1CC[C@@]2([H])[C@]3([H])CC=C4C[C@@H](O)CC[C@]4(C)[C@@]3([H])CC[C@]12C",
        "C[C@]12CC[C@H]3[C@@H](CC[C@@H]4C[C@@H](O)CC[C@]34C)[C@@H]1CC[C@H]2C(=O)CO",
        "C[C@]12CC[C@H]3[C@@H](CCC4=CC(=O)CC[C@]34C)[C@@H]1CC[C@@H]2O",
        "C[C@H]1/C=C/C=C/C=C/C=C/C=C/C=C/C=C/[C@@H](C[C@H]2[C@H]([C@H](C[C@@](O2)(C[C@H](O)[C@@H](O)[C@H](O)[C@H](C)/C=C/C(=O)O)O)O)O)OC(=O)C[C@H](O)[C@@H](C(=O)O)N1",
        
        "CC(=O)O[C@H]1C[C@@]2(C)[C@@H]3CC[C@@H]4[C@@H](CC[C@]4(C)[C@@H]3[C@H](O)[C@H]2[C@@H]1C)C(=O)OC",  # 21 – Forskolin
        "CC1=C2[C@@H]3[C@@H]4[C@H]5[C@@H](C[C@@H]([C@@H]5O)O)[C@H](O)C[C@H]4[C@@](C)(C[C@@H]3O)OC2=CC(=O)O1",  # 22 – Artemisinin
        "COC1=CC2=C(C=C1)N(C=C2)C(=O)C3=CC=CC=C3",  # 23 – Melatonin
        "CC1=CC(=O)OC2=C1C3=C(C(=O)OC2)C4=C(C=C3O)OC(=O)C5=C4C=CC=C5",  # 24 – Dicoumarol
        "CC1=CC(=O)OC2=C1C=CC(=C2)O",  # 25 – Psoralen
        "COC1=CC2=C(C=C1)C(=O)C3=C(C2=O)C=CC=C3",  # 26 – Anthraquinone
        "CC1=C2C(=O)C3=C(C(=O)C2=CC=C1)C=CC=C3",  # 27 – Anthraquinone derivative
        "CC1=CC(=O)C2=C(C1=O)C=CC=C2",  # 28 – Naphthoquinone (menadione)
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3",  # 29 – Anthraquinone
        "CC1=CC(=O)OC2=C1C=CC3=C2C=CC=C3",  # 30 – Coumarin derivative
        "COC1=CC2=C(C=C1)C3=C(C=C2)C(=O)OC3",  # 31 – Flavonoid core
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC",  # 32 – Methoxypsoralen
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC(=C3)O",  # 33 – Hydroxyanthraquinone
        "CC1=CC(=O)OC2=C1C=CC(=C2)Cl",  # 34 – Chloropsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)Br",  # 35 – Bromopsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)I",  # 36 – Iodopsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)C",  # 37 – Methylpsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)C(=O)O",  # 38 – Carboxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)C(=O)OC",  # 39 – Methoxycarbonylpsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)C(=O)N(C)C",  # 40 – Dimethylcarbamoylpsoralen
        # 41-60
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC(C)C",  # 41 – Isopropylpsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OCC(C)C",  # 42 – Isobutoxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCC3",  # 43 – Cyclohexyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCC3",  # 44 – Cyclopentyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCC3",  # 45 – Cyclobutoxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CC3",  # 46 – Cyclopropoxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCC3",  # 47 – Cycloheptyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCC3",  # 48 – Cyclooctyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCC3",  # 49 – Cyclononyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCC3",  # 50 – Cyclodecyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCCC3",  # 51 – Cycloundecyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCCCC3",  # 52 – Cyclododecyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCCCCC3",  # 53 – Cyclotridecyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCCCCCC3",  # 54 – Cyclotetradecyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCCCCCCC3",  # 55 – Cyclopentadecyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCCCCCCCC3",  # 56 – Cyclohexadecyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCCCCCCCCC3",  # 57 – Cycloheptadecyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCCCCCCCCCC3",  # 58 – Cyclooctadecyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCCCCCCCCCCC3",  # 59 – Cyclononadecyloxypsoralen
        "CC1=CC(=O)OC2=C1C=CC(=C2)OC3CCCCCCCCCCCCCCCCCCC3",  # 60 – Cycloeicosyloxypsoralen
        # 61-80
        "CC1=CC2=C(C=C1O)OC(C3=CC=C(O)C=C3)=CC2=O",  # 61 – Genistein
        "CC(=CCC1=C(O)C=C(O)C=C1)C2=C(O)C=C(O)C=C2",  # 62 – Resveratrol
        "COC1=C(OC)C=C(C=C1)C=CC2=CC(=O)C3=C(C=C(O)C=C3O)O2",  # 63 – Hesperetin
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3O",  # 64 – 1-Hydroxyanthraquinone
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3OC",  # 65 – 1-Methoxyanthraquinone
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C",  # 66 – 1-Methylanthraquinone
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3Cl",  # 67 – 1-Chloroanthraquinone
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3Br",  # 68 – 1-Bromoanthraquinone
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3I",  # 69 – 1-Iodoanthraquinone
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)O",  # 70 – 1-Carboxyanthraquinone
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)OC",  # 71 – Methyl 1-anthraquinonecarboxylate
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)N(C)C",  # 72 – N,N-Dimethyl-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NC1CCCCC1",  # 73 – N-Cyclohexyl-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NC1CC1",  # 74 – N-Cyclopropyl-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NC1CCOCC1",  # 75 – N-(4-Tetrahydropyranyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NC1CCCC1",  # 76 – N-Cyclopentyl-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NC1CCCC1=O",  # 77 – N-(2-Oxocyclopentyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NC1CCCCC1=O",  # 78 – N-(2-Oxocyclohexyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NC1CCCCC1O",  # 79 – N-(2-Hydroxycyclohexyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NC1CCCCC1N",  # 80 – N-(2-Aminocyclohexyl)-1-anthraquinonecarboxamide
        # 81-100
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCO",  # 81 – N-(2-Hydroxyethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCOC",  # 82 – N-(2-Methoxyethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCCl",  # 83 – N-(2-Chloroethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCBr",  # 84 – N-(2-Bromoethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCI",   # 85 – N-(2-Iodoethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCN",   # 86 – N-(2-Aminoethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCN(C)C",  # 87 – N-(2-Dimethylaminoethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCN2CCOCC2",  # 88 – N-(2-Morpholinoethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCN2CCCCC2",  # 89 – N-(2-Piperidinoethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCN2CCCC2",   # 90 – N-(2-Pyrrolidinoethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCC1CC1",  # 91 – N-(2-Cyclopropylethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCC1CCCC1",  # 92 – N-(2-Cyclopentylethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCC1CCCCC1",  # 93 – N-(2-Cyclohexylethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCc1ccccc1",  # 94 – N-(2-Phenylethyl)-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCc1ccc(O)cc1",  # 95 – N-[2-(4-Hydroxyphenyl)ethyl]-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCc1ccc(OC)cc1",  # 96 – N-[2-(4-Methoxyphenyl)ethyl]-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCc1ccc(Cl)cc1",  # 97 – N-[2-(4-Chlorophenyl)ethyl]-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCc1ccc(Br)cc1",  # 98 – N-[2-(4-Bromophenyl)ethyl]-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCc1ccc(I)cc1",   # 99 – N-[2-(4-Iodophenyl)ethyl]-1-anthraquinonecarboxamide
        "CC1=CC(=O)C2=C(C1=O)C(=O)C3=C(C2=O)C=CC=C3C(=O)NCCc1ccc([N+](=O)[O-])cc1",  # 100 – N-[2-(4-Nitrophenyl)ethyl]-1-anthraquinonecarboxamide
    ],
}

ALKANE_CONTROL = generate_alkane_control_set(100)
TARGETS["Random_Alkanes"] = ALKANE_CONTROL

# ---------------------------------------------------------
# 3. LOGIC ENGINE (Recursive MA + LZW)
# ---------------------------------------------------------
# Initialize Stripper once
remover = SaltRemover.SaltRemover(defnData="[Cl,Br,I,Na,K,Ca,Li]")

def get_cuts(smiles):
    # 1. INITIAL SANITIZATION
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return set()
        
        # Strip Salts (Crucial for ZINC/PubChem data)
        mol = remover.StripMol(mol)
        
        # Remove Stereochemistry if parsing fails (Optional fallback)
        # mol = Chem.RemoveHs(mol)
    except:
        return set()

    # 2. BRICS FRAGMENTATION
    try:
        frag_mol = BRICS.BreakBRICSBonds(mol)
        pieces = Chem.GetMolFrags(frag_mol, asMols=True)
    except:
        # If BRICS fails, it's likely a fundamental block
        return set()

    if len(pieces) <= 1: return set()

    cuts = set()
    for p in pieces:
        try:
            # 3. MICROSURGERY (The H-Cap Fix)
            # We convert to SMILES with dummy labels [1*], [2*]...
            temp_smi = Chem.MolToSmiles(p, isomericSmiles=True)
            
            # Regex: Replace [16*] or [*] with [H] to heal the valence
            # This turns a "Broken Phenyl" into "Benzene"
            clean_smi = re.sub(r'\[\d*\*?\]', '[H]', temp_smi)
            
            # 4. RE-VALIDATION
            clean_mol = Chem.MolFromSmiles(clean_smi)
            if clean_mol:
                # Canonicalize
                final_smi = Chem.MolToSmiles(clean_mol, isomericSmiles=True)
                cuts.add(final_smi)
        except:
            continue
            
    return cuts

def assembly_index(smiles, memo=None, depth=0, max_depth=20):
    if memo is None: memo = {}
    if smiles in memo: return memo[smiles]
    if depth > max_depth: return 1 
    
    cuts = get_cuts(smiles)
    if not cuts:
        memo[smiles] = 1
        return 1
    
    # Recursive Sum (The Walker Equation)
    total = 0
    for frag in cuts:
        total += assembly_index(frag, memo, depth+1, max_depth)
    
    memo[smiles] = total
    return total

def lzw_per_heavy_atom(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return 0.0
    heavy = mol.GetNumHeavyAtoms()
    if heavy == 0: return 0.0
    compressed = zlib.compress(smiles.encode('utf-8'))
    return len(compressed) / heavy

# ---------------------------------------------------------
# 4. DIAGNOSTIC WORKER (Patched)
# ---------------------------------------------------------
@ray.remote
class MetricWorker:
    def compute(self, smiles):
        try:
            # 1. Pre-Flight Check
            mol = Chem.MolFromSmiles(smiles)
            if not mol: 
                return ("FAIL_PARSE", None, smiles)
            
            # 2. The Science
            lz = lzw_per_heavy_atom(smiles)
            ma = assembly_index(smiles)
            
            # 3. Post-Flight Check (Zero Assembly is impossible for matter)
            if ma < 1: 
                return ("FAIL_MATH", None, smiles)
                
            return ("SUCCESS", (lz, ma), smiles)

        except Exception as e:
            # Capture the specific crash reason
            return (f"FAIL_CRASH: {str(e)}", None, smiles)

def process_group(name, smiles_list, workers):
    futures = [workers[i % len(workers)].compute.remote(s) for i, s in enumerate(smiles_list)]
    results = ray.get(futures)
    
    # --- THE SORTING HAT ---
    clean_data = []
    error_log = []
    
    for status, data, smi in results:
        if status == "SUCCESS":
            clean_data.append(data) # (lz, ma)
        else:
            error_log.append((status, smi))
    
    # --- REPORTING ---
    success_rate = (len(clean_data) / len(results)) * 100
    print(f"\n{name} | N={len(results)} | Success: {success_rate:.1f}%")
    
    if error_log:
        print(f"  ⚠️  Errors: {len(error_log)} (Top reason: {error_log[0][0]})")
    
    if not clean_data: return [], []

    lzw_vals = [d[0] for d in clean_data]
    ma_vals  = [d[1] for d in clean_data]
    
    print(f"  LZW: {mean(lzw_vals):.4f} | MA: {mean(ma_vals):.2f}")
    return lzw_vals, ma_vals

# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------
def run_vindication():
    print("🚀 GOD-STACK VINDICATION: TRUE ASSEMBLY INDEX")
    ray.init(address="auto")
    resources = ray.cluster_resources()
    print(f"✅ GRID: {int(resources.get('CPU', 1))} Cores")
    
    workers = [MetricWorker.remote() for _ in range(int(resources.get("CPU", 1)))]
    all_results = {}

    for name, mols in TARGETS.items():
        lz, ma = process_group(name, mols, workers)
        if lz: all_results[name] = (lz, ma)

    print("\n" + "="*80)
    print(f"{'GROUP':<25} {'LZW/HvyAt':<15} {'ASSEMBLY INDEX (MA)':<20}")
    print("-" * 80)
    
    # Sort for dramatic effect (Low MA to High MA)
    sorted_groups = sorted(all_results.items(), key=lambda x: mean(x[1][1]))
    
    for name, (lz, ma) in sorted_groups:
        print(f"{name:<25} {mean(lz):.4f} ± {stdev(lz):.4f}   {mean(ma):.2f} ± {stdev(ma):.2f}")
    print("="*80)

if __name__ == "__main__":
    run_vindication()
