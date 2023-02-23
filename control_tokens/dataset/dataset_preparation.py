import sys
import ast

corpus = sys.argv[1]
split = sys.argv[2]


def read_scores(path):
    with open(path) as f:
        data = f.readline()
        scores = ast.literal_eval(data)
    return scores


def read_avg(path):
    with open(path) as f:
        data = f.readline()
        data = f.readline()
        avg = ast.literal_eval(data.split()[-1])
    return round(avg * 20) * 5


def levenshtein_similarity(s1, s2):
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m
    # Initialize the distance matrix
    distance = [[0] * (n + 1) for i in range(m + 1)]
    for i in range(m + 1):
        distance[i][0] = i
    for j in range(n + 1):
        distance[0][j] = j
    # Calculate the minimum edit distance using dynamic programming
    for j in range(1, n + 1):
        for i in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,  # deletion
                distance[i][j - 1] + 1,  # insertion
                distance[i - 1][j - 1] + cost,
            )  # substitution
    # Return the Levenshtein similarity score
    return 1 - distance[m][n] / max(m, n)


def nb_chars(s1, s2):
    return len(s2) / len(s1)


tss_path = f"scores/style_{corpus}_{split}_tgt.txt"
sti_path = f"scores/sti_{corpus}_{split}.txt"
cps_path = f"scores/cps_{corpus}_{split}.txt"

tss_data = read_scores(tss_path)
sti_data = read_scores(sti_path)
cps_data = read_scores(cps_path)

tss_scores = [round(item[0]["score"], 2) for item in tss_data]
sti_scores = [round(item[0], 2) for item in sti_data]
cps_scores = [round(item["scores"][0], 2) for item in cps_data]

tss_avg = read_avg(tss_path)
sti_avg = read_avg(sti_path)
cps_avg = read_avg(cps_path)

src_path = f"{corpus}/{split}.src"
tgt_path = f"{corpus}/{split}.tgt"
with open(src_path) as f:
    src = [line.strip() for line in f]
with open(tgt_path) as f:
    tgt = [line.strip() for line in f]

lev = list()
nb = list()
if split == "train":
    for src_sen, tgt_sen in zip(src, tgt):
        lev.append(round(levenshtein_similarity(src_sen, tgt_sen) * 20) * 5)
        nb.append(round(nb_chars(src_sen, tgt_sen) * 20) * 5)
else:
    with open(f"{corpus}/test.src") as f:
        src_test = [line.strip() for line in f]
    with open(f"{corpus}/test.tgt") as f:
        tgt_test = [line.strip() for line in f]
    with open(f"{corpus}/valid.src") as f:
        src_valid = [line.strip() for line in f]
    with open(f"{corpus}/valid.tgt") as f:
        tgt_valid = [line.strip() for line in f]

    for i, src_sen in enumerate(src_test):
        x = ast.literal_eval(tgt_test[i])
        for tgt_sen in x:
            lev.append(levenshtein_similarity(src_sen, tgt_sen))
            nb.append(nb_chars(src_sen, tgt_sen))

    for i, src_sen in enumerate(src_valid):
        x = ast.literal_eval(tgt_valid[i])
        for tgt_sen in x:
            lev.append(levenshtein_similarity(src_sen, tgt_sen))
            nb.append(nb_chars(src_sen, tgt_sen))

    lev_avg = round(sum(lev) / len(lev) * 20) * 5
    nb_avg = round(sum(nb) / len(nb) * 20) * 5


src_ct = list()
for i, sentence in enumerate(src):
    if split == "train":
        new_sentence = f"<NbChars_{nb[i]}> <LevSim_{lev[i]}> {sentence}"
    else:
        new_sentence = f"<NbChars_{nb_avg}> <LevSim_{lev_avg}> {sentence}"
    src_ct.append(new_sentence)

with open(f"{corpus}_ct/{split}.src", "w") as f:
    f.write("\n".join(src_ct))
