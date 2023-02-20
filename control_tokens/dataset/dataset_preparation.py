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
    return round(avg, 2)


tss_path = f"scores/style_{corpus}_{split}_tgt.txt"
sti_path = f"scores/sti_{corpus}_{split}.txt"
cps_path = f"scores/cps_{corpus}_{split}.txt"

tss_data = read_scores(tss_path)
sti_data = read_scores(sti_path)
cps_data = read_scores(cps_path)

tss_scores = [item[0]["score"] for item in tss_data]
sti_scores = [item[0] for item in sti_data]
cps_scores = [item["scores"][0] for item in cps_data]

tss_avg = read_avg(tss_path)
sti_avg = read_avg(sti_path)
cps_avg = read_avg(cps_path)

src_path = f"{corpus}/{split}.src"
with open(src_path) as f:
    src = [line.strip() for line in f]

src_ct = list()
for i, sentence in enumerate(src):
    if split == "train":
        new_sentence = f"<TSS_{tss_scores[i]}> <STI_{sti_scores[i]}> <CPS_{cps_scores[i]}> {sentence}"
    else:
        new_sentence = f"<TSS_{tss_avg}> <STI_{sti_avg}> <CPS_{cps_avg}> {sentence}"
    src_ct.append(new_sentence)

with open(f"{corpus}_ct/{split}.src", "w") as f:
    f.write("\n".join(src_ct))
