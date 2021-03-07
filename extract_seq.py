import os
import re

full_path = os.path.realpath(__file__)
os.chdir(os.path.dirname(full_path))

print(f"Change CWD to: {os.path.dirname(full_path)}")

def extract_seq(file_path, dir_path='seq', seq_length=510):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    nseq = 0
    nsmp = 0
    data = re.split(
        r'(^>.*)', ''.join(open(file_path).readlines()), flags=re.M)
    for i in range(2, len(data), 2):
        fid = data[i-1][1:].split('|')[0]
        nseq = nseq + 1
        fasta = list(data[i].replace('\n', '').replace('\x1a', ''))
        seq = [' '.join(fasta[j:j + seq_length])
               for j in range(0, len(fasta) + 1, seq_length)]
        nsmp = nsmp + len(seq)
        ffas = open(f"{dir_path}/{fid}.seq", "w")
        ffas.write('\n'.join(seq))
    print(f"Number of sequences: {nseq}")
    print(f"Number of samples: {nsmp}")

extract_seq('non.cv.txt', 'cv_neg')
extract_seq('enhancer.cv.txt', 'cv_pos')

extract_seq('non.ind.txt', 'ind_neg')
extract_seq('enhancer.ind.txt', 'ind_pos')
