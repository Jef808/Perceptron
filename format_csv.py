#!/usr/bin/env python3

import argparse
import pathlib
import sys

def file_t(arg):
    f = pathlib.Path(arg)
    if not f.is_file():
        raise argparse.ArgumentTypeError(f"Invalid file: {f}")
    return f

def process_outfile(infile):
    return infile.with_suffix('.txt')

def init_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument("file",
                        type=file_t,
                        nargs=1)
    return parser

def parse_line(line, training=True):
    s_feats = line.split(',')
    if training:
        feats = [s_feats[i] for i in [1, 2, -8, -7, -6, -5, -3]]
        feat = { "survive": s_feats[1] == '1' } if training else dict()
    else:
        feats = [s_feats[i] for i in [0, 1, -8, -7, -6, -5, -3]]
        feat = { "p_id": int(feats[0]) }
    feat.update({
        "p_class": int(feats[1]),
        "sex": feats[2],
        "age": feats[3],
        "sib": int(feats[4]),
        "parents": int(feats[5]),
        "fare": feats[6]
    })
    return feat

def to_float(feat, training=True):
    if training:
        ret = [float(feat["survive"])]
    else:
        ret = [feat["p_id"]]
    p_cls = feat["p_class"]
    ret.append(-1.0 if p_cls == 1 else (0.0 if p_cls == 2 else 1.0))
    ret.append(-1.0 if feat["sex"] == "male" else 1.0)
    age = feat["age"]
    ret.append((float(age) - 30.0) / 60.0 if age else 0.0)
    n_sibs = feat["sib"]
    ret.append(-1.0 if n_sibs == 0 else (0.0 if n_sibs == 1 else 1.0))
    n_pars = feat["parents"]
    ret.append(-1.0 if n_pars == 0 else (0.0 if n_pars == 1 else (0.5 if n_pars == 2 else 1.0)))
    fare = feat["fare"]
    if not fare:
        fare = 0.0
    else:
        fare = float(fare)
    ret.append(1.0 if fare > 55.0 else -1.0)
    return ret

def to_str(feat, training):
    return ' '.join(map(str, to_float(feat, training)))

def parse_infile(infile, training=True):
    print("Training = ", training, file=sys.stderr)
    ret = list(dict())
    with open(infile, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:
            ret.append(parse_line(line, training))
    return ret

def write_outfile(outfile, feats, training):
    s_feats = map(lambda s: to_str(s, training), feats)
    with open(outfile, "w") as file:
        for s_feat in s_feats:
            file.write(s_feat + '\n')

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    infile = args.file[0]
    outfile = process_outfile(infile)
    print("name: ", infile.stem, file=sys.stderr)
    training = infile.stem == "train"
    feats = parse_infile(infile, training)
    write_outfile(outfile, feats, training)
