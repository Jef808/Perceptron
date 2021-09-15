#!/usr/bin/env python3

import argparse
import pathlib

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

def parse_line(line):
    s_feats = line.split(',')
    feats = [s_feats[i] for i in [1, 2, -8, -7, -6, -5, -3]]
    feat = {
        "survive": feats[0] == '1',
        "p_class": int(feats[1]),
        "sex": feats[2],
        "age": feats[3],
        "sib": int(feats[4]),
        "parents": int(feats[5]),
        "fare": float(feats[6])
    }
    return feat

def to_float(feat):
    ret = [float(feat["survive"])]
    p_cls = feat["p_class"]
    ret.append(0.0 if p_cls == 1 else (0.5 if p_cls == 2 else 1.0))
    ret.append(0.0 if feat["sex"] == "male" else 1.0)
    age = feat["age"]
    ret.append(float(age) if age else 30.0)
    n_sibs = feat["sib"]
    ret.append(0.0 if n_sibs == 0 else (0.5 if n_sibs == 1 else 1.0))
    n_pars = feat["parents"]
    ret.append(0.0 if n_pars == 0 else (0.5 if n_pars == 1 else (0.75 if n_pars == 2 else 1.0)))
    ret.append(1.0 if feat["fare"] > 55.0 else 0.0)
    return ret

def to_str(feat):
    return ' '.join(map(str, to_float(feat)))

def parse_infile(infile):
    ret = list(dict())
    with open(infile, "r") as file:
        lines = file.readlines()
        for line in lines[1:]:
            ret.append(parse_line(line))
    return ret

def write_outfile(outfile, feats):
    s_feats = map(to_str, feats)
    with open(outfile, "w") as file:
        for s_feat in s_feats:
            file.write(s_feat + '\n')

if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    infile = args.file[0]
    outfile = process_outfile(infile)
    feats = parse_infile(infile)
    write_outfile(outfile, feats)
