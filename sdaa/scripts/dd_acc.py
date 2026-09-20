"""[辅助脚本] 精度评测：预测 pose 相对晶体配体的对称校正 RMSD。

RMSD 直接复用原仓 `utils/molecules_utils.get_symmetry_rmsd`（内部为 spyrmsd），
不另写实现。评测口径与 DiffDock 论文一致：按 confidence 排序，
Top-1 RMSD ≤ 2 Å 记为成功；另外统计 10 个 pose 中 RMSD < 2 Å 的个数。

Usage (inside container, cwd = 仓库根):
    python dd_acc.py --ref examples/1a46_ligand.sdf \
                     --results <dir1> [dir2 ...] [--json out.json]
"""
import argparse
import json
import os
import re
import sys

import numpy as np
from rdkit import Chem

sys.path.insert(0, os.getcwd())  # 保证能 import 到仓库内的 utils

from utils.molecules_utils import get_symmetry_rmsd

NAME_RE = re.compile(r"^rank(\d+)_confidence(-?\d+\.\d+)\.sdf$")


def load_pos(path, remove_hs=True):
    mol = Chem.MolFromMolFile(path, removeHs=remove_hs)
    if mol is None:
        return None, None
    return mol, mol.GetConformer().GetPositions()


def eval_one(ref_path, pose_dir, cutoff):
    """对一个复合物目录算 RMSD；pose_dir 可为 out_dir 或其下的复合物子目录。"""
    ref_mol, ref_pos = load_pos(ref_path)
    if ref_mol is None:
        return {"error": "cannot read ref %s" % ref_path}
    cands = [pose_dir] + [os.path.join(pose_dir, s)
                          for s in sorted(os.listdir(pose_dir))]
    poses = []
    for cd in cands:
        if not os.path.isdir(cd):
            continue
        for f in os.listdir(cd):
            m = NAME_RE.match(f)
            if m:
                poses.append((int(m.group(1)), float(m.group(2)), os.path.join(cd, f)))
    poses.sort(key=lambda x: x[0])
    rmsds, confs = [], []
    for rank, conf, fp in poses:
        mol, pos = load_pos(fp)
        if mol is None or pos is None:
            continue
        try:
            rmsd = float(get_symmetry_rmsd(ref_mol, ref_pos, pos))
        except Exception as e:  # noqa: BLE001
            rmsd = None
            print("  [warn] rmsd failed on %s: %r" % (fp, e))
        rmsds.append(None if rmsd is None else round(rmsd, 4))
        confs.append(conf)
    valid = [r for r in rmsds if r is not None]
    return {
        "ref": ref_path,
        "n_poses": len(valid),
        "rmsd_per_pose": rmsds,
        "confidence_per_pose": confs,
        "rmsd_top1": rmsds[0] if rmsds else None,
        "rmsd_best": min(valid) if valid else None,
        "top1_lt_cutoff": (rmsds[0] is not None and rmsds[0] <= cutoff) if rmsds else None,
        "n_lt_cutoff": sum(1 for r in valid if r <= cutoff),
    }


def multi(A):
    """多复合物：每个结果目录下的每个复合物子目录各自对模板参考算 RMSD。"""
    out = {"ref_tmpl": A.ref_tmpl, "cutoff": A.cutoff, "runs": {}}
    for d in A.results:
        tag = os.path.basename(d.rstrip("/"))
        complexes = sorted(s for s in os.listdir(d) if os.path.isdir(os.path.join(d, s)))
        rec = {}
        for name in complexes:
            ref = A.ref_tmpl.format(name=name)
            if not os.path.exists(ref):
                rec[name] = {"error": "no ref " + ref}
                continue
            rec[name] = eval_one(ref, os.path.join(d, name), A.cutoff)
        vals = [v for v in rec.values() if "rmsd_top1" in v]
        out["runs"][tag] = {
            "per_complex": rec,
            "n_complexes": len(vals),
            "top1_success": sum(1 for v in vals if v["top1_lt_cutoff"]),
            "top1_rmsd_list": [v["rmsd_top1"] for v in vals],
            "best_rmsd_list": [v["rmsd_best"] for v in vals],
        }
    print("ACC_JSON " + json.dumps(out, ensure_ascii=False))
    if A.json:
        with open(A.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None, help="晶体配体 sdf（单复合物）")
    ap.add_argument("--ref-tmpl", default=None,
                    help="多复合物：参考配体路径模板，如 examples/{name}_ligand.sdf")
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--cutoff", type=float, default=2.0)
    ap.add_argument("--json", default=None)
    A = ap.parse_args()

    if A.ref is None and A.ref_tmpl is None:
        raise SystemExit("need --ref or --ref-tmpl")

    if A.ref_tmpl:
        return multi(A)

    out = {"ref": A.ref, "cutoff": A.cutoff, "runs": {}}
    for d in A.results:
        tag = os.path.basename(d.rstrip("/"))
        out["runs"][tag] = eval_one(A.ref, d, A.cutoff)

    print("ACC_JSON " + json.dumps(out, ensure_ascii=False))
    if A.json:
        with open(A.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

