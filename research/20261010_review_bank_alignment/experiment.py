"""One bounded CPU-only geometry transfer audit from frozen support caches."""
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MAX_SECONDS = 3600
N_USERS, N_ITEMS, N_EDGES, N_QUERIES = 19445, 7050, 105032, 13447
TARGET_SHA = "1d019d38c5274cb02b5dfbd434f6e6d2ce5e904fca379cc009ba857e35bdfb1a"
_DEADLINE = None
sys.dont_write_bytecode = True


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b""): h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    pending=path.with_name(path.name+".pending."+str(os.getpid()))
    with pending.open("x",encoding="utf-8") as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    pending.replace(path)


def checkpoint():
    if time.monotonic()>=_DEADLINE: raise TimeoutError("One-hour alignment diagnostic budget exceeded")


def memory_available():
    return next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))


def verify_inputs(manifest):
    actual={name:sha256(row["path"]) for name,row in manifest["files"].items()}
    if actual != {name:row["sha256"] for name,row in manifest["files"].items()}:
        raise RuntimeError("Protected cached input changed")
    return actual


def verify_bundle():
    path=ROOT/"deployment.json"; manifest=json.loads(path.read_text())
    actual={name:sha256(ROOT/name) for name in manifest["bundle_sha256"]}
    if actual!=manifest["bundle_sha256"]: raise RuntimeError("Immutable bundle changed")
    return {"manifest_sha256":sha256(path),"bundle_sha256":actual}


def load_module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    global _DEADLINE
    os.environ["CUDA_VISIBLE_DEVICES"]=""
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[name]="4"
    started=time.monotonic(); _DEADLINE=started+MAX_SECONDS
    signal.signal(signal.SIGALRM,lambda signum,frame:(_ for _ in ()).throw(
        TimeoutError("One-hour alignment diagnostic hard alarm exceeded")))
    signal.setitimer(signal.ITIMER_REAL,MAX_SECONDS)
    record={"status":"running","pid":os.getpid(),"max_seconds":MAX_SECONDS,"cpu_threads":4,
            "gpu_used":False,"new_encoding_performed":False,"raw_reviews_read":False,
            "interactions_read":False,"validation_evaluated":False,"test_evaluated":False,
            "training_started":False,"formal_training_authorized":False,"publication_evidence":False,
            "novelty_established":False,"preference_established":False}
    stage_seconds={}
    try:
        before_bundle=verify_bundle()
        manifest_path=ROOT/"input_audit.json"; manifest_sha=sha256(manifest_path)
        manifest=json.loads(manifest_path.read_text())
        if manifest["status"]!="completed": raise RuntimeError("Frozen completed input audit required")
        before_inputs=verify_inputs(manifest); paths=manifest["files"]
        f25=json.loads(Path(paths["predecessor_diagnosis"]["path"]).read_text())
        f26=json.loads(Path(paths["f26_diagnosis"]["path"]).read_text())
        if (f25["status"]!="completed" or f25["source_revision"]!=manifest["predecessor_commit"]
                or f25["heldout_target_hash"]!=TARGET_SHA
                or not f25["masking_settings"]["all_internal_targets_absent_from_support_and_records"]
                or f25["matching_audit"]["support_users"]!=N_USERS
                or f26["status"]!="completed" or not all(view["passed"] for view in f26["views"].values())):
            raise RuntimeError("Frozen support-only cache or expression provenance changed")
        while memory_available()<12*1024**3:
            checkpoint(); atomic_json(ROOT/"progress.json",dict(record,stage="waiting_for_RAM"))
            time.sleep(min(30,max(0,_DEADLINE-time.monotonic())))
        record["available_ram_bytes_at_start"]=memory_available()
        import numpy as np
        core=load_module("fixed_bank_alignment_core",ROOT/"core.py")
        residual=load_module("frozen_residual_core",paths["residual_core"]["path"])
        evidence=load_module("evidence",paths["dual_witness_evidence"]["path"])
        sys.modules["evidence"]=evidence
        diagnose=load_module("frozen_knn_diagnose",paths["dual_witness_diagnose"]["path"])
        cache=ROOT/"cache"; cache.mkdir(exist_ok=False)
        embeddings=np.load(paths["review_embeddings"]["path"],mmap_mode="r",allow_pickle=False)
        with np.load(paths["review_index"]["path"],allow_pickle=False) as index:
            users,items=index["user_ids"].copy(),index["item_ids"].copy()
            positions=index["record_indices"].copy()
            own_positions,donor_positions=index["own_record_indices"].copy(),index["donor_record_indices"].copy()
        if (embeddings.shape!=(N_EDGES,512) or users.shape!=(N_EDGES,) or items.shape!=(N_EDGES,)
                or positions.shape!=(N_EDGES,) or len(np.unique(positions))!=N_EDGES
                or not np.array_equal(positions,np.sort(positions))
                or own_positions.shape!=(N_EDGES,) or donor_positions.shape!=(N_EDGES,)
                or not np.array_equal(np.sort(own_positions),positions)
                or not np.isfinite(embeddings).all()):
            raise RuntimeError("Frozen review cache layout changed")
        own_rows=np.searchsorted(positions,own_positions)
        donor_rows=np.searchsorted(positions,donor_positions)
        if (np.any(donor_rows>=len(positions)) or not np.array_equal(positions[donor_rows],donor_positions)
                or np.any(items[own_rows]!=items[donor_rows]) or np.any(users[own_rows]==users[donor_rows])):
            raise RuntimeError("Frozen same-item different-owner donor map is invalid")
        owners,products=users[own_rows],items[own_rows]
        support=core.binary_support(owners,products,N_USERS,N_ITEMS)
        with np.load(paths["f25_per_user"]["path"],allow_pickle=False) as previous:
            query_users,targets,comparisons=(previous[name].copy() for name in ("user_ids","targets","comparisons"))
            old={name:previous[name].copy() for name in ("raw_own","same_item_donor","item_content","residual","residual_donor")}
        if query_users.shape!=(N_QUERIES,) or core.validate_pairs(support,query_users,targets,comparisons)!=TARGET_SHA:
            raise RuntimeError("Fixed global targets/comparison mask changed")
        with np.load(paths["f23_per_user"]["path"],allow_pickle=False) as prior_knn:
            if not all(np.array_equal(prior_knn[name],value) for name,value in
                       (("user_ids",query_users),("targets",targets),("comparisons",comparisons))):
                raise RuntimeError("Full-support KNN reference uses different pairs")
            full_support_reference={name:prior_knn[name].copy() for name in ("itemknn","userknn")}
        bank=np.load(paths["global_bank"]["path"],mmap_mode="r",allow_pickle=False)
        if bank.shape!=(N_ITEMS,2,512) or not np.isfinite(bank).all(): raise RuntimeError("Global bank changed")
        target_vectors=residual.normalize_rows(bank.sum(axis=1)); item_text=bank[:,1]
        started_stage=time.monotonic()
        own_raw=embeddings[own_rows]; donor_raw=embeddings[donor_rows]
        own_residual=residual.residualize(own_raw,item_text[products])
        donor_residual=residual.residualize(donor_raw,item_text[products])
        profiles={"candidate":residual.profile(own_residual,owners,N_USERS),
                  "raw_review_bank":residual.profile(own_raw,owners,N_USERS),
                  "raw_donor_bank":residual.profile(donor_raw,owners,N_USERS),
                  "residual_donor_bank":residual.profile(donor_residual,owners,N_USERS),
                  "item_content":residual.profile(item_text[products],owners,N_USERS)}
        stage_seconds["frozen_profiles"]=time.monotonic()-started_stage
        scores={"old_target_geometry":residual.score_rows(profiles["candidate"],target_vectors,query_users,comparisons),
                "item_content":residual.score_rows(profiles["item_content"],target_vectors,query_users,comparisons)}
        per_user={name:core.auc(value) for name,value in scores.items()}
        if (not np.array_equal(per_user["old_target_geometry"],old["residual"])
                or not np.array_equal(per_user["item_content"],old["item_content"])):
            raise RuntimeError("Old geometry per-user AUCs failed exact F25 reproduction")
        for previous_name,profile_name in (("raw_own","raw_review_bank"),
                  ("same_item_donor","raw_donor_bank"),("residual_donor","residual_donor_bank")):
            previous_scores=residual.score_rows(profiles[profile_name],target_vectors,query_users,comparisons)
            if not np.array_equal(core.auc(previous_scores),old[previous_name]):
                raise RuntimeError("Frozen own/donor mapping failed exact F25 profile reproduction")
            del previous_scores
        atomic_json(ROOT/"progress.json",dict(record,stage="old_scores_exactly_reproduced"))
        started_stage=time.monotonic()
        banks={"raw":core.item_bank(own_raw,owners,products,N_USERS,N_ITEMS),
               "residual":core.item_bank(own_residual,owners,products,N_USERS,N_ITEMS)}
        availability={}; removal_counts={}
        for name,view in (("candidate","residual"),("raw_review_bank","raw"),
                          ("raw_donor_bank","raw"),("residual_donor_bank","residual")):
            scores[name],availability[name],removal_counts[name]=core.bank_scores(
                profiles[name],banks[view],query_users,comparisons,block_size=128,checkpoint=checkpoint)
            if removal_counts[name]!=0: raise RuntimeError("Unexpected query owner review in fixed target/negative item banks")
            per_user[name]=core.auc(scores[name])
        profile_nonzero=np.linalg.norm(profiles["candidate"][query_users],axis=1)>core.EPS
        covered=profile_nonzero & availability["candidate"].all(axis=1)
        coverage=float(covered.mean())
        stage_seconds["new_bank_scores"]=time.monotonic()-started_stage
        atomic_json(ROOT/"progress.json",dict(record,stage="computing_fixed_support_controls",coverage=coverage))
        started_stage=time.monotonic()
        popularity=np.asarray(support.sum(axis=0)).reshape(-1)
        scores["popularity"]=popularity[comparisons]
        cooc=(support.T@support).tocsr(); cooc.setdiag(0); cooc.eliminate_zeros()
        inverse=1/np.sqrt(np.maximum(popularity,1e-6))
        similarity=cooc.multiply(inverse[:,None]).multiply(inverse[None,:]).tocsr()
        scores["itemknn"]=diagnose.item_scores(evidence.prune_topk(similarity,1000),support,query_users,comparisons)
        del cooc,similarity
        checkpoint()
        weights=evidence.build_user_evidence(support,mode="userknn",topk=100,block_size=256)
        scores["userknn"]=evidence.gather_scores(weights,support,query_users,comparisons)
        del weights
        for name in ("popularity","itemknn","userknn"): per_user[name]=core.auc(scores[name])
        stage_seconds["fixed_support_knn"]=time.monotonic()-started_stage
        started_stage=time.monotonic()
        decision,halves=core.statistics(per_user,coverage,query_users,bootstrap=2000,checkpoint=checkpoint)
        stage_seconds["paired_statistics"]=time.monotonic()-started_stage
        np.savez_compressed(cache/"per_user.npz",user_ids=query_users,targets=targets,comparisons=comparisons,
                            user_hash_half=halves,candidate_covered=covered,**per_user)
        np.savez_compressed(cache/"scores.npz",user_ids=query_users,comparisons=comparisons,**scores)
        np.savez_compressed(cache/"full_support_reference.npz",user_ids=query_users,targets=targets,
                            comparisons=comparisons,**full_support_reference)
        checkpoint(); after_inputs=verify_inputs(manifest); after_bundle=verify_bundle()
        if before_inputs!=after_inputs or before_bundle!=after_bundle or sha256(manifest_path)!=manifest_sha:
            raise RuntimeError("Protected inputs or deployment changed during diagnostic")
        if "torch" in sys.modules or "clip" in sys.modules: raise RuntimeError("CPU cache audit imported forbidden encoder")
        record.update(status="completed",stage="done",decision=decision,
            source_revision=json.loads((ROOT/"source_revision.json").read_text())["commit"],
            predecessor_commit=manifest["predecessor_commit"],heldout_target_hash=TARGET_SHA,
            diagnostic_users=N_QUERIES,support_edges=support.nnz,all_targets_absent=True,
            all_comparisons_absent_from_query_support=True,old_per_user_aucs_exactly_reproduced=True,
            query_owner_contributions_removed=removal_counts,
            coverage={"candidate_all17":coverage,"candidate_profile":float(profile_nonzero.mean()),
                "candidate_target_bank":float(availability["candidate"][:,0].mean()),
                "candidate_all_comparison_banks":float(availability["candidate"].all(axis=1).mean()),
                "nonzero_residual_item_banks":int((np.linalg.norm(banks["residual"]["sums"],axis=1)>core.EPS).sum()),
                "catalog_items":N_ITEMS},
            metrics={name:{"mean_auc":float(per_user[name].mean())} for name in core.ARMS},
            full_support_knn_references={name:{"mean_auc":float(value.mean()),"support_edges":105104,
                "extra_edges_vs_gate_controls":72,"enters_gate":False} for name,value in full_support_reference.items()},
            inputs_sha256_before=before_inputs,inputs_sha256_after=after_inputs,
            deployment_before=before_bundle,deployment_after=after_bundle,input_audit_sha256=manifest_sha,
            elapsed_seconds=time.monotonic()-started,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            stage_seconds=stage_seconds,interpretation="observational TRAIN internal target geometry diagnostic only")
        lines=["arm\tmean_user_auc\tcandidate_minus_control\tci_lower\tci_upper"]
        for name in core.ARMS:
            pair=decision["paired_vs_controls"].get(name)
            lines.append("{}\t{:.10f}\t{}\t{}\t{}".format(name,per_user[name].mean(),
                "" if pair is None else pair["mean_delta"],"" if pair is None else pair["user_bootstrap_95ci"][0],
                "" if pair is None else pair["user_bootstrap_95ci"][1]))
        with (ROOT/"results.tsv").open("x",encoding="utf-8") as stream: stream.write("\n".join(lines)+"\n")
        atomic_json(ROOT/"diagnosis.json",record); atomic_json(ROOT/"progress.json",record)
        print(json.dumps(record,allow_nan=False),flush=True)
    except BaseException as error:
        record.update(status="technical_failure",error=repr(error),traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic()-started,stage_seconds=stage_seconds)
        atomic_json(ROOT/"diagnosis.json",record); atomic_json(ROOT/"progress.json",record)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__ == "__main__": main()
