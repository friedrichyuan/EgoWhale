"""Transport for incremental Base_IK. No joint arrays or numerical logic in the scheduler."""

from egowhale.batch.dag import POOL
from egowhale.step import Result, short_error


def choose(active, capacity, refill=False):
    ready = [j for j in active if ("base_ik", None) in j.ready()]
    if ready and (refill or len(ready) >= capacity):
        return "base_ik", ready[:capacity]
    for job in active:
        for stage, _ in job.ready():
            if POOL[stage] == "action" and stage != "base_ik":
                return stage, [job]
    producers = any(not j.failed and ("retarget", None) not in j.done for j in active)
    if ready and not producers:
        return "base_ik", ready[:capacity]
    return None


class EpisodeActions:
    def __init__(self, scheduler, capacity, batch_size):
        self.s, self.capacity, self.batch_size = scheduler, capacity, batch_size
        self.slots = [dict(lane=lane, gpu=None, next=None, live={}) for lane in scheduler.action_lanes]

    def feed(self):
        for slot in self.slots:
            if slot["next"] is None:
                chosen = choose(self.s.active, self.capacity, refill=bool(slot["live"]))
                if chosen is not None:
                    stage, jobs = chosen
                    for job in jobs:
                        job.running.add((stage, None))
                    work = dict(stage=stage, jobs=jobs, data=None, offset=0)
                    slot["next"] = work
                    ref = slot["lane"].reader.load_many.remote(stage, [(str(j.src), str(j.dst)) for j in jobs])
                    self.s.pending[ref] = (self.on_read, (slot, work))
            self.launch(slot)

    def on_read(self, ref, slot, work):
        work["data"] = ref
        self.launch(slot)

    def launch(self, slot):
        if slot["gpu"] is not None:
            return
        work = slot["next"]
        if work is not None and work["data"] is not None and work["stage"] != "base_ik":
            slot["next"] = None
            ref = slot["lane"].actor.compute_many.remote(work["stage"], work["data"], self.batch_size)
            self.s.pending[ref] = (self.on_other, (slot, work))
        else:
            payload, start, count = None, 0, 0
            if work is not None and work["data"] is not None:
                count = min(self.capacity - len(slot["live"]), len(work["jobs"]) - work["offset"])
                if count:
                    start, payload = work["offset"], work["data"]
                    for job in work["jobs"][start : start + count]:
                        slot["live"][str(job.dst)] = job
                    work["offset"] += count
                    if work["offset"] == len(work["jobs"]):
                        slot["next"] = None
            if not slot["live"]:
                return
            ref = slot["lane"].actor.advance_ik.remote(payload, self.batch_size, start, count)
            self.s.pending[ref] = (self.on_ik, (slot,))
        slot["gpu"] = ref

    def deliver(self, slot, job, stage, row):
        if not row["ok"]:
            self.s.apply([(job, None)], [Result(False, 0.0, row["error"])], stage)
            return
        ref = slot["lane"].writer.save.remote(row["save_ref"])
        self.s.pending[ref] = (self.s.on_write, (stage, [(job, None)], [Result(True, row["seconds"])]))

    def on_ik(self, ref, slot):
        import ray

        slot["gpu"] = None
        try:
            payload = ray.get(ref)
            tokens = [row["token"] for row in payload["rows"]]
            if (
                len(set(tokens)) != len(tokens)
                or set(tokens) | set(payload["live"]) != set(slot["live"])
                or set(tokens) & set(payload["live"])
            ):
                raise ValueError("IK stream result identities mismatch")
            for row in payload["rows"]:
                job = slot["live"].pop(row["token"])
                self.deliver(slot, job, "base_ik", row)
        except Exception as exc:
            for job in slot["live"].values():
                self.s.apply([(job, None)], [Result(False, 0.0, short_error(exc))], "base_ik")
            slot["live"].clear()
        # pump/feed handles replacement admission before issuing the next quantum.

    def on_other(self, ref, slot, work):
        import ray

        slot["gpu"] = None
        try:
            rows = ray.get(ref)
            if len(rows) != len(work["jobs"]):
                raise ValueError("action result count mismatch")
        except Exception as exc:
            rows = [dict(ok=False, error=short_error(exc)) for _ in work["jobs"]]
        for job, row in zip(work["jobs"], rows, strict=True):
            self.deliver(slot, job, work["stage"], row)
