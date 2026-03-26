# engine/cache_manager.py
import bisect


class SlotCacheManager:
    """
    最简单的 slot manager。
    只负责本地 slot 生命周期，不关心 request 调度。
    """

    def __init__(self, max_num_seqs: int):
        self.max_num_seqs = max_num_seqs
        self.free_slots = list(range(max_num_seqs))
        self.used_slots = set()

    def allocate_slot(self) -> int:
        if not self.free_slots:
            raise RuntimeError("No free slots available")
        slot = self.free_slots.pop(0)
        self.used_slots.add(slot)
        return slot

    def free_slot(self, slot: int) -> None:
        if slot in self.used_slots:
            self.used_slots.remove(slot)
            bisect.insort(self.free_slots, slot)

    def reset(self):
        self.free_slots = list(range(self.max_num_seqs))
        self.used_slots.clear()