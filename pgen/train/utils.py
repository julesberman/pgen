class Accumlator:

    def __init__(self) -> None:
        self.vals = {}
        self.counts = {}
        self.streams = {}
        self.count = 0

    def add(self, in_data: dict):
        for k, v in in_data.items():
            if k not in self.vals:
                self.vals[k] = 0.0
                self.counts[k] = 0
            self.vals[k] += v
            self.counts[k] += 1

    def save(self):
        for k, v in self.vals.items():
            if self.counts[k] > 0:
                if k not in self.streams:
                    self.streams[k] = []
                self.streams[k].append(v / self.counts[k])
        self.vals = {}
        self.counts = {}

    @property
    def means(self):
        # print(self.counts, self.vals)
        return {k: v / self.counts[k] for k, v in self.vals.items()}
