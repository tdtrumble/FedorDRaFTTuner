from collections import OrderedDict
from jobs import BaseJob

class ExtensionJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        from extensions_built_in.krea2_draft_trainer.Krea2DraftTrainer import Krea2DraftTrainer

        self.device = self.get_conf('device', 'cuda')
        self.load_processes({'krea2_draft': Krea2DraftTrainer})

    def run(self):
        super().run()

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()
