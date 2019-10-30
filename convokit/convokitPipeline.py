from sklearn.pipeline import Pipeline
from convokit.model import Utterance

class ConvokitPipeline(Pipeline):
    
    def __init__(self, steps):
        Pipeline.__init__(self, steps)
    def transform_utterance(self, utt):
        if isinstance(utt, str):
            utt = Utterance(text=utt)
        for name, transform in self.steps:
            utt = transform.transform_utterance(utt)
        return utt