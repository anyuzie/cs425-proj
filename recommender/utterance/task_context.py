"""Independent task/context scores for clauses; no preference or ranking logic."""
import math
from pathlib import Path

LABELS=('task','context')

def route_scores(task_score,context_score,task_threshold=.5,context_threshold=.5):
    for value in (task_score,context_score):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Scores must be finite numbers between zero and one')
    for value in (task_threshold,context_threshold):
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<1:
            raise ValueError('Thresholds must be finite numbers strictly between zero and one')
    return {'task_score':float(task_score),'context_score':float(context_score),
        'route_task':task_score>=task_threshold,'route_context':context_score>=context_threshold}


class TaskContextClassifier:
    def __init__(self,checkpoint,*,task_threshold=None,context_threshold=None,batch_size=32):
        import torch
        from transformers import AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
        checkpoint=Path(checkpoint)
        config=AutoConfig.from_pretrained(checkpoint,local_files_only=True)
        settings=getattr(config,'task_context_classifier',{})
        if (config.model_type!='bert' or config.num_labels!=2 or config.problem_type!='multi_label_classification'
            or tuple(config.id2label.get(i) for i in range(2))!=LABELS or settings.get('trained') is not True):
            raise ValueError('Expected a trained BERT task/context checkpoint with two independent outputs')
        defaults=settings.get('thresholds',[.5,.5])
        self.task_threshold=defaults[0] if task_threshold is None else task_threshold
        self.context_threshold=defaults[1] if context_threshold is None else context_threshold
        route_scores(.5,.5,self.task_threshold,self.context_threshold)
        if type(batch_size)is not int or batch_size<1:raise ValueError('batch_size must be positive')
        self.batch_size=batch_size;self.max_length=settings['max_length']
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer=AutoTokenizer.from_pretrained(checkpoint,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(checkpoint,local_files_only=True).to(self.device).eval()

    def predict(self,texts):
        import torch
        if not isinstance(texts,(list,tuple)) or any(not isinstance(t,str) or not t.strip() for t in texts):
            raise ValueError('Provide a list of nonempty clause texts')
        results=[]
        for start in range(0,len(texts),self.batch_size):
            encoded=self.tokenizer(list(texts[start:start+self.batch_size]),padding=True,truncation=True,
                max_length=self.max_length,return_tensors='pt').to(self.device)
            with torch.inference_mode():scores=self.model(**encoded).logits.sigmoid().cpu().tolist()
            for task,context in scores:
                results.append(route_scores(task,context,self.task_threshold,self.context_threshold))
        return results
