"""Print independent task/context scores; each --text argument is one clause."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
try:from recommender.utterance.task_context import TaskContextClassifier
except ImportError:from task_context import TaskContextClassifier

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/task_context_bert')
    parser.add_argument('--text',action='append',required=True)
    parser.add_argument('--task-threshold',type=float,default=None)
    parser.add_argument('--context-threshold',type=float,default=None)
    args=parser.parse_args()
    classifier=TaskContextClassifier(args.checkpoint,task_threshold=args.task_threshold,context_threshold=args.context_threshold)
    for text,result in zip(args.text,classifier.predict(args.text)):
        print(json.dumps(dict(text=text,**result),ensure_ascii=False))

if __name__=='__main__':main()
