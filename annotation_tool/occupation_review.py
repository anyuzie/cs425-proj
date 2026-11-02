"""Isolated occupation CSV review service; no utterance or ranking imports."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
from threading import RLock
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

EXTRA = ['reviewer_start', 'reviewer_end', 'reviewer_identity', 'reviewed_at',
         'active', 'replaces_ids', 'replacement_ids', 'training_eligible']
CATEGORIES = {'activity': ('1','0'), 'job_nature': ('0','1'), 'both': ('1','1'),
              'neither': ('0','0'), 'uncertain': ('','')}


class ReviewClause(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    category: str
    text: str
    notes: str = ''


class ReviewSave(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    revision: str
    ids: list[str]
    clauses: list[ReviewClause]
    reviewer: str
    confirmed: bool


def is_active(row):
    return row.get('active', '') != 'false'


def eligible(row):
    return (is_active(row) and row.get('review_status') == 'reviewed'
            and row.get('reviewer_activity') in ('0','1')
            and row.get('reviewer_job_nature') in ('0','1'))


class OccupationReviewStore:
    def __init__(self, path, batch_path):
        self.path = Path(path).resolve()
        self.batch_path = Path(batch_path).resolve()
        if self.path == self.batch_path:
            raise ValueError('Batch filter must be separate from the authoritative CSV')
        self.lock = RLock()
        with self.batch_path.open(encoding='utf-8-sig', newline='') as handle:
            self.batch_ids = {row['clause_id'] for row in csv.DictReader(handle)}
        self.read()

    def read(self):
        raw = self.path.read_bytes()
        reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
        fields = reader.fieldnames
        required = {'clause_id','source_text','occupation_code','source_field','start','end'}
        if not fields or not required <= set(fields):
            raise ValueError('Occupation CSV is missing required columns')
        rows = list(reader)
        if len({r['clause_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate clause IDs in occupation CSV')
        return rows, fields, hashlib.sha256(raw).hexdigest()

    def snapshot(self):
        with self.lock:
            rows, _, revision = self.read()
            for row in rows:
                ancestors = set(json.loads(row.get('replaces_ids') or '[]'))
                row['in_first_batch'] = row['clause_id'] in self.batch_ids or bool(ancestors & self.batch_ids)
                row['training_eligible'] = eligible(row)
            active = [r for r in rows if is_active(r)]
            return {'rows': rows, 'revision': revision, 'csv_path': str(self.path),
                    'progress': {'total': len(active), 'reviewed': sum(eligible(r) for r in active),
                    'uncertain': sum(r.get('review_status') == 'uncertain' for r in active),
                    'pending': sum(r.get('review_status') not in ('reviewed','uncertain') for r in active)}}

    def save(self, request):
        with self.lock:
            lockpath = self.path.with_suffix(self.path.suffix + '.lock')
            try:
                descriptor = os.open(lockpath, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                raise FileExistsError('Another save is in progress; reload and retry')
            try:
                os.close(descriptor)
                return self._save(request)
            finally:
                lockpath.unlink()

    def _save(self, request):
        rows, fields, revision = self.read()
        if request.revision != revision:
            raise FileExistsError('CSV changed since loading. Reload before saving; your edits were not written.')
        if not request.confirmed or not request.reviewer.strip():
            raise ValueError('Explicit confirmation and reviewer name are required')
        if not request.ids or len(set(request.ids)) != len(request.ids) or not request.clauses:
            raise ValueError('Select source clauses and provide at least one reviewed clause')
        by_id = {r['clause_id']:r for r in rows}
        if any(i not in by_id for i in request.ids):
            raise ValueError('Unknown clause ID')
        parents = [by_id[i] for i in request.ids]
        if any(not is_active(r) for r in parents):
            raise ValueError('Replaced clauses cannot be edited')
        source_key = lambda r: (r['occupation_code'],r['source_field'],r.get('task_id',''),r['source_text'])
        if len({source_key(r) for r in parents}) != 1:
            raise ValueError('Merges must use clauses from the same occupation and source')
        source = parents[0]['source_text']
        clauses = sorted(request.clauses, key=lambda c:c.start)
        previous = -1
        for clause in clauses:
            if not 0 <= clause.start < clause.end <= len(source) or clause.start < previous:
                raise ValueError('Corrected spans must be valid and non-overlapping')
            if not source[clause.start:clause.end].strip() or not clause.text.strip():
                raise ValueError('Clause text cannot be empty')
            if clause.category not in CATEGORIES:
                raise ValueError('Choose Activity, Job nature, Both, Neither or Uncertain')
            previous = clause.end
            prepared_label_only = (len(parents) == len(clauses) == 1
                and parents[0].get('preparation_status') == 'assistant_checked'
                and clause.start == int(parents[0].get('reviewer_start') or parents[0]['start'])
                and clause.end == int(parents[0].get('reviewer_end') or parents[0]['end']))
            # A corrected span must not silently swallow another active annotation.
            for other in rows:
                if prepared_label_only:
                    continue
                if other['clause_id'] in request.ids or not is_active(other) or source_key(other) != source_key(parents[0]):
                    continue
                lo, hi = int(other.get('reviewer_start') or other['start']), int(other.get('reviewer_end') or other['end'])
                if max(lo,clause.start) < min(hi,clause.end):
                    raise ValueError('Corrected span overlaps another active clause; select it for merging too')
        same_span = (len(parents)==len(clauses)==1
            and clauses[0].start == int(parents[0].get('reviewer_start') or parents[0]['start'])
            and clauses[0].end == int(parents[0].get('reviewer_end') or parents[0]['end']))
        stamp = datetime.now(timezone.utc).isoformat()
        saved = []
        ancestors = set(request.ids)
        for parent in parents:
            ancestors.update(json.loads(parent.get('replaces_ids') or '[]'))
        for clause in clauses:
            row = parents[0] if same_span else dict(parents[0])
            if not same_span:
                row['clause_id'] = 'OPR-' + uuid.uuid4().hex
                row['replaces_ids'] = json.dumps(sorted(ancestors))
                row['replacement_ids'] = ''
            a,n = CATEGORIES[clause.category]
            row.update(reviewer_start=str(clause.start), reviewer_end=str(clause.end),
                reviewer_text=clause.text.strip(), reviewer_activity=a, reviewer_job_nature=n,
                reviewer_notes=clause.notes, reviewer_identity=request.reviewer.strip(), reviewed_at=stamp,
                active='true', review_status='uncertain' if clause.category=='uncertain' else 'reviewed')
            row['training_eligible'] = str(eligible(row)).lower()
            saved.append(row)
        if not same_span:
            for parent in parents:
                parent.update(active='false', training_eligible='false',
                    replacement_ids=json.dumps([r['clause_id'] for r in saved]))
            rows.extend(saved)
        for name in EXTRA:
            if name not in fields: fields.append(name)
        output = io.StringIO(newline='')
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
        raw = output.getvalue().encode('utf-8-sig')
        backup = self.path.with_name(self.path.name + '.' + uuid.uuid4().hex + '.bak')
        backup.write_bytes(self.path.read_bytes())
        temporary = self.path.with_name('.' + self.path.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with temporary.open('xb') as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            # Catch external edits during validation/serialization as well.
            if hashlib.sha256(self.path.read_bytes()).hexdigest() != revision:
                raise FileExistsError('CSV changed during save; reload before retrying')
            os.replace(temporary,self.path)
        finally:
            if temporary.exists(): temporary.unlink()
        return {'saved_ids':[r['clause_id'] for r in saved], 'backup_path':str(backup), **self.snapshot()}


def create_occupation_app(csv_path, batch_path):
    store = OccupationReviewStore(csv_path,batch_path)
    app = FastAPI(title='Occupation clause review', docs_url=None, redoc_url=None)
    @app.get('/')
    def home():
        return FileResponse(Path(__file__).with_name('occupation_review.html'))
    @app.get('/api/occupation/rows')
    def rows():
        return store.snapshot()
    @app.post('/api/occupation/save')
    def save(request: ReviewSave):
        try: return store.save(request)
        except FileExistsError as error: raise HTTPException(409,str(error))
        except ValueError as error: raise HTTPException(422,str(error))
        except OSError as error: raise HTTPException(503,'CSV could not be saved; reload to verify. '+str(error))
    return app


def main():
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=Path(__file__).with_name('review_all_213.csv'))
    parser.add_argument('--batch',type=Path,default=Path(__file__).with_name('review_first_100.csv'))
    parser.add_argument('--port',type=int,default=8011)
    args=parser.parse_args()
    uvicorn.run(create_occupation_app(args.csv,args.batch),host='127.0.0.1',port=args.port)

if __name__=='__main__': main()
