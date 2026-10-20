"""Message structures, cutoff-safe loading, conversation context and review sampling."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class Message:
    id: str
    parent_id: str | None
    platform: str
    channel: str
    user_name: str
    title: str | None
    content: str | None
    created_at: datetime

    @property
    def text(self):
        return '\n'.join((v for v in (self.title, self.content) if v))

    @property
    def actor(self):
        return (self.platform, self.user_name) if self.user_name.strip() else (self.platform, 'unknown:' + self.id)

@dataclass(frozen=True)
class Concern:
    id: str
    voice_id: str
    target: str
    issue_zh: str
    issue_en: str
    category: str
    position: str
    stance: str
    style: str
    evidence_type: str
    quote: str
    context_ids: tuple[str, ...] = ()

@dataclass
class Interpretation:
    voice_id: str
    literal_en: str
    meaning_en: str
    feeling: str
    concerns: list[Concern] = field(default_factory=list)

@dataclass
class Topic:
    id: str
    title: str
    title_en: str
    category: str
    members: list[Concern] = field(default_factory=list)
STANCES = {'critical', 'positive', 'neutral', 'mixed', 'unclear'}
STYLES = {'literal', 'humorous', 'sarcastic', 'unclear'}
EVIDENCE = {'firsthand', 'agreement', 'disagreement', 'allegation', 'quotation', 'other'}
STATUSES = {'new', 'growing', 'one-off', 'fading', 'not real'}
import csv, json
from datetime import datetime, timedelta, timezone
from pathlib import Path
HK = timezone(timedelta(hours=8))

def parse_timestamp(value):
    d = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if d.tzinfo is None:
        raise ValueError('Timestamp requires a timezone')
    return d.astimezone(HK)

def cutoff_exclusive(day):
    return datetime.strptime(day, '%Y-%m-%d').replace(tzinfo=HK) + timedelta(days=1)

def load_messages(path, as_of):
    cutoff = cutoff_exclusive(as_of)
    out = []
    seen = set()
    with Path(path).open(encoding='utf-8-sig') as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                stamp = parse_timestamp(row['created_at'])
                if stamp >= cutoff:
                    continue
                for k in ['id', 'platform', 'channel', 'user_name']:
                    if not isinstance(row.get(k), str):
                        raise ValueError('Missing/non-string ' + k)
                if not row['id'] or not row['platform']:
                    raise ValueError('Empty ID/platform')
                if row['id'] in seen:
                    raise ValueError('Duplicate ID')
                for k in ['parent_id', 'title', 'content']:
                    if row.get(k) is not None and (not isinstance(row[k], str)):
                        raise ValueError('Invalid ' + k)
                out.append(Message(**{k: row.get(k) for k in Message.__dataclass_fields__ if k != 'created_at'}, created_at=stamp))
                seen.add(row['id'])
            except (ValueError, TypeError, KeyError, AttributeError) as e:
                raise ValueError(f'{path}:{line_no}: {e}') from e
    return sorted(out, key=lambda m: (m.created_at, m.id))

def load_events(path, as_of):
    if not path:
        return []
    cutoff = cutoff_exclusive(as_of).date()
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return [dict(r) for r in csv.DictReader(f) if datetime.strptime(r['date'], '%Y-%m-%d').date() < cutoff]
import re, unicodedata
TRAD = str.maketrans({'確': '确', '實': '实', '釣': '钓', '魚': '鱼', '體': '体', '壓': '压', '鐵': '铁', '鍋': '锅', '賽': '赛', '來': '来', '週': '周', '務': '务', '個': '个', '說': '说'})

def normalize(text):
    text = unicodedata.normalize('NFKC', text or '').translate(TRAD).lower()
    text = re.sub('<[^>]+>|https?://\\S+', ' ', text)
    return ''.join(re.findall('[\\u4e00-\\u9fff]+|[a-z0-9]+|\\+1', text))

def compact(text, limit=200):
    return re.sub('<image:[^>]+>', '<image unavailable>', text or '')[:limit]

class ContextIndex:

    def __init__(self, messages):
        self.by_id = {m.id: m for m in messages}
        self.aliases = []
        for m in messages:
            for name, alias in re.findall('([\\u4e00-\\u9fff]{2,6})\\s*[=＝]\\s*([\\u4e00-\\u9fff]{2,6})', m.text):
                if not any((a['name'] == name and a['alias'] == alias for a in self.aliases)):
                    self.aliases.append({'name': name, 'alias': alias, 'voice_id': m.id, 'created_at': m.created_at})

    def ancestors(self, m, depth=2):
        seen = {m.id}
        out = []
        pid = m.parent_id
        while pid and len(out) < depth:
            p = self.by_id.get(pid)
            if not p or p.id in seen or p.created_at > m.created_at or (p.platform != m.platform):
                break
            out.append(p)
            seen.add(p.id)
            pid = p.parent_id
        return out

    def known_aliases(self, m):
        return [a for a in self.aliases if a['created_at'] <= m.created_at and (a['name'] in m.text or a['alias'] in m.text)]

    def matching_text(self, m):
        text = m.text
        for a in self.known_aliases(m):
            text = text.replace(a['alias'], a['name'])
        return normalize(text)

    def signature(self, m):
        return (self.matching_text(m), tuple((normalize(p.text) for p in self.ancestors(m))))

    def bundle(self, m):
        return {'id': m.id, 'title': compact(m.title, 60), 'text': compact(m.content, 220), 'parents': [{'id': p.id, 'title': compact(p.title, 60), 'text': compact(p.content, 100)} for p in self.ancestors(m)], 'aliases': [{'name': a['name'], 'alias': a['alias'], 'evidence_id': a['voice_id']} for a in self.known_aliases(m)[:3]]}
from collections import defaultdict
import hashlib
FILLER = {'前排', '沙发', '晚安', '早安', '666', '顶', '已三连'}

def eligible(m):
    text = normalize(m.text)
    return bool(text) and text not in FILLER and (not ('欢迎来到' in text and '群公告' in text))

def select_messages(messages, context, budget=96):
    pool = [m for m in messages if eligible(m)]
    if budget == 0 or len(pool) <= budget:
        return pool
    from datetime import timedelta
    recent_start = messages[-1].created_at.date() - timedelta(days=2)
    baseline_start = recent_start - timedelta(days=14)
    buckets = defaultdict(list)
    for m in pool:
        buckets[m.created_at.date(), m.platform].append(m)
    for key in buckets:
        buckets[key].sort(key=lambda m: hashlib.sha256((m.id + 'review-v2').encode()).hexdigest())
    recent = sorted((k for k in buckets if k[0] >= recent_start), reverse=True)
    baseline = sorted((k for k in buckets if baseline_start <= k[0] < recent_start), reverse=True)
    older = sorted((k for k in buckets if k[0] < baseline_start), reverse=True)
    keys = []
    for i in range(max(len(recent), len(baseline), len(older))):
        for group in (recent, baseline, older):
            if i < len(group):
                keys.append(group[i])
    chosen = []
    used = set()
    positions = defaultdict(int)
    while len(chosen) < budget:
        changed = False
        for key in keys:
            b = buckets[key]
            while positions[key] < len(b):
                m = b[positions[key]]
                positions[key] += 1
                signature = (key, context.signature(m))
                if signature in used:
                    continue
                used.add(signature)
                chosen.append(m)
                changed = True
                break
            if len(chosen) >= budget:
                break
        if not changed:
            break
    return sorted(chosen, key=lambda m: (m.created_at, m.id))
