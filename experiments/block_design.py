"""Stable block identities: selecting fewer blocks never renumbers their orders."""
import itertools
from collections import Counter

POLICIES = ['separate', 'interleaved', 'mixed']
ORDERS = list(itertools.permutations(POLICIES))
DEFAULT_BLOCK_IDS = '0,1,2,3,4,5'
# Two-sided 95% Student t critical values, df=1..5 (at most six blocks).
T95 = {1: 12.706204736432095, 2: 4.302652729911275, 3: 3.182446305284263,
       4: 2.7764451051977987, 5: 2.570581835636305}


def parse_block_ids(value):
    try:
        ids = [int(part.strip()) for part in value.split(',')]
    except (ValueError, AttributeError):
        raise ValueError('Use comma-separated block IDs, e.g. 0,3,4') from None
    if not ids or len(set(ids)) != len(ids) or any(x not in range(6) for x in ids):
        raise ValueError('Block IDs must be distinct integers from 0 through 5')
    return ids


def position_balanced(ids):
    return all(len(set(Counter(ORDERS[b].index(policy) for b in ids).get(i, 0)
                       for i in range(3))) == 1 for policy in POLICIES)
