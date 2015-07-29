"""exp.py"""

from __future__ import division
import math
import random
import numpy as np
import collections
import operator
import itertools
import copy
import csv
import functools as ft
import multiprocessing as mp
import logging
import argparse
import json
from build_citation_graph import generate_data

ALPHA = 1
BETA = 100

logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)

#-------- helpers ---------
def truncated_normal_sample(u, std, n):
    """Sample from a N(u, std) but round to nearest of n evenly-spaced bins.

    Bins are evenly spaced across the 95% confidence interval.
    Ties broken randomly.

    """
    if std == 0:
        return u
    left = max(0, u - std * 1.96)
    right = min(1, u + std * 1.96)
    
    v = None
    while not (v >= left and v <= right):
        v = np.random.normal(loc=u, scale=std, size=None)

    endpoints = np.linspace(left, right, n + 1)
    midpoints = []
    for p1, p2 in zip(endpoints[:-1], endpoints[1:]):
        midpoints.append((p2 + p1) / 2)

    distances = [abs(p - v) for p in midpoints]
    min_distance = min(distances)
    closest_midpoints = [p for i, p in enumerate(midpoints) if
                         distances[i] == min_distance]
    return random.choice(closest_midpoints)

def f_arg(f, d):
    """Return pair with arg and function value"""
    return (d, f(d))

def f_expand(f, d):
    """Call f with d expanded"""
    return f(**d)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for
                                         r in range(len(s)+1))

def expectation_over_powerset(d, f):
    """Return expectation of d over all subsets.
        
    Applies function f to subsets of the keys of f, weighting by the product
    of probabilities from the keys of d. Uses probability if key in subset,
    else uses (1 - probability).

    Args:
        d:      Dictionary from keys to probabilities.
        f:      Function whose domain includes subsets of d.

    """
    v = 0
    for s in powerset(d): 
        p = reduce(operator.mul,
                   (d[k] if k in s else (1 - d[k]) for k in d),
                   1)
        v += p * f(s)
    return v

def expectation_bounded(d, f):
    """Return expectation that exploits bounded number of probability classes.

    Assumes items are identical from the point of view of function f.
    Could be modified to allow for classes of items.

    """
    probs_items = collections.defaultdict(list)
    for k in d:
        if d[k] > 0:
            probs_items[d[k]].append(k)

    probs = sorted(probs_items)
    v = 0
    for tup in itertools.product(*(xrange(len(probs_items[k]) + 1) for
                                   k in probs)):
        pr = 1
        items = []
        combinations = 1
        for p, n in zip(probs, tup):
            total_len = len(probs_items[p])
            pr *= p ** n * (1 - p) ** (total_len - n)
            items += probs_items[p][:n]
            combinations *= math.factorial(total_len) / \
                            math.factorial(total_len - n) /\
                            math.factorial(n)
        v += pr * f(items) * combinations
    return v

#---------- main functions ---------
def calculate_probabilities(n_authors=1000, n_resources=1000,
                            n_links=10, p_cite=0.5, p_use=0.45,
                            p_response_used=0.059,
                            p_response_used_std=0.0,
                            p_response_used_bins=1,
                            p_response_unused=0.016,
                            p_response_unused_std=0.0,
                            p_response_unused_bins=1,
                            p_response_uncited=0.0,
                            p_response_uncited_std=0.0,
                            p_response_uncited_bins=1):
    """Return mapping from author-resource pairs to probability of response
    
    Returns:
        Tuple of (true_probabilities), (estimated_probabilities)
    """
    data = generate_data(n_authors, n_resources, n_links, p_cite, p_use)
    authors = data['author_papers'].keys()
    resources = data['resources_id']

    probs_true = dict()
    probs_estimated = dict()
    for a in authors:
        used, unused, uncited = data['author_resources'][a] 
        # TODO: Sample here.
        for r in used:
            probs_estimated[a, r] = p_response_used
            probs_true[a, r] = truncated_normal_sample(p_response_used,
                                                       p_response_used_std,
                                                       p_response_used_bins)
        for r in unused:
            probs_estimated[a, r] = p_response_unused
            probs_true[a, r] = truncated_normal_sample(p_response_unused,
                                                       p_response_unused_std,
                                                       p_response_unused_bins)
        for r in uncited:
            probs_estimated[a, r] = p_response_uncited
            probs_true[a, r] = truncated_normal_sample(p_response_uncited,
                                                       p_response_uncited_std,
                                                       p_response_uncited_bins)
    return probs_true, probs_estimated

def f_utility_h(lst):
    return ALPHA * math.log1p(BETA * len(lst))

def f_utility(requests):
    """Utility of contributions resulting from a set of requests.

    Args:
        requests:   Iterable of (author, resource) pairs.

    Returns:
        Utility of contributions.

    """
    d = collections.defaultdict(list)
    for a, r in requests:
        d[r].append(a)

    f = f_utility_h
    return sum(f(d[r]) for r in d)

class Evaluator():
    def __init__(self, probs, f, exploit_bounded=True):
        """

        Args:
            probs:                  Dictionary from (author, resource) pairs to
                                    probability of contribution.
            exploit_bounded:        Take advantage of bounded number of
                                    probability classes.

        """
        self.probs = probs
        self.f = f
        self.exploit_bounded = exploit_bounded
        self.utilities_by_resource = dict()
        self.requests_by_resource = collections.defaultdict(set)
        self.utilities = [f([])]

    def deepcopy_ignore_probs(self):
        r = copy.copy(self)
        r.utilities_by_resource = copy.deepcopy(self.utilities_by_resource)
        r.requests_by_resource = copy.deepcopy(self.requests_by_resource)
        r.utilities = copy.deepcopy(self.utilities)
        return r

    def add(self, x):
        """Add x to set of requests.
        
        Recomputes utility for that resource and adds total expected utility
        to list of utilities.

        """
        a, r = x
        if x in self.requests_by_resource[r]:
            raise Exception('Already issued request')
        self.requests_by_resource[r].add(x)
        if self.probs[a, r] > 0:
            probs_r = dict((x_, self.probs[x_]) for
                           x_ in self.requests_by_resource[r])
            if self.exploit_bounded:
                self.utilities_by_resource[r] = expectation_bounded(
                    probs_r, self.f)
            else:
                self.utilities_by_resource[r] = expectation_over_powerset(
                    probs_r, self.f)
        self.utilities.append(sum(self.utilities_by_resource.itervalues()))

    def max_resource_requests(self):
        """Return maximum requests issued for any resource."""
        return max(len(x) for x in requests_by_resource.itervalues())

def single_run(probs_true, probs_estimated, policy, **args):
    """Execute a single policy run

    Policies are as follows:
        'random': Select a random author, assign that author a random resource.

    Args:
        probs_true:         Dictionary from (author, resource) to true
                            probability of contribution.
        probs_estimated:    Dictionary from (author, resource) to estimated
                            probability of contribution.
        policy:             Dictionary of policy settings.

    """
    authors = set(a for a, r in probs_estimated)
    resources = list(set(r for a, r in probs_estimated))
    evaluator_true = Evaluator(probs_true, f_utility,
                               exploit_bounded=True)
    evaluator_estimated  = Evaluator(probs_estimated, f_utility,
                                     exploit_bounded=True)

    def update(request):
        """Execute code that should be run after each decision."""
        evaluator_true.add(next_request)
        evaluator_estimated.add(next_request)
        logger.info('{}: {}'.format(policy['type'],
                                    len(evaluator_true.utilities),
                                    evaluator_true.utilities[-1],
                                    evaluator_estimated.utilities[-1]))

    if 'author_order' in policy:
        if policy['author_order'] == 'random':
            authors_sorted = list(authors)
            random.shuffle(authors_sorted)
        elif policy['author_order'] == 'highest_prob':
            # Sort in order of increasing highest probability, since
            # later we use .pop() and traverse list in reverse.
            authors_sorted = sorted(authors, key=lambda a: max(
                probs_estimated[a, r] for r in resources))
        else:
            raise NotImplementedError

    if policy['type'] == 'random':
        while len(authors_sorted) > 0:
            a = authors_sorted.pop()
            next_request = a, random.choice(resources)
            update(next_request)
    elif policy['type'] == 'greedy':
        while len(authors_sorted) > 0:
            a = authors_sorted.pop()
            possible_requests = [(a, r) for r in resources]
            max_prob = max(probs_estimated[x] for x in possible_requests)
            requests_max_prob = [x for x in possible_requests if
                                 probs_estimated[x] == max_prob]
            next_request = random.choice(requests_max_prob)
            update(next_request)
    elif policy['type'] == 'dt':
        marginal_utilities = dict((x, None) for x in probs_estimated)
        next_request = None
        while len(marginal_utilities) > 0:
            last_utility = evaluator_estimated.utilities[-1]
            if next_request is None:
                needs_update = marginal_utilities
            else:
                needs_update = [x for x in marginal_utilities if
                                x[1] == next_request[1]]
            for x in needs_update:
                evaluator_prime = evaluator_estimated.deepcopy_ignore_probs()
                evaluator_prime.add(x)
                marginal_utilities[x] = evaluator_prime.utilities[-1] - \
                                        evaluator_prime.utilities[-2]
            max_marginal_utility = marginal_utilities[
                max(marginal_utilities, key=marginal_utilities.get)]
            requests_max_marginal_utility = [
                x for x in marginal_utilities if
                marginal_utilities[x] == max_marginal_utility]
            next_request = random.choice(requests_max_marginal_utility)
            update(next_request)

            # Remove other candidate requests for selected author.
            for k in [k for k in marginal_utilities if
                      k[0] == next_request[0]]:
                del marginal_utilities[k]
    else: 
        raise NotImplementedError

    return evaluator_true.utilities, evaluator_estimated.utilities


def run_exp(output_file, config, policies, iterations):
    fp = open(output_file + '.csv', 'w')
    writer = csv.DictWriter(
        fp, fieldnames=['iteration', 'policy', 't', 'u_true', 'u_estimated'])
    writer.writeheader()

    # Create worker processes.
    def init_worker():
        """Function to make sure everyone happily exits on KeyboardInterrupt

        See https://stackoverflow.com/questions/1408356/
        keyboard-interrupts-with-pythons-multiprocessing-pool
        """
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    all_probs = [(i, calculate_probabilities(**config)) for
                 i in xrange(iterations)]
    pool = mp.Pool(initializer=init_worker)
    try:
        args = itertools.product(all_probs, policies)
        args = ({'probs_true': a[0][1][0],
                 'probs_estimated': a[0][1][1],
                 'policy': a[1],
                 'iteration': a[0][0]} for a in args)
        f = ft.partial(f_arg, ft.partial(f_expand, single_run))
        for d, (u_true, u_estimated) in pool.imap_unordered(f, args):
            print 'saving {} ({})'.format(d['policy'], d['iteration'])
            for t in xrange(len(u_true)):
                d_prime = dict((k, d[k]) for k in ['policy', 'iteration'])
                d_prime['t'] = t
                d_prime['u_true'] = u_true[t]
                d_prime['u_estimated'] = u_estimated[t]
                writer.writerow(d_prime)
            fp.flush()
        pool.close()
        pool.join()
    except KeyboardInterrupt:
        logger.warn('Control-C pressed')
        pool.terminate()
    finally:
        fp.close()

def main():
    parser = argparse.ArgumentParser(description='Run')
    parser.add_argument('--config', '-c', type=argparse.FileType('r'))
    parser.add_argument('--policies', '-p', type=argparse.FileType('r'))
    parser.add_argument('--iterations', '-i', type=int, default=10)
    parser.add_argument('--outfile', '-o', type=str, default='out')
    args = parser.parse_args()

    policies = json.load(args.policies)
    config = json.load(args.config)
    run_exp(args.outfile, config=config, policies=policies,
            iterations=args.iterations)


if __name__ == '__main__':
    main()
