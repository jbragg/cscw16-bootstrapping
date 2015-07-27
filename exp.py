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
    left = max(0, u - std * 1.96 / 2)
    right = min(1, u + std * 1.96 / 2)

    endpoints = np.linspace(left, right, n + 1)
    midpoints = []
    for p1, p2 in zip(endpoints[:-1], endpoints[1:]):
        midpoints.append((p2 + p1) / 2)

    v = np.random.normal(loc=u, scale=std, size=None)

    distances = [abs(p - v) for p in midpoints]
    min_distance = min(distances)
    closest_midpoints = [p for i, p in enumerate(midpoints) if
                         distances[i] == min_distance]
    return random.choice(closest_midpoints)

def f_arg(f, d):
    """Return pair with arg and function value"""
    return (d, f(d))

#def f_no_arg(f, arg):
#    """Return function that ignores arg keyword"""
#    return lambda d: f(dict((k,v) for k in d if k != arg))

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
        probs_items[d[k]].append(k)

    probs = sorted(probs_items)
    v = 0
    for tup in itertools.product(*(xrange(len(probs_items[k]) + 1) for
                                   k in probs)):
        pr = 1
        items = []
        for p, n in zip(probs, tup):
            pr *= p ** n * (1 - p) ** (len(probs_items[p]) - n)
            items += probs_items[p][:n]
        v += pr * f(items)
    return v

#---------- main functions ---------
def calculate_probabilities(n_authors=1000, n_resources=1000,
                            n_links=10, p_cite=0.5, p_use=0.45,
                            p_response_used=0.059,
                            p_response_unused=0.016,
                            p_response_uncited=0.0):
    """Return mapping from author-resource pairs to probability of response"""
    data = generate_data(n_authors, n_resources, n_links, p_cite, p_use)
    authors = data['author_papers'].keys()
    resources = data['resources_id']

    probs = dict()
    for a in authors:
        used, unused, uncited = data['author_resources'][a] 
        for r in used:
            probs[a, r] = p_response_used
        for r in unused:
            probs[a, r] = p_response_unused
        for r in uncited:
            probs[a, r] = p_response_uncited
    return probs

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

    f = lambda lst: ALPHA * math.log1p(BETA * len(lst))
    return sum(f(d[r]) for r in d)

def expected_utilities(probs, utilities_by_resource={},
                       resources_modified=None, verbose=False,
                       exploit_bounded=True):
    """Expected utilities of contributions resulting from a set of requests.

    Recalculate only resources in resources_modified if
    resources_modified is not None.
    
    Args:
        probs:                  Dictionary from (author, resource) pairs to
                                probability of contribution.
        utilities_by_resource:  Dictionary of old utilities by resource.
        resources_modified:     List of resources modified since last result.

    Returns:
        Dictionary from resource to expected utility

    """
    if resources_modified is not None:
        resources = resources_modified
    else:
        resources = list(set(r for a, r in probs))
    max_requests_by_resource = 0
    u = copy.copy(utilities_by_resource)
    for r in resources:
        probs_r = dict((k, probs[k]) for k in probs if k[1] == r)
        max_requests_by_resource = max(max_requests_by_resource, len(probs_r))
        if exploit_bounded:
            u[r] = expectation_bounded(probs_r, f_utility)
        else:
            u[r] = expectation_over_powerset(probs_r, f_utility)
    if verbose:
        logger.info('max size: {}'.format(max_requests_by_resource))
    return u

def single_run(probs, policy='greedy', ignore_uncited=False, **args):
    """Execute a single policy run

    Policies are as follows:
        'random': Select a random author, assign that author a random resource.

    Args:
        probs:              Dictionary from (author, resource) to probability
                            of contribution.
        policy:
        ignore_uncited:     Don't use uncited items, since they have
                            probability 0. NOT IMPLEMENTED.

    """
    authors = set(a for a, r in probs)
    resources = list(set(r for a, r in probs))
    s = dict()  # Final set of requests to issue
    utilities_by_resource = dict()
    utilities = []
    utilities.append(sum(utilities_by_resource.itervalues()))
    if policy == 'random':
        authors_rand = list(authors)
        random.shuffle(authors_rand)
        while len(authors_rand) > 0:
            a = authors_rand.pop()
            next_request = a, random.choice(resources)
            s[next_request] = probs[next_request]
            utilities_by_resource = expected_utilities(
                s, utilities_by_resource, [next_request[1]])
            #utilities_by_resource = expected_utilities(s)
            utilities.append(sum(utilities_by_resource.itervalues()))
            logger.info('{}: {}'.format(len(authors_rand), utilities[-1]))
    elif policy == 'greedy':
        authors_rand = list(authors)
        random.shuffle(authors_rand)
        while len(authors_rand) > 0:
            a = authors_rand.pop()
            possible_requests = [(a, r) for r in resources]
            max_prob = max(probs[x] for x in possible_requests)
            requests_max_prob = [x for x in possible_requests if
                                 probs[x] == max_prob]
            next_request = random.choice(requests_max_prob)
            # TODO: Duplicate code.
            s[next_request] = probs[next_request]
            utilities_by_resource = expected_utilities(
                s, utilities_by_resource, [next_request[1]])
            utilities.append(sum(utilities_by_resource.itervalues()))
            logger.info('{}: {}'.format(len(authors_rand), utilities[-1]))
    elif policy == 'dt':
        marginal_utilities = dict((x, None) for x in probs)
        next_request = None
        while len(marginal_utilities) > 0:
            last_utility = utilities[-1]
            if next_request is None:
                needs_update = marginal_utilities
            else:
                needs_update = [x for x in marginal_utilities if
                                x[1] == next_request[1]]
            for x in needs_update:
                s_prime = copy.copy(s)
                s_prime[x] = probs[x]
                utilities_by_resource_prime = expected_utilities(
                    s_prime, utilities_by_resource, [x[1]])
                marginal_utilities[x] = sum(
                    utilities_by_resource_prime.itervalues()) - utilities[-1]
            max_marginal_utility = marginal_utilities[
                max(marginal_utilities, key=marginal_utilities.get)]
            requests_max_marginal_utility = [
                x for x in marginal_utilities if
                marginal_utilities[x] == max_marginal_utility]
            next_request = random.choice(requests_max_marginal_utility)
            utilities.append(utilities[-1] + marginal_utilities[next_request])

            # Remove other candidate requests for selected author.
            for k in [k for k in marginal_utilities if
                      k[0] == next_request[0]]:
                del marginal_utilities[k]
            logger.info('{}: {}'.format(
                len(set(a for a, r in marginal_utilities)), utilities[-1]))
    else: 
        raise NotImplementedError

    return utilities


def run_exp(output_file, policies=['random', 'greedy', 'dt'], iterations=100):
    fp = open(output_file, 'w')
    writer = csv.DictWriter(fp, fieldnames=['iteration', 'policy', 't', 'v'])
    writer.writeheader()

    # Create worker processes.
    def init_worker():
        """Function to make sure everyone happily exits on KeyboardInterrupt

        See https://stackoverflow.com/questions/1408356/
        keyboard-interrupts-with-pythons-multiprocessing-pool
        """
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    pool = mp.Pool(initializer=init_worker)
    all_probs = [(i, calculate_probabilities()) for i in xrange(iterations)]
    try:
        args = itertools.product(policies, all_probs)
        args = ({'probs': a[1][1],
                 'policy': a[0],
                 'iteration': a[1][0]} for a in args)
        f = ft.partial(f_arg, ft.partial(f_expand, single_run))
        for d, v in pool.imap_unordered(f, args):
            print 'saving {} ({})'.format(d['policy'], d['iteration'])
            del d['probs']
            for t in xrange(len(v)):
                d_prime = copy.copy(d)
                d_prime['t'] = t
                d_prime['v'] = v[t]
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
    parser.add_argument('--policies', '-p', type=str, nargs='*')
    parser.add_argument('--iterations', '-i', type=int, default=10)
    parser.add_argument('--outfile', '-o', type=str, default='out.csv')
    args = parser.parse_args()

    run_exp(args.outfile, policies=args.policies, iterations=args.iterations)


if __name__ == '__main__':
    main()
