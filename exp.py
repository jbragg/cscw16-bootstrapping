"""exp.py"""

from __future__ import division
import math
import random
import collections
import operator
import itertools
import copy
from build_citation_graph import generate_data

ALPHA = 1
BETA = 100

#-------- helpers ---------
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
                       resources_modified=None, verbose=False):
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
        u[r] = expectation_over_powerset(probs_r, f_utility)
    if verbose:
        print 'max size: {}'.format(max_requests_by_resource)
    return u

def single_run(probs, policy='greedy', ignore_uncited=False):
    """Execute a single policy run

    Policies are as follows:
        'random': Select a random author, assign that author a random resource.

    Args:
        probs:              Dictionary from (author, resource) to probability
                            of contribution.
        policy:
        ignore_uncited:     Don't use uncited items, since they have
                            probability 0.

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
            print len(authors_rand), utilities[-1]
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
            print len(authors_rand), utilities[-1]
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
            print len(set(a for a, r in marginal_utilities)), utilities[-1]
    else: 
        raise NotImplementedError

    return utilities

def main():
    probs = calculate_probabilities()
    #print single_run(probs, policy='random')
    print single_run(probs, policy='dt')

    #print len(set(a for a, r in probs))
    #print len(set(r for a, r in probs))
    #print len(probs)
    #print collections.Counter(probs.values())

    #print utility(probs)
    #print utility(random.sample(probs, int(round(0.9 * len(probs)))))
    #print utility(random.sample(probs, int(round(0.8 * len(probs)))))
    #print utility(random.sample(probs, int(round(0.7 * len(probs)))))
    #print utility(random.sample(probs, int(round(0.6 * len(probs)))))
    #print utility(random.sample(probs, int(round(0.5 * len(probs)))))
    #print utility(random.sample(probs, int(round(0.4 * len(probs)))))
    #print utility(random.sample(probs, int(round(0.3 * len(probs)))))
    #print utility(random.sample(probs, int(round(0.2 * len(probs)))))
    #print utility(random.sample(probs, int(round(0.1 * len(probs)))))
    #print utility(random.sample(probs, int(round(0 * len(probs)))))



if __name__ == '__main__':
    main()
