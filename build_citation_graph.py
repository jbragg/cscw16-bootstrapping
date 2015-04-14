
import numpy as np
import csv
import random
import math
import operator
from os import sys
import pickle
import matplotlib.pyplot as plt

ALPHA = 10
BETA = 10

def sample_from_pareto(a, m, size):
    s = np.random.pareto(a, size) + m
    for i in range(len(s)):
        s[i] = round(s[i])

    return s


def sample_from_data(data_file_name, size):
    data_file = open(data_file_name)
    reader = csv.reader(data_file)
    data_list = []
    init = True
    for row in reader:
        if init:
            init = False
            continue
        data_list.append(int(row[1]))

    s = random.sample(data_list,size)
    return s

def sample_resource_papers(paper_count, resource_count):
    return random.sample(range(1,paper_count+1), resource_count)

def rich_get_richer(author_publication_count, p, link_count):

    paper_count = sum(author_publication_count)
    paper_id_list = range(1,paper_count+1)
    random.shuffle(paper_id_list)

    #the author of the paper
    paper_authors = {}
    
    #the papers the author writes
    author_papers = {}

    #the papers the paper cites 
    paper_links = {}
    
    #the papers that cite the paper
    reverse_paper_links = {}
    
    curr = 0

    for i in range(len(author_publication_count)):
        for j in range(author_publication_count[i]):
            paper_authors[paper_id_list[curr]] = i
            if i in author_papers:
                author_papers[i].append(paper_id_list[curr])
            else:
                author_papers[i] = [paper_id_list[curr]]
            curr+=1

    for i in range(1, paper_count+1):
        paper_links[i] = []
        reverse_paper_links[i] = []
        new_link = None
        if i == 1:
            continue
        for j in range(link_count):
            if random.random() < p:
                new_link = random.randint(1,i-1)

            else:
                new_ref = random.randint(1,i-1)
                if len(paper_links[new_ref]) > 0: 
                    new_link = random.sample(paper_links[new_ref],1)[0]

            if new_link and new_link not in paper_links[i]:
                paper_links[i].append(new_link)
                reverse_paper_links[new_link].append(i)

    return paper_authors, author_papers, paper_links, reverse_paper_links

def separate_resources(individual_author_papers, paper_links, resouces_id):
    author_citations = set()
    used_resources = []
    cited_resources = []
    not_cited_resources = []

    for p in individual_author_papers:
        #print p
        for l in paper_links[p]:
            author_citations.add(l)

    for r in resouces_id:
        if r in author_citations:
            if random.random() < 0.45:
                used_resources.append(r)
            else:
                cited_resources.append(r)
        else:
            not_cited_resources.append(r)
    return used_resources, cited_resources, not_cited_resources

def expected_contribution_utility(prob, existing_expected_contribution):

    ##question: should expectation be inside the power or outside the power##
    #original_util = pow(ALPHA,BETA*math.log1p(existing_expected_contribution))
    #new_util = pow(ALPHA,BETA*math.log1p(existing_expected_contribution+1))
    original_util = ALPHA*math.log1p(BETA*existing_expected_contribution)
    new_util = ALPHA*math.log1p(BETA*existing_expected_contribution+1)

    return prob*(new_util - original_util)

def compute_community_utility(expected_contributions):
    community_utility = 0
    for r in expected_contributions:
        community_utility += ALPHA*math.log1p(BETA*expected_contributions[r])
        #community_utility += pow(ALPHA,BETA*math.log1p(expected_contributions[r]))
    return round(community_utility,2)

def compute_contribution_prob(r, used_resources, cited_resources):

    if r in used_resources:
        contribution_prob = 0.059
    elif r in cited_resources:
        contribution_prob = 0.016
    else:
        contribution_prob = 0

    return contribution_prob    

def print_distribution(expected_contributions):
    contribution_list = []
    for key in expected_contributions:
        contribution_list.append(expected_contributions[key])

    data = np.array(contribution_list)

    hist,bins=np.histogram(data, bins=np.linspace(0,0.6,21))
    printing_hist = []
    for i in range(len(hist)):
        printing_hist.append(str(hist[i]))

    print '\t'.join(printing_hist)
    print bins
    return 

def assign_resources(author_papers, paper_links, reverse_paper_links, resources_id, method):


    requests = {}
    expected_contributions = {}
    for r in resources_id:
        expected_contributions[r] = 0

    print len(author_papers)

    expected_utilities = []
    contribution_probs = []
    community_utility_stages = []
    request_count_stages = []    

    if method == 'gg':
        # initialization
        # compute contribution probabilities for every author-resource pair
        for i in range(len(author_papers)):
            used_resources, cited_resources, not_cited_resources = separate_resources(author_papers[i], paper_links, resources_id)


            expected_utilities.append([])
            contribution_probs.append([])
            for j in range(len(resources_id)):
                r = resources_id[j]
                contribution_prob = compute_contribution_prob(r,used_resources, cited_resources)
                contribution_probs[i].append(contribution_prob)
                expected_utilities[i].append(expected_contribution_utility(contribution_prob,0))
        
        for count in range(len(author_papers)):
 
            if (count+1) %100 == 0:
                print count+1
                request_count_stages.append(count+1)
                community_utility_stages.append(compute_community_utility(expected_contributions))
            curr_max = -1
            curr_max_row = -1
            curr_max_col = -1


            curr_max_row_col = []
            #find the highest author-resource pair that has the highest expected utility
            for i in range(len(author_papers)):
                new_max = max(expected_utilities[i])
                if new_max > curr_max:
                    curr_max = new_max
                    curr_max_row_col = [[i, expected_utilities[i].index(max(expected_utilities[i]))]]
                    #curr_max_row = i
                    #curr_max_col = expected_utilities[i].index(max(expected_utilities[i]))
                elif new_max == curr_max:
                    curr_max_row_col.append([i, expected_utilities[i].index(max(expected_utilities[i]))])
                    '''
                    if random.random() > 0.5:
                        curr_max_row = i
                        curr_max_col = expected_utilities[i].index(max(expected_utilities[i]))
                    '''
            random_id = random.randint(0,len(curr_max_row_col)-1)

            curr_max_row = curr_max_row_col[random_id][0]
            curr_max_col = curr_max_row_col[random_id][1]

            expected_contributions[resources_id[curr_max_col]] += contribution_probs[curr_max_row][curr_max_col]

            #because the author already assign to a resource, it's not possible for it to assign to another resource
            #this can be optimized by creating a set of assigned authors, and check if they are in the list. Then, the system doesn't need to go through the row.
            for j in range(len(resources_id)):
                expected_utilities[curr_max_row][j] = -1


            #update expected contribution utility for every author resource pair
            for i in range(len(author_papers)):
                if expected_utilities[i][curr_max_col] > -1: 
                   expected_utilities[i][curr_max_col] = expected_contribution_utility(contribution_probs[i][curr_max_col],expected_contributions[resources_id[curr_max_col]])

        #print expected_utilities[0]
        print_results(expected_contributions,community_utility_stages,request_count_stages)
        return

    if method == 'sort':
        author_resources = {}
        author_cited_count = {}
        for i in range(len(author_papers)):
            used_resources, cited_resources, not_cited_resources = separate_resources(author_papers[i], paper_links, resources_id)

            author_resources[i] = [used_resources, cited_resources, not_cited_resources]
            author_cited_count[i] = len(used_resources)

            sorted_cited_count = sorted(author_cited_count.items(), key=operator.itemgetter(1))

            #sorted_cited_count.reverse()   

        i = 0        
        for pair in sorted_cited_count:

            if (i+1) %100 == 0:
                print i+1
                request_count_stages.append(i+1)
                community_utility_stages.append(compute_community_utility(expected_contributions))            
            i+=1

            used_resources, cited_resources, not_cited_resources = author_resources[pair[0]]
            expected_utilities = {}
            for r in resources_id:
                contribution_prob = compute_contribution_prob(r, used_resources, cited_resources)
                expected_utilities[r] = expected_contribution_utility(contribution_prob,expected_contributions[r])
            assigned_resource = max(expected_utilities.iteritems(), key=operator.itemgetter(1))[0]             

            expected_contributions[assigned_resource] += compute_contribution_prob(assigned_resource, used_resources, cited_resources)        
        
        print_results(expected_contributions,community_utility_stages,request_count_stages)
        return


    for i in range(len(author_papers)):
        if (i+1) %100 == 0:
            print i+1
            request_count_stages.append(i+1)
            community_utility_stages.append(compute_community_utility(expected_contributions))  
        used_resources, cited_resources, not_cited_resources = separate_resources(author_papers[i], paper_links, resources_id)
        if method == 'greedy':
            expected_utilities = {}
            for r in resources_id:
                contribution_prob = compute_contribution_prob(r, used_resources, cited_resources)
                expected_utilities[r] = expected_contribution_utility(contribution_prob,expected_contributions[r])
            assigned_resource = max(expected_utilities.iteritems(), key=operator.itemgetter(1))[0]            
        elif method == 'random':
            assigned_resource = random.sample(resources_id,1)[0]
        elif method == 'highest':
            if len(used_resources) > 0:
                assigned_resource = random.sample(used_resources,1)[0]
            elif len(cited_resources) > 0:
                assigned_resource = random.sample(cited_resources,1)[0]
            else:
                assigned_resource = random.sample(resources_id,1)[0]
        expected_contributions[assigned_resource] += compute_contribution_prob(assigned_resource, used_resources, cited_resources)


    print_results(expected_contributions,community_utility_stages,request_count_stages)    

    return

def print_results(expected_contributions,community_utility_stages,request_count_stages):

    print "community utility"
    print compute_community_utility(expected_contributions)
    print "contribution distribution"
    print_distribution(expected_contributions) 
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(request_count_stages, community_utility_stages, '-o')
    for i,j in zip(request_count_stages,community_utility_stages):
        ax.annotate(str(j),xy=(i,j))
    plt.show()
    return   

def generate_data():

    author_publication_count = sample_from_data('authors_publication.csv',1000)
    #print max(author_publication_count)
    paper_authors, author_papers, paper_links, reverse_paper_links = rich_get_richer(author_publication_count, 0.5, 10)

    print len(paper_authors)

    resources_id = sample_resource_papers(len(paper_authors),10000)

    data = {}

    data['author_papers'] = author_papers
    data['paper_links'] = paper_links
    data['reverse_paper_links'] = reverse_paper_links
    data['resources_id'] = resources_id

    fout = open('citation_graph','w')

    pickle.dump(data, fout)
    print "finish generating and dumping data"
    fout.close()
    return author_papers, paper_links, reverse_paper_links, resources_id

def load_data():

    fin = open('citation_graph')
    data = pickle.load(fin)
    fin.close()
    print "finish loading data"

    return data['author_papers'], data['paper_links'], data['reverse_paper_links'], data['resources_id']

def main():
    
    method = sys.argv[1]

    loading = sys.argv[2]

    if loading != '1':
        author_papers, paper_links, reverse_paper_links, resources_id = generate_data()
    else:
        author_papers, paper_links, reverse_paper_links, resources_id = load_data()

    assign_resources(author_papers, paper_links, reverse_paper_links, resources_id, method)

    #author_publication_count = sample_from_pareto(0.4481292, 1., 1000)
    #print max(author_publication_count)

if __name__ == '__main__':
    main()