# original author: Tiffany Wang (tw292@cornell.edu)

import json
import os

location = os.getcwd()
subs_dict = {}
for file in os.listdir(location+'/reddit-data'):
    if file.endswith(".jsonlist"):
        with open('reddit-data/'+file) as f:
            subs_dict[file[:-9]] = []
            for line in f:
                line_json = json.loads(line)
                if line_json not in subs_dict[file[:-9]]:
                    subs_dict[file[:-9]].append(line_json) #remove duplicates

def children(comment, newdict=[]):
    """Returns all children of a root comment recursively.
    Parameters: comment (root comment), newdict (list)
    Return: tuple, where tuple[1] = newdict, containing children of root comment 
    - NOT including root comment itself."""
    if comment['children'] == []:
        return comment, newdict
    else:
        for child in comment['children']:
            newdict.append(child)
            return children(child, newdict), newdict

def del_dups(convos):
    """Delete duplicates in created convokit object.
    Parameter: Dataset in Convokit format.
    Returns: Deduplicated list"""
    ids = set()
    unique = []
    for x in convos: 
        if x['id'] not in ids:
            ids.add(x['id'])
            unique.append(x)
    return unique

##create new json list
reddit_convos = []
for subreddit in subs_dict:
    for comment in subs_dict[subreddit]:
        replies = children(comment, []) #generates list of children 
        for child in replies[1]: #do some formatting
            child.pop('name') 
            child.pop('children')
            child.pop('subreddit')
            child.pop('subreddit_id')
            d = {}
            d['root'] = comment['id']
            d['id'] = comment['id']
            d['reply-to'] = child.pop('parent_id')
            d['text'] = child.pop('body')
            d['timestamp'] = child.pop('created_utc')
            d['user-info'] = child
            reddit_convos.append(child) #append to convokit
        addcomment = {}#dict(comment)
        addcomment['root'] = comment.pop('name') #cleanup comment
        addcomment['id'] = comment['id']
        addcomment['text'] = comment.pop('selftext')
        addcomment['timestamp'] = comment.pop('created_utc')
        comment.pop('children')
        comment.pop('created')
        addcomment['user-info'] = comment
        reddit_convos.append(addcomment) #add the original comment

reddit_convos = del_dups(reddit_convos)
reddit_convos = [d for d in reddit_convos if "text" in d]

with open('reddit_convos.json', 'w') as fp:
    json.dump(reddit_convos, fp, indent=2)
