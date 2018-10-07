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

def children(comment):
    l = [{k: v for k, v in d.items() if k != "children"}
        for d in comment["children"]]
    for child in comment["children"]:
        l += children(child)
    return l

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
        #replies = children(comment, []) #generates list of children 
        for child in children(comment): #do some formatting
            if "body" not in child: continue
            child.pop('name') 
            #child.pop('children')
            child.pop('subreddit')
            child.pop('subreddit_id')
            d = {}
            d['root'] = comment['id']
            d['id'] = child['id']
            d['reply-to'] = child.pop('parent_id')
            d['text'] = child.pop('body')
            d['timestamp'] = child.pop('created_utc')
            d['user'] = child.pop('author')
            d['user-info'] = child
            reddit_convos.append(d) #append to convokit
        addcomment = {}#dict(comment)
        addcomment['root'] = comment['id'] #cleanup comment
        addcomment['id'] = comment.pop('id')
        addcomment['text'] = comment.pop('selftext')
        addcomment['timestamp'] = comment.pop('created_utc')
        addcomment['user'] = comment.pop('author')
        comment.pop('children')
        comment.pop('created')
        addcomment['user-info'] = comment
        reddit_convos.append(addcomment) #add the original comment

reddit_convos = del_dups(reddit_convos)
reddit_convos = [d for d in reddit_convos if "text" in d]

with open('reddit-convos.json', 'w') as fp:
    json.dump(reddit_convos, fp, indent=2)
