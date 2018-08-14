# Data format for [ConvoKit](http://convokit.cornell.edu/)

ConvoKit expects a standard json-based format.  A corpus is a list of json objects, each representing a comment.

A sample dataset containing the comments from three conversations is available [here](http://zissou.infosci.cornell.edu/socialkit/datasets/wiki-corpus/sample.json)

The mandatory fields for each comment-json are: "id", "reply-to", "root", "text", "user".   Other fields can be added optionally depending on a particular data and intended use (e.g., "timestamp","user-info").

Here is a description of the format of a json object for a comment in the sample above (optional fields are shown in squared brackets)

{

    "id": "2",  ->  id of this comment
    "reply-to": "1",  -> id of the comment to which this comment replies to (field is absent if this comment is not a reply)
    "root": "1", -> id of conversation (i.e., the id of the root comment in the conversation)
    "text": "Nagma'a site, at least that filmography page, looks like taken from Wikipedia itself. I didn't remove it from time being, but I don't think it's true. ", -> text of the comment (with escaped characters)
    "timestamp": "1.310746500E09", [-> time of comment]

    "user": "Johannes003", -> username of comment author

    "user-info": {  [-> dictionary with  user metadata]

      "gender": "unknown", [ -> gender of user ]

      "is-admin": false, [ -> admin role]

      "user-edit-count": "8283" [-> edit count]
      }
 }    
      
