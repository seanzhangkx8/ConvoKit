from .speaker import Speaker
from .corpusUtil import warn

class User(Speaker):
    def __init__(self, *args, **kwargs):
        warn("The User class is deprecated. Use the Speaker class instead.")

        super().__init__(*args, **kwargs)