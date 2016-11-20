.PHONY: doc

doc:
	pydoc -w `find socialkit -name '*.py'`
	mv *.html doc
