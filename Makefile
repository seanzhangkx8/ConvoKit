.PHONY: doc

doc:
	mkdir -p doc
	pydoc -w `find socialkit -name '*.py'`
	mv *.html doc
