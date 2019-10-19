init-conda-env:
	conda env create
save-conda-env:
	conda env export > generated-environment.yml
generate:
	./ipynb-to-markdown.sh && hugo
serve:
	./ipynb-to-markdown.sh && hugo server -D
deploy:
	./ipynb-to-markdown.sh && hugo && ./deploy.sh
start-jupyter:
	cd content/notebooks && jupyter notebook
