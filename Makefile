generate:
	./ipynb-to-markdown.sh && hugo
serve:
	./ipynb-to-markdown.sh && hugo server -D
deploy:
	./ipynb-to-markdown.sh && hugo && ./deploy.sh
jupyter:
	cd content/notebooks && source /opt/anaconda/bin/activate root && jupyter notebook
