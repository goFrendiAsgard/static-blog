generate:
	./ipynb-to-markdown.sh && hugo
serve:
	./ipynb-to-markdown.sh && hugo server -D
deploy:
	./ipynb-to-markdown.sh && hugo && ./deploy.sh
jupyter:
	cd content/posts && source /opt/anaconda/bin/activate root && jupyter notebook
