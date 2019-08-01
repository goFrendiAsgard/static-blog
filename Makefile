generate:
	jupyter nbconvert ./content/posts/*.ipynb --to markdown && hugo
serve:
	hugo server -D
deploy:
	./deploy.sh
