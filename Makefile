IP=$(shell ifconfig | grep 192.168 | awk '{print $$2}')
.PHONY: dev udpate 

dev:
	@echo "Listen to http://$(IP):4000"
	@bundle exec jekyll serve --host 0.0.0.0 --trace 
	

update:
	@python3 update_post.py 


