docker run --rm \
--env PYTHONPATH=/source \
-p 10.13.13.2:8888:8888 \
-it \
-v $(pwd):/home/fenics \
quay.io/fenicsproject/stable:current \
'jupyter-notebook --ip=0.0.0.0 --NotebookApp.token=""'

#--network host \     # gethostbyname failed
