docker run --rm \
--env PYTHONPATH=/source \
-p 127.0.0.1:8888:8888 \
-it \
-v $(pwd):/home/fenics \
quay.io/fenicsproject/stable:current \
'jupyter-notebook --ip=0.0.0.0 --NotebookApp.token="tokentoken"'

#--network host \     # gethostbyname failed
